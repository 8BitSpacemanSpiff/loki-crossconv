"""Phase B — emit the per-head Mistral calibration artifact to DISK.

There is no `--emit-rq` path in this checkout, so this extracts the full data
contract from scratch via hooks. NO scoring happens here (so the documented
key-norm fallback is structurally impossible -- we dump raw projections).

Per (layer, KV-head) we capture, on real Mistral-7B-v0.2 over c4:
  K_pre   : pre-RoPE  keys   (seq, head_dim)  -> selectors + geometry (content, hard rule 5)
  K_post  : post-RoPE keys   (seq, head_dim)  -> Δ and oracle (real attention)
  V       : values           (seq, head_dim)  -> Δ
  Q_post  : post-RoPE queries for the 4 query heads in this KV-head's GQA group,
            split into DISJOINT Q_oracle / Q_eval (oracle picks top-b by mass over
            Q_oracle; Δ for all selectors is over Q_eval).

Causality without masking: the key cloud is keys[0:P] with P = seq - n_q_total, and
the oracle/eval queries are the last n_q_total positions -- every one causally
attends to ALL of [0:P], so the measured cloud is identical and complete for every
query, no mask needed.

Capture mechanism: apply_rotary_pos_emb is a module-global called once per layer in
layer order, so a call counter maps call -> layer index; v_proj is hooked per layer.

Run (smoke):  python -m stage0.emit_calibration --smoke
Run (full):   python -m stage0.emit_calibration --seq-len 8192 --n-seq 24 --out-dir outputs/calib
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral import modeling_mistral

MODEL_ID = "mistral-community/Mistral-7B-v0.2"  # canonical ungated Mistral-7B-v0.2 base

# ---- capture state -----------------------------------------------------------
_STATE = {"counter": 0, "rope": {}, "v": {}}
_ORIG_ROPE = modeling_mistral.apply_rotary_pos_emb


def _patched_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_post, k_post = _ORIG_ROPE(q, k, cos, sin, position_ids, unsqueeze_dim)
    idx = _STATE["counter"]
    _STATE["counter"] += 1
    _STATE["rope"][idx] = (k.detach(), k_post.detach(), q_post.detach())  # k_pre,k_post,q_post
    return q_post, k_post


def _v_hook(idx, n_kv, hd):
    def hook(_mod, _inp, out):
        b, s, _ = out.shape
        _STATE["v"][idx] = out.detach().view(b, s, n_kv, hd).transpose(1, 2)  # (b,n_kv,seq,hd)
    return hook


def _c4_stream():
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    return iter(ds)


def _next_sequence(it, tok, seq_len):
    bos = tok.bos_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else bos
    ids = [bos]
    while len(ids) < seq_len:
        doc = next(it)["text"]
        ids += tok(doc, add_special_tokens=False)["input_ids"] + [eos]
    return torch.tensor(ids[:seq_len], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--n-q-total", type=int, default=256, help="last positions used as queries")
    ap.add_argument("--out-dir", type=str, default="outputs/calib")
    ap.add_argument("--layers", type=str, default="all", help="'all' or e.g. '0,15' for smoke")
    args = ap.parse_args()
    if args.smoke:
        args.seq_len, args.n_seq, args.out_dir = 2048, 2, "outputs/calib_smoke"
        args.layers = "0,15"

    t0 = time.time()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to(dev).eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    n_heads = cfg.num_attention_heads
    hd = cfg.hidden_size // n_heads
    group = n_heads // n_kv
    P = args.seq_len - args.n_q_total          # key-cloud length
    n_qo = args.n_q_total // 2                 # disjoint oracle / eval halves
    print(f"[{time.time()-t0:.0f}s] model loaded: layers={n_layers} kv={n_kv} heads={n_heads} "
          f"head_dim={hd} group={group} rope_theta={getattr(cfg,'rope_theta',None)} "
          f"sliding={getattr(cfg,'sliding_window',None)}", flush=True)
    assert group == 4, f"expected GQA group 4, got {group}"

    # install captures
    modeling_mistral.apply_rotary_pos_emb = _patched_rope
    for i, lyr in enumerate(model.model.layers):
        lyr.self_attn.v_proj.register_forward_hook(_v_hook(i, n_kv, hd))

    want_layers = list(range(n_layers)) if args.layers == "all" \
        else [int(x) for x in args.layers.split(",")]
    # per-layer accumulators across sequences
    acc = {L: {k: [] for k in ("K_pre", "K_post", "V", "Q_oracle", "Q_eval")} for L in want_layers}
    q_pos = torch.arange(P, args.seq_len)
    oracle_pos, eval_pos = q_pos[0::2], q_pos[1::2]      # disjoint
    assert len(set(oracle_pos.tolist()) & set(eval_pos.tolist())) == 0

    it = _c4_stream()
    for s in range(args.n_seq):
        ids = _next_sequence(it, tok, args.seq_len).to(dev)
        _STATE["counter"] = 0
        _STATE["rope"].clear()
        _STATE["v"].clear()
        with torch.no_grad():
            model(ids.unsqueeze(0), use_cache=False)
        for L in want_layers:
            k_pre, k_post, q_post = _STATE["rope"][L]      # (1,n_kv,seq,hd),(1,n_kv,seq,hd),(1,n_heads,seq,hd)
            v = _STATE["v"][L]                              # (1,n_kv,seq,hd)
            acc[L]["K_pre"].append(k_pre[0, :, :P, :].to("cpu", torch.float16))
            acc[L]["K_post"].append(k_post[0, :, :P, :].to("cpu", torch.float16))
            acc[L]["V"].append(v[0, :, :P, :].to("cpu", torch.float16))
            qg = q_post[0].view(n_kv, group, args.seq_len, hd)         # (n_kv,group,seq,hd)
            acc[L]["Q_oracle"].append(qg[:, :, oracle_pos, :].to("cpu", torch.float16))
            acc[L]["Q_eval"].append(qg[:, :, eval_pos, :].to("cpu", torch.float16))
        print(f"[{time.time()-t0:.0f}s] seq {s+1}/{args.n_seq} done (len {args.seq_len})", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # sanity stats from the first stored layer/sequence
    L0 = want_layers[0]
    kp0 = acc[L0]["K_pre"][0]
    sane = bool(torch.isfinite(kp0).all()) and float(kp0.float().std()) > 1e-4
    for L in want_layers:
        d = {k: torch.stack(v, 0) for k, v in acc[L].items()}  # (n_seq, ...)
        d["oracle_pos"] = oracle_pos
        d["eval_pos"] = eval_pos
        torch.save(d, out / f"layer_{L:02d}.pt")
    meta = {
        "model_id": MODEL_ID, "note_model_host_substitution":
            "mistralai/Mistral-7B-v0.2 was never published; this is the canonical ungated "
            "mistral-community/Mistral-7B-v0.2 base conversion (identical weights).",
        "n_layers_emitted": len(want_layers), "layers": want_layers,
        "n_seq": args.n_seq, "seq_len": args.seq_len, "cloud_len_P": P,
        "n_kv_heads": n_kv, "n_q_heads": n_heads, "gqa_group": group, "head_dim": hd,
        "n_q_oracle": int(len(oracle_pos)), "n_q_eval": int(len(eval_pos)),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "sliding_window": getattr(cfg, "sliding_window", None),
        "dtype": "float16", "key_norm_fallback": False,
        "shapes": {"K_pre/K_post/V": [args.n_seq, n_kv, P, hd],
                   "Q_oracle/Q_eval": [args.n_seq, n_kv, group, n_qo, hd]},
        "sanity_first_layer_finite_and_nontrivial": sane,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{time.time()-t0:.0f}s] WROTE {len(want_layers)} layer files to {out}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
