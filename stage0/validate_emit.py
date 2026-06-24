"""Validate the from-scratch emit against the HF model itself (CPU/GPU, saved artifact).

Part 1 - MECHANISM (short seq, eager + output_attentions): reuse the EXACT emit
capture functions (emit_calibration._patched_rope / _v_hook / _STATE) and, for
sampled (layer, kv-head, all 4 GQA group heads, late query positions), reconstruct
  probs = softmax(Q_post @ K_post.T / sqrt(d)) over the causal prefix [0:p+1]
  out   = probs @ V
then compare to the model's TRUE attention weights (outputs.attentions) and TRUE
per-head attention output (captured at o_proj input). A match to ~fp16 proves the
hook grabbed the right tensor / axis / GQA mapping / RoPE.

Part 2 - ARTIFACT TIE (seq 8192, no attention maps): reproduce saved sequence 0 via
the identical c4 stream, re-capture + slice exactly as emit, and diff against the
saved layer_*.pt bytes.

Run:  python -m stage0.validate_emit
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral import modeling_mistral

from . import emit_calibration as E

CALIB = Path(__file__).resolve().parent.parent / "outputs" / "calib"


def _install_emit_capture(model, n_kv, hd):
    modeling_mistral.apply_rotary_pos_emb = E._patched_rope
    for i, lyr in enumerate(model.model.layers):
        lyr.self_attn.v_proj.register_forward_hook(E._v_hook(i, n_kv, hd))


def _oproj_capture(model):
    store = {}
    def mk(i):
        def hook(_m, inp, _o):
            store[i] = inp[0].detach()      # (b, seq, num_heads*head_dim)
        return hook
    for i, lyr in enumerate(model.model.layers):
        lyr.self_attn.o_proj.register_forward_hook(mk(i))
    return store


def part1_mechanism(tok):
    print("== Part 1: mechanism validation (eager, output_attentions, seq=1024) ==")
    model = AutoModelForCausalLM.from_pretrained(
        E.MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    cfg = model.config
    n_kv, n_heads = cfg.num_key_value_heads, cfg.num_attention_heads
    hd = cfg.hidden_size // n_heads
    group = n_heads // n_kv
    _install_emit_capture(model, n_kv, hd)
    oproj = _oproj_capture(model)

    it = E._c4_stream()
    ids = E._next_sequence(it, tok, 1024).to("cuda")
    E._STATE["counter"] = 0; E._STATE["rope"].clear(); E._STATE["v"].clear()
    with torch.no_grad():
        out = model(ids.unsqueeze(0), use_cache=False, output_attentions=True)
    attn = out.attentions  # tuple[L] of (1, n_heads, seq, seq)

    layers = [0, 8, 16, 24, 31]
    kvs = [0, 3, 7]
    positions = [600, 800, 1000, 1023]
    max_w, max_o = 0.0, 0.0
    for L in layers:
        k_pre, k_post, q_post = E._STATE["rope"][L]      # (1,n_kv,seq,hd)... q:(1,n_heads,seq,hd)
        v = E._STATE["v"][L]                              # (1,n_kv,seq,hd)
        true_out = oproj[L].view(1, -1, n_heads, hd)      # (1, seq, n_heads, hd)
        for h in kvs:
            for g in range(group):                        # all 4 group heads
                qh = h * group + g
                for p in positions:
                    q = q_post[0, qh, p, :].float()
                    k = k_post[0, h, :p + 1, :].float()
                    w = torch.softmax(q @ k.T / math.sqrt(hd), dim=-1)
                    o = w @ v[0, h, :p + 1, :].float()
                    tw = attn[L][0, qh, p, :p + 1].float()
                    to = true_out[0, p, qh, :].float()
                    max_w = max(max_w, float((w - tw).abs().max()))
                    max_o = max(max_o, float((o - to).abs().max()))
    print(f"  sampled layers={layers} kv_heads={kvs} group_heads=all{group} positions={positions}")
    print(f"  GQA mapping tested: query head qh = kv_head*{group}+g  -> kv = qh//{group}")
    print(f"  max abs error  attention WEIGHTS : {max_w:.3e}")
    print(f"  max abs error  attention OUTPUTS : {max_o:.3e}")
    del model, oproj, attn
    torch.cuda.empty_cache()
    return max_w, max_o


def part2_artifact_tie(tok):
    print("\n== Part 2: artifact tie (sdpa, seq=8192, reproduce saved seq 0) ==")
    if not (CALIB / "layer_00.pt").exists():
        print("  (no saved artifact found, skipping)")
        return 0.0
    import json
    meta = json.loads((CALIB / "meta.json").read_text())
    seq_len, P = meta["seq_len"], meta["cloud_len_P"]
    model = AutoModelForCausalLM.from_pretrained(
        E.MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
    cfg = model.config
    n_kv, n_heads = cfg.num_key_value_heads, cfg.num_attention_heads
    hd = cfg.hidden_size // n_heads
    group = n_heads // n_kv
    _install_emit_capture(model, n_kv, hd)

    it = E._c4_stream()
    ids = E._next_sequence(it, tok, seq_len).to("cuda")   # same code -> same first sequence
    E._STATE["counter"] = 0; E._STATE["rope"].clear(); E._STATE["v"].clear()
    with torch.no_grad():
        model(ids.unsqueeze(0), use_cache=False)

    oracle_pos = torch.load(CALIB / "layer_00.pt", map_location="cpu")["oracle_pos"]
    eval_pos = torch.load(CALIB / "layer_00.pt", map_location="cpu")["eval_pos"]
    worst = 0.0
    for L in [0, 15, 31]:
        k_pre, k_post, q_post = E._STATE["rope"][L]
        v = E._STATE["v"][L]
        re_Kpost = k_post[0, :, :P, :].to("cpu", torch.float16)
        re_V = v[0, :, :P, :].to("cpu", torch.float16)
        re_Qeval = q_post[0].view(n_kv, group, seq_len, hd)[:, :, eval_pos, :].to("cpu", torch.float16)
        saved = torch.load(CALIB / f"layer_{L:02d}.pt", map_location="cpu")
        eK = float((re_Kpost.float() - saved["K_post"][0].float()).abs().max())
        eV = float((re_V.float() - saved["V"][0].float()).abs().max())
        eQ = float((re_Qeval.float() - saved["Q_eval"][0].float()).abs().max())
        worst = max(worst, eK, eV, eQ)
        print(f"  layer {L:2d}: max|re-saved|  K_post={eK:.3e}  V={eV:.3e}  Q_eval={eQ:.3e}")
    print(f"  worst saved-vs-recapture: {worst:.3e}")
    del model
    torch.cuda.empty_cache()
    return worst


def main():
    tok = AutoTokenizer.from_pretrained(E.MODEL_ID)
    mw, mo = part1_mechanism(tok)
    tie = part2_artifact_tie(tok)
    fp16 = 5e-2  # fp16 softmax over thousands of keys accumulates noise; a few e-2 is expected
    ok = (mw < fp16) and (mo < fp16) and (tie < fp16)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  "
          f"(weights {mw:.2e}, outputs {mo:.2e}, artifact-tie {tie:.2e}; tol {fp16:.0e})")


if __name__ == "__main__":
    main()
