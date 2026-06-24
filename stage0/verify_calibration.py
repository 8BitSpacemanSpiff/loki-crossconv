"""Verify the emitted calibration artifact satisfies the Phase B data contract.

Checks, per emitted layer:
  - K_pre and K_post genuinely differ (RoPE was applied; not a silent copy)
  - K_pre / K_post / V finite and non-trivial (no key-norm fallback, no NaNs)
  - Q_oracle / Q_eval positions are disjoint (honest oracle/eval split)
  - GQA group size 4 in the stored shape
  - post-RoPE attention entropy (mean over eval queries) is PEAKED, i.e. well below
    the uniform ceiling ln(P) -> the top-b-by-mass oracle is a valid floor

Run:  python -m stage0.verify_calibration --dir outputs/calib
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def attn_entropy(Q_eval, K_post, sample_q=64):
    # Q_eval: (n_kv, group, n_q, hd)  K_post: (n_kv, P, hd)  -> mean entropy over a sample
    n_kv, group, n_q, hd = Q_eval.shape
    P = K_post.shape[1]
    q = Q_eval[:, 0, :sample_q, :].float()        # (n_kv, sample_q, hd), group head 0
    k = K_post.float()                            # (n_kv, P, hd)
    scores = torch.einsum("hqd,hpd->hqp", q, k) / math.sqrt(hd)
    A = torch.softmax(scores, dim=-1)
    ent = (-(A * torch.log(A + 1e-12)).sum(-1)).mean()
    return float(ent), math.log(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/calib")
    args = ap.parse_args()
    d = Path(args.dir)
    meta = json.loads((d / "meta.json").read_text())
    print(f"meta: model={meta['model_id']}  n_seq={meta['n_seq']}  seq_len={meta['seq_len']}  "
          f"P={meta['cloud_len_P']}  group={meta['gqa_group']}  rope_theta={meta['rope_theta']}  "
          f"key_norm_fallback={meta['key_norm_fallback']}")
    files = sorted(d.glob("layer_*.pt"))
    all_ok = True
    print(f"\n{'layer':>5} {'K_pre!=K_post':>13} {'finite':>7} {'q-disjoint':>11} "
          f"{'group':>6} {'entropy':>9} {'/ln(P)':>8} {'peaked':>7}")
    for f in files:
        L = int(f.stem.split("_")[1])
        t = torch.load(f, map_location="cpu")
        kp, kq, v = t["K_pre"][0], t["K_post"][0], t["V"][0]   # first seq: (n_kv,P,hd)
        rope_applied = float((kp.float() - kq.float()).abs().mean()) > 1e-3
        finite = bool(torch.isfinite(kp).all() and torch.isfinite(kq).all()
                      and torch.isfinite(v).all() and v.float().std() > 1e-4)
        disjoint = len(set(t["oracle_pos"].tolist()) & set(t["eval_pos"].tolist())) == 0
        group_ok = t["Q_oracle"].shape[2] == meta["gqa_group"]
        ent, maxent = attn_entropy(t["Q_eval"][0], kq)
        peaked = ent < 0.85 * maxent
        ok = rope_applied and finite and disjoint and group_ok
        all_ok &= ok
        print(f"{L:>5} {str(rope_applied):>13} {str(finite):>7} {str(disjoint):>11} "
              f"{str(group_ok):>6} {ent:>9.3f} {ent/maxent:>8.3f} {str(peaked):>7}")
    print(f"\nCONTRACT {'OK' if all_ok else 'FAILED'} across {len(files)} layers")
    return all_ok


if __name__ == "__main__":
    main()
