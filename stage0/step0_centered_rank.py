"""Phase D STEP 0 — does the rank-1 K_pre finding survive mean-centering?

resume_checks (a) found mean effective rank ~1.4/128 on the UNCENTERED second moment
M = K^T K. That underpins the whole "volume-diversity (logdet) is structurally wrong"
framing. But an uncentered second moment is inflated by any shared DC offset: if every
key carries a big common mean vector, M is dominated by that one rank-1 mean term even
when the *centered* content spans many dimensions.

So here we recompute, per (layer, KV-head), averaged over sequences, BOTH:
  participation ratio  PR     = (sum lambda)^2 / sum(lambda^2)         (smooth eff. dim)
  trace-eff-rank       effr   = trace / lambda_max  (= resume_checks' stat)
on the uncentered second moment K^T K AND the mean-centered covariance Kc^T Kc, where
Kc = K - mean_over_keys(K).

Decision (per the user's gate):
  centered rank still low (~1-3)  -> finding is key geometry, rock-solid; proceed to STEP 1.
  centered rank jumps high        -> keys share a big common mean, NOT intrinsically
                                     low-rank; logdet still loses empirically but the
                                     MECHANISTIC claim changes -> STOP, report.

CPU is fine (per-head eig is on the 128x128 second moment). Run: python -m stage0.step0_centered_rank
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

CALIB = Path(__file__).resolve().parent.parent / "outputs" / "calib"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "phase_d_step0_centered_rank.md"
F64 = torch.float64


def _pr_effr(evals):
    ev = evals.clamp_min(0.0)
    s1 = ev.sum()
    s2 = (ev * ev).sum()
    pr = float((s1 * s1) / (s2 + 1e-30))
    effr = float(s1 / (ev.max() + 1e-30))
    return pr, effr


def head_stats(K):
    """K: (n, d) fp -> (PR, effr) uncentered and centered."""
    K = K.to(F64)
    M = K.T @ K
    ev_u = torch.linalg.eigvalsh(M)
    Kc = K - K.mean(0, keepdim=True)
    C = Kc.T @ Kc
    ev_c = torch.linalg.eigvalsh(C)
    return _pr_effr(ev_u), _pr_effr(ev_c)


def main(layers=range(32), n_seq_avg=8):
    meta = json.loads((CALIB / "meta.json").read_text())
    n_kv = meta["n_kv_heads"]
    rows = []
    for L in layers:
        t = torch.load(CALIB / f"layer_{L:02d}.pt", map_location="cpu")
        Kpre = t["K_pre"]                       # (n_seq, n_kv, P, hd)
        ns = min(n_seq_avg, Kpre.shape[0])
        for h in range(n_kv):
            pu, eu, pc, ec = [], [], [], []
            for s in range(ns):
                (pru, effu), (prc, effc) = head_stats(Kpre[s, h])
                pu.append(pru); eu.append(effu); pc.append(prc); ec.append(effc)
            rows.append({
                "layer": L, "kv": h,
                "PR_unc": sum(pu) / ns, "effr_unc": sum(eu) / ns,
                "PR_cen": sum(pc) / ns, "effr_cen": sum(ec) / ns,
            })
        del t
        print(f"  [step0] layer {L} done", flush=True)

    def col(name):
        return [r[name] for r in rows]

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        return mean, xs[0], xs[n // 2], xs[-1]

    lines = ["# Phase D STEP 0 — centered vs uncentered K_pre rank", "",
             f"Per (layer, KV-head) averaged over {n_seq_avg} sequences; d = 128.",
             "PR = participation ratio (sum l)^2/sum l^2 ; effr = trace/lambda_max.", "",
             f"  {'stat':<14} {'mean':>8} {'min':>8} {'median':>8} {'max':>8}"]
    for nm, label in [("PR_unc", "PR uncentered"), ("PR_cen", "PR CENTERED"),
                      ("effr_unc", "effr uncentered"), ("effr_cen", "effr CENTERED")]:
        m, lo, md, hi = stats(col(nm))
        lines.append(f"  {label:<14} {m:>8.2f} {lo:>8.2f} {md:>8.2f} {hi:>8.2f}")

    # full per-head table (compact)
    lines += ["", "  per-head (PR_unc / PR_cen | effr_unc / effr_cen):",
              f"  {'L':>2} {'kv':>2} | {'PR_unc':>7} {'PR_cen':>7} | {'effr_unc':>8} {'effr_cen':>8}"]
    for r in rows:
        lines.append(f"  {r['layer']:>2} {r['kv']:>2} | {r['PR_unc']:>7.2f} {r['PR_cen']:>7.2f} | "
                     f"{r['effr_unc']:>8.2f} {r['effr_cen']:>8.2f}")

    m_pr_cen, _, md_pr_cen, hi_pr_cen = stats(col("PR_cen"))
    m_effr_cen = sum(col("effr_cen")) / len(rows)
    low = md_pr_cen < 4.0 and m_effr_cen < 4.0
    lines += ["", "## Verdict",
              f"  centered PR: mean {m_pr_cen:.2f}, median {md_pr_cen:.2f}, max {hi_pr_cen:.2f}; "
              f"centered effr mean {m_effr_cen:.2f}",
              ("  -> CENTERED RANK STILL LOW: the rank-1 finding is KEY GEOMETRY, not a shared DC "
               "offset. Volume-diversity is structurally wrong on these heads. PROCEED to STEP 1."
               if low else
               "  -> CENTERED RANK JUMPS HIGH: keys share a large common-mean component; the cloud "
               "is NOT intrinsically low-rank. logdet still loses empirically but the MECHANISTIC "
               "claim changes. STOP and report so the framing can be rewritten.")]
    md = "\n".join(lines) + "\n"
    OUT.write_text(md)
    print(md)
    print(f"Wrote {OUT}  (low_rank_centered={low})")


if __name__ == "__main__":
    main()
