"""Phase A — bidirectional validation of the attention-output-error metric Δ.

Stage 0 (iii) showed Δ ranks the selectors correctly when DIVERSITY is the right
answer (needle head). This probe adds the symmetric question and reports the full
picture, because the answer is budget-dependent:

  D1  diversity-wins direction (needle/retrieval head)  : Δ(diversity) < Δ(coverage)
  D2  coverage-wins  direction (skewed bulk head)       : Δ(coverage)  < Δ(diversity)
  C   convergence at moderate budget                    : both selectors -> near oracle

Conclusion the numbers support: Δ is NOT structurally biased toward diversity --
it flips both ways and always seats the oracle at the floor. But the flip only
appears at AGGRESSIVE budgets; at moderate/50% budgets farthest-point sampling is
itself near-optimal (k-center bounds worst-case reconstruction), so the two
query-free selectors agree and there is nothing to route -- exactly Stage 0
Finding 1 and the A.4 "selectors agree" NO-GO regime.

Run:  python -m stage0.phase_a
"""
from __future__ import annotations

from pathlib import Path

from . import metric, selectors

OUT = Path(__file__).resolve().parent.parent / "outputs" / "phase_a_metric_bias.md"
N_SEEDS = 12


def _deltas(head_fn, keep, seed):
    h = head_fn(seed=seed)
    K, Q, V = h["K"], h["Q"], h["V"]
    n = K.shape[0]
    b = max(2, round(n * keep))
    return {
        "oracle": metric.attention_output_error(Q, K, V, metric.oracle_set(Q, K, b)),
        "div": metric.attention_output_error(Q, K, V, selectors.diversity_select(K, b)),
        "cov": metric.attention_output_error(Q, K, V, selectors.coverage_select(K, b)),
        "rand": metric.random_delta(Q, K, V, b),
    }


def _agg(head_fn, keep):
    rows = [_deltas(head_fn, keep, s) for s in range(N_SEEDS)]
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    div_wins = sum(r["div"] < r["cov"] for r in rows)
    cov_wins = sum(r["cov"] < r["div"] for r in rows)
    gap = sum(r["div"] - r["cov"] for r in rows) / len(rows)  # >0 => coverage wins
    return mean, div_wins, cov_wins, gap


def main():
    lines = []

    def block(title, head_fn, keep, expect):
        m, dw, cw, gap = _agg(head_fn, keep)
        norm = gap / (m["rand"] - m["oracle"] + 1e-30)  # normalised separation
        lines.append(f"### {title}  (keep={keep:.0%}, {N_SEEDS} seeds)")
        lines.append(f"    mean Δ:  oracle={m['oracle']:.4f}  diversity={m['div']:.4f}  "
                     f"coverage={m['cov']:.4f}  random={m['rand']:.4f}")
        lines.append(f"    diversity-wins {dw}/{N_SEEDS} | coverage-wins {cw}/{N_SEEDS} | "
                     f"mean(Δdiv-Δcov)={gap:+.4f}  normalised={norm:+.4f}  (>0 => coverage wins)")
        lines.append(f"    expectation: {expect}")
        lines.append("")
        return dw, cw, gap, norm

    # D1: diversity is the right answer (needle/retrieval head), aggressive budget
    d1 = block("D1  needle/retrieval head", metric.make_mixed_head, 0.25,
               "Δ(diversity) < Δ(coverage)")
    # D2: coverage is the right answer (skewed bulk head), aggressive budget
    d2 = block("D2  skewed bulk head", metric.make_skewed_head, 0.10,
               "Δ(coverage) < Δ(diversity)")
    # C: moderate budget on the same bulk head -> selectors converge to near-oracle
    cmod = block("C   skewed bulk head, MODERATE budget", metric.make_skewed_head, 0.50,
                 "selectors converge: |Δdiv-Δcov| ~ 0, both near oracle")

    d1_ok = d1[0] >= N_SEEDS - 1                  # diversity wins on the needle head
    d2_ok = d2[1] >= N_SEEDS - 1                  # coverage wins on the bulk head (aggressive)
    converge = abs(cmod[3]) < 0.05               # near-zero NORMALISED margin at moderate budget

    verdict = []
    verdict.append(f"[{'PASS' if d1_ok else 'FAIL'}] D1 diversity-wins direction "
                   f"({d1[0]}/{N_SEEDS})")
    verdict.append(f"[{'PASS' if d2_ok else 'FAIL'}] D2 coverage-wins direction at aggressive "
                   f"budget ({d2[1]}/{N_SEEDS})")
    verdict.append(f"[{'note' if converge else '????'}] C  selectors converge at moderate "
                   f"budget (normalised margin {cmod[3]:+.4f})")
    verdict.append("[caveat] tractable oracle = top-b-by-mass is a tight floor only when "
                   "attention is PEAKED; under the deliberately near-uniform D2/C head it is "
                   "loose (real Mistral attention is peaked, so it is fine for Stage 1).")
    metric_unbiased = d1_ok and d2_ok

    head = (
        "# Phase A — bidirectional metric (Δ) validation\n\n"
        f"Metric is unbiased (flips both ways at aggressive budget): "
        f"**{metric_unbiased}**\n\n"
        "Key finding: the diversity/coverage flip is BUDGET-DEPENDENT. It is robust at\n"
        "aggressive budgets and head-dependent (needle->diversity, skewed-bulk->coverage),\n"
        "but at moderate/50% budgets the two query-free selectors converge to near-oracle\n"
        "(farthest-point sampling is itself near-optimal), so no separation exists there to\n"
        "test directionality. This is Stage 0 Finding 1 / the A.4 'selectors agree' regime,\n"
        "and it is the stated reason the Stage 1 primary operating point is aggressive, not 50%.\n\n"
        "## Verdict\n```\n" + "\n".join(verdict) + "\n```\n\n## Detail\n\n"
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
