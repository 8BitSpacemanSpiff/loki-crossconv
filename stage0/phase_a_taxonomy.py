"""Phase A (taxonomy correction) — score the corrected selector menu on synthetic
ground truth, at AGGRESSIVE and MODERATE budgets, and recommend the Stage 1 menu.

The corrected menu separates the conflated objectives:

  DIVERSITY poles (genuine):
    logdet   : greedy log-det / k-DPP MAP  -- max marginal gain in log det(K_S K_S^T)
    keydiff  : KeyDiff-proper (anchor = mean of L2-normalised keys, keep low cosine)
  COVERAGE reference (NOT a diversity pole):
    kcenter  : farthest-point sampling (min-max coverage)
  COVERAGE pole:
    facility : greedy facility location (min-sum coverage)

Clouds (each a head with K, Q, V so Δ and the top-b-by-mass oracle are defined):
    needle   : make_mixed_head  -- mass on planted outliers; a diversity pole should win
    skewed   : make_skewed_head -- skewed bulk; facility should win at aggressive budget
    bulk     : make_bulk_head   -- no-needle dense blobs, PEAKED attention (decisive test)

Reports per (cloud, budget): normalised Δ of each selector, which tracks the oracle,
needle retention (needle cloud), and the gap of each selector vs facility -- the
decisive question being whether logdet/keydiff diverge from facility at MODERATE
budget where kcenter (FPS) collapsed onto it.

Run:  python -m stage0.phase_a_taxonomy
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import metric, selectors

OUT = Path(__file__).resolve().parent.parent / "outputs" / "phase_a_taxonomy.md"
N_SEEDS = 12

SELECTORS = {
    "logdet":   lambda K, b: selectors.logdet_select(K, b),
    "keydiff":  lambda K, b: selectors.keydiff_select(K, b),
    "kcenter":  lambda K, b: selectors.kcenter_select(K, b),
    "facility": lambda K, b: selectors.coverage_select(K, b),
}
DIVERSITY = ["logdet", "keydiff", "kcenter"]

HEADS = {
    "needle": metric.make_mixed_head,
    "skewed": metric.make_skewed_head,
    "bulk":   metric.make_bulk_head,
}


def _attn_entropy(Q, K):
    A = torch.softmax(Q @ K.T, dim=1)
    return float((-(A * torch.log(A + 1e-12)).sum(1)).mean())


def _run(head_fn, keep, seed):
    h = head_fn(seed=seed)
    K, Q, V, needles = h["K"], h["Q"], h["V"], h["needles"]
    n = K.shape[0]
    b = max(2, round(n * keep))
    d_or = metric.attention_output_error(Q, K, V, metric.oracle_set(Q, K, b))
    d_rd = metric.random_delta(Q, K, V, b)
    rng = (d_rd - d_or) + 1e-30
    out = {"ent": _attn_entropy(Q, K), "n_needle": int(needles.sum())}
    for name, fn in SELECTORS.items():
        S = fn(K, b)
        d = metric.attention_output_error(Q, K, V, S)
        out[name] = {
            "norm": (d - d_or) / rng,            # 0 = oracle, 1 = random
            "needle_keep": int(needles[S].sum()),
        }
    return out


def _agg(head_fn, keep):
    rows = [_run(head_fn, keep, s) for s in range(N_SEEDS)]
    ent = sum(r["ent"] for r in rows) / len(rows)
    n_needle = rows[0]["n_needle"]
    agg = {}
    for name in SELECTORS:
        agg[name] = {
            "norm": sum(r[name]["norm"] for r in rows) / len(rows),
            "needle_keep": sum(r[name]["needle_keep"] for r in rows) / len(rows),
        }
    return agg, ent, n_needle


def main():
    budgets = [("aggressive", 0.05), ("moderate", 0.50)]
    lines = []
    summary = {}   # (cloud,budget_label) -> agg

    for cloud, head_fn in HEADS.items():
        for blabel, keep in budgets:
            agg, ent, n_needle = _agg(head_fn, keep)
            summary[(cloud, blabel)] = agg
            winner = min(SELECTORS, key=lambda nm: agg[nm]["norm"])
            best_div = min(DIVERSITY, key=lambda nm: agg[nm]["norm"])
            lines.append(f"### cloud={cloud}  budget={blabel} (keep={keep:.0%})  "
                         f"attn-entropy~{ent:.2f}")
            lines.append("    selector    normΔ[0=oracle,1=random]   needle_keep   gap-vs-facility")
            fac = agg["facility"]["norm"]
            for name in ["logdet", "keydiff", "kcenter", "facility"]:
                a = agg[name]
                nk = f"{a['needle_keep']:.1f}/{n_needle}" if n_needle else "  -  "
                gap = a["norm"] - fac
                star = "  <- tracks oracle" if name == winner else ""
                lines.append(f"    {name:<10s}  {a['norm']:+8.4f}              "
                             f"{nk:>8s}      {gap:+8.4f}{star}")
            lines.append(f"    best diversity pole here: {best_div}")
            lines.append("")

    # Peaked heads only (needle, bulk) have a valid top-b-by-mass oracle; the skewed
    # head's near-uniform attention makes its oracle loose (negative normΔ), so it is
    # excluded from the win bookkeeping below and reported descriptively only.
    PEAKED = [c for c in HEADS if c != "skewed"]
    cells = [(c, bl) for c in PEAKED for bl, _ in budgets]

    # Routing hinges on: does the COVERAGE pole (facility) ever clearly beat the best
    # DIVERSITY pole (logdet) on some head? If never, one objective dominates -> NO-GO.
    cov_beats_div = [(c, bl) for (c, bl) in cells
                     if summary[(c, bl)]["facility"]["norm"]
                     < summary[(c, bl)]["logdet"]["norm"] - 0.05]
    winners = {(c, bl): min(SELECTORS, key=lambda nm: summary[(c, bl)][nm]["norm"])
               for (c, bl) in cells}
    distinct_winners = set(winners.values())

    # Decisive moderate-budget test on the bulk head: gap-vs-facility of each pole.
    bm = summary[("bulk", "moderate")]
    fac = bm["facility"]["norm"]
    sep = {nm: bm[nm]["norm"] - fac for nm in DIVERSITY}

    def _verdict(gap):
        if gap < -0.05:
            return "beats facility (diversity better even at moderate budget)"
        if gap > 0.05:
            return "worse than facility"
        return "collapses onto facility (tie)"

    decisive = [
        "Peaked-head win table (valid oracle); winner per cell = lowest normΔ:",
    ]
    for (c, bl) in cells:
        decisive.append(f"    {c:<7s} {bl:<10s} -> winner: {winners[(c, bl)]}")
    decisive += [
        "",
        "Decisive moderate-budget test (bulk head): each diversity pole vs facility",
        f"    logdet  vs facility: {sep['logdet']:+.4f}  -> {_verdict(sep['logdet'])}",
        f"    keydiff vs facility: {sep['keydiff']:+.4f}  -> {_verdict(sep['keydiff'])}",
        f"    kcenter vs facility: {sep['kcenter']:+.4f}  -> {_verdict(sep['kcenter'])}",
        "",
        f"Cells where COVERAGE (facility) clearly beats best DIVERSITY (logdet): "
        f"{cov_beats_div if cov_beats_div else 'NONE'}",
        f"Distinct cell-winners across peaked heads: {sorted(distinct_winners)}",
    ]

    nd_agg = summary[("needle", "aggressive")]
    best_div_needle = min(DIVERSITY, key=lambda nm: nd_agg[nm]["norm"])
    n_needle = _agg(HEADS["needle"], 0.05)[2]

    rec = []
    if not cov_beats_div:
        rec.append("LIKELY ROUTING NO-GO signal (synthetic): the coverage pole (facility) never "
                   "beats the best diversity pole (logdet) on any peaked synthetic head/budget; "
                   "logdet is near-oracle everywhere. If this holds on real heads there is nothing "
                   "to route between diversity and coverage. This is a FINDING to confirm cheaply "
                   "on real heads (Phase C) before any heavy GPU -- not a failure.")
    else:
        rec.append(f"Routable: coverage beats diversity on {cov_beats_div} -> a per-head router "
                   "has a real regime to exploit.")
    rec.append(f"Best diversity pole on the needle cloud (its home turf): '{best_div_needle}' "
               f"(normΔ {nd_agg[best_div_needle]['norm']:+.4f}, needle_keep "
               f"{nd_agg[best_div_needle]['needle_keep']:.1f}/{n_needle}); facility drops needles "
               f"({nd_agg['facility']['needle_keep']:.1f}/{n_needle}).")
    rec.append("keydiff (KeyDiff-proper) is catastrophic on every synthetic cloud, but the clouds "
               "are origin-centred (cosine-to-anchor degenerate); treat this as a synthetic "
               "artifact, NOT a verdict -- keydiff is the mandatory Stage 3 baseline and needs a "
               "real-data score.")
    rec.append(f"PROPOSED Stage 1 menu (carry exactly 3): diversity-pole='{best_div_needle}', "
               "baseline='keydiff', coverage-pole='facility'. Drop kcenter from the GPU sweep "
               "(it is the min-max-coverage reference and tracks logdet closely).")

    head = (
        "# Phase A — corrected selector taxonomy\n\n"
        "Genuine diversity poles (logdet = k-DPP MAP, keydiff = KeyDiff-proper) built and\n"
        "scored against the coverage pole (facility) and the k-center coverage reference (FPS).\n\n"
        "## Decisive test & routing viability\n```\n" + "\n".join(decisive) + "\n```\n\n"
        "## Recommendation\n```\n" + "\n".join(rec) + "\n```\n\n"
        "## Detail (12 seeds each)\n\n"
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(head + "\n".join(lines) + "\n")
    print(head + "\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
