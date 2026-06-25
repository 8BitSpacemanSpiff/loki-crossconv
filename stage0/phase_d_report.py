"""Phase D report + plots — reads outputs/phase_d_raw.pt (from phase_d_step1).

Produces (no Stage-2 AUC fit -- that is the next gate):
  outputs/phase_d_step1_measurement.md  -- winner tally per budget, margin DISTRIBUTION
      (near-tie mass flagged), winner stability across the held-out sequence split,
      baseline (routed min{facility,kcenter} vs keydiff_unc), reference columns.
  outputs/phase_d_step2_geometry.md      -- g_h per head + winner-conditioned summary.
  outputs/phase_d_perhead.csv            -- per-head: winner, margins, all norm cols, g_h.
  outputs/phase_d_gh_by_winner.png       -- each g_h distribution split by winner.
  outputs/phase_d_margin_vs_budget.png   -- margin distribution across the budget sweep.
  outputs/phase_d_gh_scatter.png         -- pairwise g_h scatter coloured by winner.

Routing label = argmin over {facility, kcenter} (user's STEP-0b decision); a 3-way
sensitivity argmin{facility,kcenter,keydiff_cen} is also reported. Win condition =
beating keydiff_unc (deployable baseline).

Run: python -m stage0.phase_d_report   [path-to-raw.pt]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import torch

OUTDIR = Path(__file__).resolve().parent.parent / "outputs"
RAW = OUTDIR / "phase_d_raw.pt"
GEOMS = ["PR", "spectral_tail", "outlier_fraction", "clusteredness"]
DIFFUSE_THRESH = -0.02


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _agg(per_seq, cols, sel):
    """sel: slice of sequence indices -> {col: mean over those seqs} at one budget."""
    return {c: _mean(per_seq[c][sel[0]:sel[1]]) for c in cols}


def load(path):
    p = torch.load(path)
    rows = p["rows"]
    budgets, route, cols = p["budgets"], p["route"], p["cols"]
    primary, split, nseq = p["primary"], p["split"], p["n_seq"]
    H = []
    for r in rows:
        rec = {"layer": r["layer"], "kv": r["kv"], "geom": r["geom"], "bud": {}}
        for bf in budgets:
            ps = r["per_seq"][bf]
            allm = {c: _mean(ps[c]) for c in cols}
            A = {c: _mean(ps[c][:split]) for c in cols}
            B = {c: _mean(ps[c][split:]) for c in cols}
            win = min(route, key=lambda c: allm[c])
            win3 = min(route + ["keydiff_cen"], key=lambda c: allm[c])
            margin = abs(allm[route[0]] - allm[route[1]])
            winA = min(route, key=lambda c: A[c])
            winB = min(route, key=lambda c: B[c])
            diffuse = min(allm[c] for c in route) < DIFFUSE_THRESH
            rec["bud"][bf] = {"all": allm, "A": A, "B": B, "win": win, "win3": win3,
                              "margin": margin, "winA": winA, "winB": winB, "diffuse": diffuse}
        H.append(rec)
    return H, budgets, route, cols, primary, split, nseq


def _dist(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return dict(min=0, p10=0, p25=0, median=0, p75=0, p90=0, max=0)
    q = lambda p: xs[min(n - 1, int(p * n))]
    return dict(min=xs[0], p10=q(.10), p25=q(.25), median=xs[n // 2], p75=q(.75),
                p90=q(.90), max=xs[-1])


def measurement_md(H, budgets, route, primary, split, nseq):
    L = ["# Phase D STEP 1 — full-resolution per-head measurement", "",
         f"256 heads (32 layers x 8 KV), FULL cloud (7936 keys), n_seq={nseq} "
         f"(held-out split {split}/{nseq-split}), budgets {[f'{int(b*100)}%' for b in budgets]}.",
         "Normalised Δ: oracle=0, random=1 (lower=better). Routing label = argmin{facility,kcenter}.",
         "Δ + oracle on real K_post/V; selectors/geometry on K_pre (centered where noted).", ""]

    for bf in budgets:
        valid = [h for h in H if not h["bud"][bf]["diffuse"]]
        excl = [h for h in H if h["bud"][bf]["diffuse"]]
        wc = Counter(h["bud"][bf]["win"] for h in valid)
        wc3 = Counter(h["bud"][bf]["win3"] for h in valid)
        margins = [h["bud"][bf]["margin"] for h in valid]
        md = _dist(margins)
        tie05 = sum(1 for m in margins if m < 0.05)
        tie10 = sum(1 for m in margins if m < 0.10)
        flips = sum(1 for h in valid if h["bud"][bf]["winA"] != h["bud"][bf]["winB"])
        # baseline: routed winner Δ vs keydiff_unc
        beat = sum(1 for h in valid
                   if min(h["bud"][bf]["all"][c] for c in route) < h["bud"][bf]["all"]["keydiff_unc"])
        tag = "  <<< PRIMARY (aggressive)" if bf == primary else ""
        L += [f"## budget {int(bf*100)}%{tag}",
              f"  valid heads {len(valid)}/{len(H)} (diffuse-excluded {len(excl)})",
              f"  winner tally {{facility,kcenter}}: {dict(wc)}",
              f"  3-way sensitivity {{+keydiff_cen}}: {dict(wc3)}",
              f"  margin |fac-kc|: min {md['min']:.3f} p10 {md['p10']:.3f} p25 {md['p25']:.3f} "
              f"med {md['median']:.3f} p75 {md['p75']:.3f} p90 {md['p90']:.3f} max {md['max']:.3f}",
              f"  NEAR-TIE mass: <0.05 -> {tie05}/{len(valid)}, <0.10 -> {tie10}/{len(valid)} "
              f"(label noise the AUC must not fit on)",
              f"  winner STABILITY across split A vs B: {len(valid)-flips}/{len(valid)} stable, "
              f"{flips} flip",
              f"  BASELINE: routed min{{fac,kc}} beats keydiff_unc on {beat}/{len(valid)} heads",
              ""]

    # winner migration across budgets (valid at primary)
    L += ["## winner migration across budgets (heads valid at all budgets)"]
    allvalid = [h for h in H if all(not h["bud"][bf]["diffuse"] for bf in budgets)]
    mig = Counter(tuple(h["bud"][bf]["win"] for bf in budgets) for h in allvalid)
    L.append(f"  {len(allvalid)} heads valid at every budget; (win@3%,@10%,@25%,@50%) patterns:")
    for pat, c in mig.most_common(12):
        L.append(f"    {pat}: {c}")

    # per-head table at primary
    L += ["", f"## per-head @ {int(primary*100)}% (primary)",
          f"  {'L':>2} {'kv':>2} | {'fac':>6} {'kc':>6} {'kd_unc':>7} {'kd_cen':>7} {'ld_cen':>7} "
          f"| {'win':>8} {'margin':>6} {'A':>8} {'B':>8} {'flag':>7}"]
    for h in H:
        bd = h["bud"][primary]
        a = bd["all"]
        flag = "DIFFUSE" if bd["diffuse"] else ("FLIP" if bd["winA"] != bd["winB"] else "")
        L.append(f"  {h['layer']:>2} {h['kv']:>2} | {a['facility']:>6.2f} {a['kcenter']:>6.2f} "
                 f"{a['keydiff_unc']:>7.2f} {a['keydiff_cen']:>7.2f} {a['logdet_cen']:>7.2f} | "
                 f"{bd['win']:>8} {bd['margin']:>6.2f} {bd['winA']:>8} {bd['winB']:>8} {flag:>7}")
    (OUTDIR / "phase_d_step1_measurement.md").write_text("\n".join(L) + "\n")


def geometry_md(H, primary, route):
    valid = [h for h in H if not h["bud"][primary]["diffuse"]]
    L = ["# Phase D STEP 2 — geometry predictors g_h (centered K_pre)", "",
         "Per head (avg over sequences): PR (participation ratio), spectral_tail "
         "(1 - lambda_max/trace), outlier_fraction (>med+3MAD dist-to-centroid), "
         "clusteredness (2-means variance reduction). NOT fit yet -- raw predictors.", ""]
    # winner-conditioned summary at primary budget
    for w in route:
        hs = [h for h in valid if h["bud"][primary]["win"] == w]
        L.append(f"## winner={w}  ({len(hs)} heads)")
        for g in GEOMS:
            xs = [h["geom"][g] for h in hs]
            d = _dist(xs)
            L.append(f"  {g:<17} mean {_mean(xs):.3f}  [min {d['min']:.3f} med {d['median']:.3f} "
                     f"max {d['max']:.3f}]")
        L.append("")
    # separation: mean g_h per winner class side by side
    L.append("## class separation (mean g_h | facility-win vs kcenter-win)")
    fac = [h for h in valid if h["bud"][primary]["win"] == "facility"]
    kc = [h for h in valid if h["bud"][primary]["win"] == "kcenter"]
    L.append(f"  {'g_h':<17} {'facility':>10} {'kcenter':>10} {'|diff|':>8}")
    for g in GEOMS:
        mf, mk = _mean([h["geom"][g] for h in fac]), _mean([h["geom"][g] for h in kc])
        L.append(f"  {g:<17} {mf:>10.3f} {mk:>10.3f} {abs(mf-mk):>8.3f}")
    (OUTDIR / "phase_d_step2_geometry.md").write_text("\n".join(L) + "\n")


def csv_dump(H, budgets, route, cols, primary):
    path = OUTDIR / "phase_d_perhead.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["layer", "kv", "win_primary", "win3_primary", "diffuse_primary"]
        head += [f"margin_{int(b*100)}" for b in budgets]
        head += [f"{c}_{int(b*100)}" for b in budgets for c in cols]
        head += GEOMS
        w.writerow(head)
        for h in H:
            bp = h["bud"][primary]
            row = [h["layer"], h["kv"], bp["win"], bp["win3"], int(bp["diffuse"])]
            row += [f"{h['bud'][b]['margin']:.4f}" for b in budgets]
            row += [f"{h['bud'][b]['all'][c]:.4f}" for b in budgets for c in cols]
            row += [f"{h['geom'][g]:.4f}" for g in GEOMS]
            w.writerow(row)


def plots(H, budgets, route, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [h for h in H if not h["bud"][primary]["diffuse"]]
    colmap = {"facility": "tab:blue", "kcenter": "tab:orange"}

    # 1) g_h distribution by winner (4 panels)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, g in zip(axes.ravel(), GEOMS):
        for w in route:
            xs = [h["geom"][g] for h in valid if h["bud"][primary]["win"] == w]
            ax.hist(xs, bins=15, alpha=0.55, label=f"{w} (n={len(xs)})", color=colmap[w])
        ax.set_title(g)
        ax.set_xlabel(g)
        ax.legend(fontsize=8)
    fig.suptitle(f"g_h by routing winner @ {int(primary*100)}% (centered K_pre)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "phase_d_gh_by_winner.png", dpi=110)
    plt.close(fig)

    # 2) margin vs budget (strip + median)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, bf in enumerate(budgets):
        vv = [h for h in H if not h["bud"][bf]["diffuse"]]
        ms = [h["bud"][bf]["margin"] for h in vv]
        x = [i + (j / len(ms) - 0.5) * 0.6 for j in range(len(ms))]
        ax.scatter(x, ms, s=10, alpha=0.5, color="tab:gray")
        ax.plot([i - 0.3, i + 0.3], [sorted(ms)[len(ms)//2]] * 2, color="red", lw=2)
    ax.axhline(0.05, ls="--", color="k", lw=0.8, label="near-tie 0.05")
    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels([f"{int(b*100)}%" for b in budgets])
    ax.set_xlabel("budget (keep %)")
    ax.set_ylabel("margin |fac-kc| (norm-Δ units)")
    ax.set_title("routing margin across budget sweep (red=median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "phase_d_margin_vs_budget.png", dpi=110)
    plt.close(fig)

    # 3) pairwise g_h scatter coloured by winner (PR vs clusteredness, outlier vs spectral_tail)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pairs = [("PR", "clusteredness"), ("outlier_fraction", "spectral_tail")]
    for ax, (gx, gy) in zip(axes, pairs):
        for w in route:
            hs = [h for h in valid if h["bud"][primary]["win"] == w]
            ax.scatter([h["geom"][gx] for h in hs], [h["geom"][gy] for h in hs],
                       s=22, alpha=0.7, label=w, color=colmap[w])
        ax.set_xlabel(gx); ax.set_ylabel(gy); ax.legend()
    fig.suptitle(f"g_h scatter by winner @ {int(primary*100)}%")
    fig.tight_layout()
    fig.savefig(OUTDIR / "phase_d_gh_scatter.png", dpi=110)
    plt.close(fig)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW
    H, budgets, route, cols, primary, split, nseq = load(path)
    print(f"loaded {len(H)} heads from {path}")
    measurement_md(H, budgets, route, primary, split, nseq)
    geometry_md(H, primary, route)
    csv_dump(H, budgets, route, cols, primary)
    plots(H, budgets, route, primary)
    print("wrote: phase_d_step1_measurement.md, phase_d_step2_geometry.md, "
          "phase_d_perhead.csv, phase_d_gh_by_winner.png, phase_d_margin_vs_budget.png, "
          "phase_d_gh_scatter.png")


if __name__ == "__main__":
    main()
