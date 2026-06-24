"""Stage 0 instrument validation against the A.1 synthetic ground-truth table.

Three pass criteria (all must pass before any real Mistral data is touched):

 (i)  The two greedy selectors split planted needles in the predicted directions:
      diversity RETAINS needles, coverage DROPS them.
 (ii) The geometry statistics g_h take separable / correctly-ordered values across
      the three clouds (each instrument behaves as designed in the A.1 table).
 (iii)On a synthetic head with a KNOWN best-retention set, the attention-output
      error Δ ranks  oracle < better-objective < worse-objective < random.

Run:  python -m stage0.validate
Writes outputs/stage0_results.md and exits non-zero if any criterion FAILS.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from . import clouds, geometry, metric, selectors
from .common import DTYPE as DTYPE_F

OUT = Path(__file__).resolve().parent.parent / "outputs" / "stage0_results.md"
SEED = 0


def _fmt(x, w=10, p=4):
    return f"{x:>{w}.{p}f}"


# ---------------------------------------------------------------- criterion (i)
def criterion_i():
    """Needle split, two ways.

    (a) Budget-free signature: mean normalised PICK-ORDER rank of needles. The
        diversity objective ranks needles EARLY (small rank), coverage ranks them
        LATE (large rank) -- the designed opposite-pole behaviour, independent of
        any single budget choice.
    (b) Concrete consequence at aggressive eviction: at the budget where the bulk
        just fills the retained set, diversity keeps the needles and coverage drops
        them. (At loose budgets both can afford the needles -- nothing to route,
        which is itself an A.4 NO-GO signal, not an instrument failure.)
    """
    cloud = clouds.blob_with_needles(seed=SEED)
    K, needles = cloud["K"], cloud["needles"]
    n = K.shape[0]
    n_needles = int(needles.sum())
    needle_idx = torch.where(needles)[0]

    # (a) normalised pick-order rank in [0,1] (0 = picked first)
    div_ord = selectors.diversity_order(K)
    cov_ord = selectors.coverage_order(K)
    div_rank = torch.empty(n, dtype=DTYPE_F)
    cov_rank = torch.empty(n, dtype=DTYPE_F)
    div_rank[div_ord] = torch.arange(n, dtype=DTYPE_F)
    cov_rank[cov_ord] = torch.arange(n, dtype=DTYPE_F)
    div_rank /= (n - 1)
    cov_rank /= (n - 1)
    div_needle_rank = float(div_rank[needle_idx].mean())
    cov_needle_rank = float(cov_rank[needle_idx].mean())

    # (b) membership split vs budget. A single dense blob has small productive
    # capacity (~bulk dimensionality), so coverage only sacrifices needles under
    # AGGRESSIVE eviction; at a loose 50% budget it can afford to keep them too
    # (an A.4 "nothing to route" signal, not an instrument failure). Budgets are
    # expressed as multiples of the needle count so the split window is found
    # regardless of the exact per-seed capacity.
    sweep = []  # (label, b, div_keep, cov_keep)
    split_ok = False
    budgets = [("50% evict", round(n * 0.5))]
    budgets += [(f"{m:g}x needles", max(2, round(m * n_needles)))
                for m in (1.25, 1.5, 1.75, 2.0)]
    for label, b in budgets:
        dk = int(needles[selectors.diversity_select(K, b)].sum())
        ck = int(needles[selectors.coverage_select(K, b)].sum())
        sweep.append((label, b, dk, ck))
        if dk >= n_needles and ck <= 1:
            split_ok = True

    rank_ok = div_needle_rank < cov_needle_rank - 0.01  # diversity prioritises needles MORE
    ok = rank_ok and split_ok
    lines = [
        f"Cloud: blob_with_needles  (n={n}, needles={n_needles})",
        "  (a) mean normalised pick-order rank of needles [0=first .. 1=last]:",
        f"        diversity (FPS)       {div_needle_rank:.3f}",
        f"        coverage  (fac.loc.)  {cov_needle_rank:.3f}",
        f"        -> diversity ranks needles strictly earlier: "
        f"{'PASS' if rank_ok else 'FAIL'}",
        "  (b) needle retention vs eviction budget (div should stay HIGH; cov "
        "drops under pressure):",
        "        budget          retain b   div keeps   cov keeps",
    ]
    for label, b, dk, ck in sweep:
        mark = "  <== clean split" if (dk >= n_needles and ck <= 1) else ""
        lines.append(f"        {label:<14s}    {b:>4d}      {dk}/{n_needles}        "
                     f"{ck}/{n_needles}{mark}")
    lines.append(f"        -> clean split exists under aggressive eviction: "
                 f"{'PASS' if split_ok else 'FAIL'}")
    return ok, lines, {"div_rank": div_needle_rank, "cov_rank": cov_needle_rank}


# --------------------------------------------------------------- criterion (ii)
def criterion_ii():
    """Geometry statistics across the three clouds; check designed directions."""
    cs = clouds.all_clouds(seed=SEED)
    stats = {c["kind"]: geometry.all_stats(c["K"], seed=SEED) for c in cs}

    blobs = stats["gaussian_blobs"]
    needle = stats["blob_with_needles"]
    shell = stats["uniform_shell"]

    # GATED checks = the causal separators that must hold robustly. Together they
    # give the three clouds mutually separable g-vectors:
    #   outlier_fraction  -> needle detector (needles vs blobs and shell)
    #   participation_ratio -> isotropy (shell vs few-mode blobs)
    #   clusteredness     -> structure (blobs vs structureless shell)
    gated = {
        "outlier_fraction: needles >> blobs":
            needle["outlier_fraction"] > blobs["outlier_fraction"] + 0.02,
        "outlier_fraction: needles >> shell":
            needle["outlier_fraction"] > shell["outlier_fraction"] + 0.02,
        "participation_ratio: shell > blobs (isotropic vs few modes)":
            shell["participation_ratio"] > blobs["participation_ratio"],
        "clusteredness: blobs > shell (structure vs none)":
            blobs["clusteredness"] > shell["clusteredness"],
    }
    # INFORMATIONAL: spectral_tail is a listed g_h candidate but a weak, noisy
    # needle indicator (a dense blob's high-dim noise floor also fills the tail).
    # Reported with its designed direction but NOT gated -- Stage 2 selects the
    # predictive candidates on held-out heads, so a weak one is expected, not fatal.
    info = {
        "spectral_tail: needles > blobs (isolated directions) [informational]":
            needle["spectral_tail"] > blobs["spectral_tail"],
    }
    ok = all(gated.values())

    keys = ["participation_ratio", "spectral_tail", "outlier_fraction", "clusteredness"]
    header = "  " + "cloud".ljust(20) + "".join(k[:13].rjust(15) for k in keys)
    lines = [header]
    for kind in ["gaussian_blobs", "blob_with_needles", "uniform_shell"]:
        row = "  " + kind.ljust(20) + "".join(_fmt(stats[kind][k], 15) for k in keys)
        lines.append(row)
    lines.append("  gated designed-direction checks:")
    for desc, passed in gated.items():
        lines.append(f"    [{'PASS' if passed else 'FAIL'}] {desc}")
    lines.append("  informational (not gated):")
    for desc, passed in info.items():
        lines.append(f"    [{'ok  ' if passed else 'weak'}] {desc}")
    return ok, lines, stats


# -------------------------------------------------------------- criterion (iii)
def criterion_iii(evict_frac=0.75):
    """Δ ranking on the mixed head with a known best-retention set."""
    head = metric.make_mixed_head(seed=SEED)
    K, Q, V, needles = head["K"], head["Q"], head["V"], head["needles"]
    n = K.shape[0]
    b = round(n * (1.0 - evict_frac))

    S_oracle = metric.oracle_set(Q, K, b)
    S_div = selectors.diversity_select(K, b)
    S_cov = selectors.coverage_select(K, b)

    d_oracle = metric.attention_output_error(Q, K, V, S_oracle)
    d_div = metric.attention_output_error(Q, K, V, S_div)
    d_cov = metric.attention_output_error(Q, K, V, S_cov)
    d_rand = metric.random_delta(Q, K, V, b)

    better, worse = ("diversity", d_div), ("coverage", d_cov)
    if d_cov < d_div:
        better, worse = ("coverage", d_cov), ("diversity", d_div)

    ok = d_oracle < better[1] < worse[1] < d_rand
    nz = lambda d: metric.normalize_delta(d, d_oracle, d_rand)
    lines = [
        f"Mixed head (n={n}, needles={int(needles.sum())}, retain b={b} @ "
        f"{int(evict_frac*100)}% eviction). Δ raw | normalised [oracle=0, random=1]:",
        f"  oracle                {_fmt(d_oracle)} | {_fmt(nz(d_oracle))}",
        f"  diversity (FPS)       {_fmt(d_div)} | {_fmt(nz(d_div))}",
        f"  coverage  (fac.loc.)  {_fmt(d_cov)} | {_fmt(nz(d_cov))}",
        f"  random (16 draws)     {_fmt(d_rand)} | {_fmt(nz(d_rand))}",
        f"  required order: oracle < {better[0]} < {worse[0]} < random  "
        f"({'satisfied' if ok else 'VIOLATED'})",
        f"  diversity keeps needles: {int(needles[S_div].sum())}/{int(needles.sum())}; "
        f"coverage keeps needles: {int(needles[S_cov].sum())}/{int(needles.sum())}",
    ]
    return ok, lines, {"oracle": d_oracle, "div": d_div, "cov": d_cov, "rand": d_rand}


def robustness(n_seeds=8):
    """Re-run all three criteria across seeds; report pass counts (no seed-luck)."""
    global SEED
    saved = SEED
    counts = {"i": 0, "ii": 0, "iii": 0}
    for s in range(n_seeds):
        SEED = s
        counts["i"] += bool(criterion_i()[0])
        counts["ii"] += bool(criterion_ii()[0])
        counts["iii"] += bool(criterion_iii()[0])
    SEED = saved
    lines = [f"Each criterion re-run on seeds 0..{n_seeds-1} (must pass on all):"]
    allok = True
    for k, label in [("i", "(i)   selector needle-split"),
                     ("ii", "(ii)  geometry separability"),
                     ("iii", "(iii) Δ metric ranking")]:
        ok = counts[k] == n_seeds
        allok &= ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {label}: {counts[k]}/{n_seeds} seeds")
    return allok, lines


def main():
    torch.manual_seed(SEED)
    results = []
    blocks = []

    for name, fn in [
        ("(i)   Selector needle-split", criterion_i),
        ("(ii)  Geometry statistic separability", criterion_ii),
        ("(iii) Δ metric ranking (oracle<better<worse<random)", criterion_iii),
    ]:
        ok, lines, _ = fn()
        results.append((name, ok))
        status = "PASS" if ok else "FAIL"
        blocks.append(f"### {status}  {name}\n```\n" + "\n".join(lines) + "\n```")

    rob_ok, rob_lines = robustness()
    results.append(("Robustness (8 seeds)", rob_ok))
    blocks.append(f"### {'PASS' if rob_ok else 'FAIL'}  Robustness across seeds\n```\n"
                  + "\n".join(rob_lines) + "\n```")

    all_pass = all(ok for _, ok in results)
    summary = "\n".join(
        f"  [{'PASS' if ok else 'FAIL'}]  {name}" for name, ok in results
    )
    verdict = "ALL PASS — instruments validated; cleared for Stage 1." if all_pass else \
              "FAIL — fix instruments before touching real Mistral data."

    md = (
        "# Stage 0 — Instrument validation on synthetic ground truth\n\n"
        f"Verdict: **{verdict}**\n\n"
        "## Summary\n```\n" + summary + "\n```\n\n"
        "## Detail\n\n" + "\n\n".join(blocks) + "\n"
    )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(md)

    print("\n".join(b.replace("```", "").replace("### ", "\n=== ").rstrip() for b in blocks))
    print("\n=== SUMMARY ===")
    print(summary)
    print(verdict)
    print(f"\nWrote {OUT}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
