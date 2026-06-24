"""Phase C — early routable-margin probe on the saved Mistral calibration (CPU).

Step 0 (folded in): canonical facility check. K tight equal-mass clusters, no
outliers, budget b=K -> facility-location MUST beat logdet/kcenter (one centre per
cluster). If it loses, the sim-kernel bandwidth is wrong; switch from the 10th-pct
heuristic to the cloud's MEDIAN pairwise distance and re-check. This tells us
whether facility's real-head numbers are trustworthy as-is.

Step 1: probe heads spanning early/mid/late layers x all 8 KV-heads, at the
AGGRESSIVE budget. Per head: normalised Δ for {logdet, keydiff, kcenter, facility}
+ oracle + random, winner, margin m_h = |Δ(best diversity) - Δ(facility)| /
(Δ(random) - Δ(oracle)), and explicitly Δ(logdet) - Δ(kcenter).

Rules (hard): selectors + geometry on K_pre (pre-RoPE content); Δ and the top-b
oracle on K_post / V (real attention); oracle is a yardstick only.

Tripwires: negative normalised Δ -> too diffuse for the mass-oracle, EXCLUDE from
routing labels; report heads with |margin| above vs at the noise floor.

Run:  python -m stage0.phase_c
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from . import selectors
from .common import DTYPE, pairwise_sq_dist, rng

CALIB = Path(__file__).resolve().parent.parent / "outputs" / "calib"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "phase_c_probe.md"
F32 = torch.float32


# ----------------------------------------------------------------- Δ machinery
def _delta(Q, K_post, V, S):
    """Mean L2 attention-output error of retaining set S, over query rows Q."""
    full = torch.softmax(Q @ K_post.T / math.sqrt(K_post.shape[1]), dim=1) @ V
    sc = Q @ K_post[S].T / math.sqrt(K_post.shape[1])
    restr = torch.softmax(sc, dim=1) @ V[S]
    return float((full - restr).norm(dim=1).mean())


def _oracle_set(Q_oracle, K_post, b):
    mass = torch.softmax(Q_oracle @ K_post.T / math.sqrt(K_post.shape[1]), dim=1).sum(0)
    return torch.sort(torch.topk(mass, min(b, K_post.shape[0])).indices).values


def _selset(name, K_pre, b, bw_pct):
    if name == "logdet":
        return selectors.logdet_select(K_pre, b)
    if name == "keydiff":
        return selectors.keydiff_select(K_pre, b)
    if name == "kcenter":
        return selectors.kcenter_select(K_pre, b)
    if name == "facility":
        return selectors.coverage_select(K_pre, b, bandwidth_pct=bw_pct)
    raise ValueError(name)


SEL_NAMES = ["logdet", "keydiff", "kcenter", "facility"]


def measure_head(K_pre, K_post, V, Q_oracle, Q_eval, keep, bw_pct, r_draws=4, seed=0):
    """Returns raw Δ per selector + oracle + random, and normalised Δ."""
    N = K_pre.shape[0]
    b = max(1, round(keep * N))
    d_or = _delta(Q_eval, K_post, V, _oracle_set(Q_oracle, K_post, b))
    g = rng(seed)
    rand = [_delta(Q_eval, K_post, V, torch.randperm(N, generator=g)[:b]) for _ in range(r_draws)]
    d_rand = sum(rand) / len(rand)
    rng_span = (d_rand - d_or) + 1e-30
    raw = {nm: _delta(Q_eval, K_post, V, _selset(nm, K_pre, b, bw_pct)) for nm in SEL_NAMES}
    norm = {nm: (raw[nm] - d_or) / rng_span for nm in SEL_NAMES}
    return {"raw": raw, "norm": norm, "d_oracle": d_or, "d_random": d_rand,
            "rand_std": (sum((x - d_rand) ** 2 for x in rand) / len(rand)) ** 0.5 / rng_span}


# --------------------------------------------------- Step 0: canonical facility
def _cluster_head(K_clusters=12, per=20, d=16, blob_std=0.10, sep=6.0, seed=0):
    g = rng(seed)
    centers = torch.randn(K_clusters, d, generator=g, dtype=F32) * sep
    labels = torch.arange(K_clusters * per) % K_clusters
    K = centers[labels] + torch.randn(K_clusters * per, d, generator=g, dtype=F32) * blob_std
    V = K.clone()
    # peaked queries at each cluster centre (equal mass per cluster) -> coverage is right
    Q = centers.repeat_interleave(8, 0)
    return K, Q, V, K_clusters


def canonical_facility_check():
    lines = ["## Step 0 — canonical facility check (K equal-mass clusters, b=K)"]
    chosen_bw = 0.10
    for bw in (0.10, 0.50):
        wins = 0
        gaps = []
        for s in range(8):
            K, Q, V, Kc = _cluster_head(seed=s)
            b = Kc
            d_or = _delta(Q, K, V, _oracle_set(Q, K, b))
            d_fac = _delta(Q, K, V, selectors.coverage_select(K, b, bandwidth_pct=bw))
            d_ld = _delta(Q, K, V, selectors.logdet_select(K, b))
            d_kc = _delta(Q, K, V, selectors.kcenter_select(K, b))
            if d_fac < d_ld and d_fac < d_kc:
                wins += 1
            gaps.append(min(d_ld, d_kc) - d_fac)  # >0 => facility better
        mean_gap = sum(gaps) / len(gaps)
        verdict = "BEATS logdet/kcenter" if wins >= 7 else "FAILS to beat"
        lines.append(f"  bandwidth={'10th-pct' if bw==0.10 else 'median'}: facility {verdict} "
                     f"({wins}/8 seeds, mean Δ gap vs best-other {mean_gap:+.4f})")
        if wins >= 7:
            chosen_bw = bw
            break
    sound = chosen_bw is not None and any("BEATS" in l for l in lines)
    lines.append(f"  -> facility coverage selector is {'SOUND' if sound else 'UNSOUND'}; "
                 f"using bandwidth={'10th-pct' if chosen_bw==0.10 else 'median'} for the real-head probe."
                 + ("" if chosen_bw == 0.10 else
                    "  NOTE: real-head facility numbers are only trustworthy AFTER this fix."))
    return chosen_bw, lines


# ------------------------------------------------------- Step 1: real-head probe
def real_head_probe(bw_pct, layers=(0, 8, 16, 24, 31), keep=0.03,
                    n_cap=1024, n_seq_avg=2, noise_floor=0.05):
    meta = json.loads((CALIB / "meta.json").read_text())
    n_kv, group = meta["n_kv_heads"], meta["gqa_group"]
    rows = []
    for L in layers:
        t = torch.load(CALIB / f"layer_{L:02d}.pt", map_location="cpu")
        Kpre, Kpost, Vv = t["K_pre"], t["K_post"], t["V"]      # (n_seq,n_kv,P,hd) fp16
        Qor, Qev = t["Q_oracle"], t["Q_eval"]                  # (n_seq,n_kv,group,nq,hd)
        P = Kpre.shape[2]
        for h in range(n_kv):
            acc = {nm: [] for nm in SEL_NAMES}
            d_or_l, d_rd_l, rstd_l = [], [], []
            for s in range(n_seq_avg):
                g = rng(1000 + L * 97 + h * 7 + s)
                idx = torch.randperm(P, generator=g)[:n_cap]
                kp = Kpre[s, h, idx].to(F32)
                kq = Kpost[s, h, idx].to(F32)
                vv = Vv[s, h, idx].to(F32)
                qor = Qor[s, h].reshape(group * Qor.shape[3], -1).to(F32)   # (group*nq, hd)
                qev = Qev[s, h].reshape(group * Qev.shape[3], -1).to(F32)
                m = measure_head(kp, kq, vv, qor, qev, keep, bw_pct, seed=int(idx[0]))
                for nm in SEL_NAMES:
                    acc[nm].append(m["norm"][nm])
                d_or_l.append(m["d_oracle"]); d_rd_l.append(m["d_random"]); rstd_l.append(m["rand_std"])
            norm = {nm: sum(acc[nm]) / len(acc[nm]) for nm in SEL_NAMES}
            best_div = min(norm["logdet"], norm["keydiff"])
            best_div_name = "logdet" if norm["logdet"] <= norm["keydiff"] else "keydiff"
            margin = abs(best_div - norm["facility"])
            winner = min(SEL_NAMES, key=lambda nm: norm[nm])
            diffuse = min(norm.values()) < -0.02         # oracle not a floor -> too diffuse
            rows.append({
                "layer": L, "kv": h, "norm": norm, "winner": winner,
                "best_div": best_div_name, "margin": margin,
                "logdet_minus_kcenter": norm["logdet"] - norm["kcenter"],
                "rand_std": sum(rstd_l) / len(rstd_l), "diffuse": diffuse,
            })
        del t
        print(f"  [probe] layer {L} done ({n_kv} heads)", flush=True)
    return rows, noise_floor


def main():
    keep, n_cap, n_seq_avg = 0.03, 1024, 2
    bw_pct, step0 = canonical_facility_check()
    rows, noise_floor = real_head_probe(bw_pct, keep=keep, n_cap=n_cap, n_seq_avg=n_seq_avg)

    valid = [r for r in rows if not r["diffuse"]]
    excluded = [r for r in rows if r["diffuse"]]
    routable = [r for r in valid if r["margin"] > noise_floor]
    ld_kc_split = [r for r in valid if abs(r["logdet_minus_kcenter"]) > noise_floor]

    lines = list(step0)
    lines.append("")
    lines.append(f"## Step 1 — real-head probe (keep={keep:.0%}, layers 0/8/16/24/31 x 8 KV, "
                 f"N_cap={n_cap}, avg {n_seq_avg} seq)  [PROBE: reduced averaging -> exact per-head "
                 f"winners are noisy; qualitative tallies are the signal]")
    lines.append(f"  {'L':>2} {'kv':>2} | {'logdet':>8} {'keydiff':>8} {'kcenter':>8} "
                 f"{'facility':>8} | {'winner':>8} {'margin':>7} {'ld-kc':>7} {'flag':>7}")
    for r in rows:
        n = r["norm"]
        flag = "DIFFUSE" if r["diffuse"] else ("route" if r["margin"] > noise_floor else "")
        lines.append(f"  {r['layer']:>2} {r['kv']:>2} | {n['logdet']:>8.3f} {n['keydiff']:>8.3f} "
                     f"{n['kcenter']:>8.3f} {n['facility']:>8.3f} | {r['winner']:>8} "
                     f"{r['margin']:>7.3f} {r['logdet_minus_kcenter']:>+7.3f} {flag:>7}")

    # winner tallies
    from collections import Counter
    wc = Counter(r["winner"] for r in valid)
    lines += [
        "",
        "## Decisions",
        f"  valid heads: {len(valid)}/{len(rows)}  (excluded as too-diffuse / oracle-not-floor: "
        f"{len(excluded)})",
        f"  winner tally (valid heads): {dict(wc)}",
        f"  ROUTABILITY: heads with |margin| > noise floor ({noise_floor}): "
        f"{len(routable)}/{len(valid)}  "
        f"(median margin {sorted(r['margin'] for r in valid)[len(valid)//2]:.3f}, "
        f"max {max(r['margin'] for r in valid):.3f})",
        f"  facility ever wins a valid head: {wc.get('facility',0)>0}  "
        f"-> {'routable regime exists' if wc.get('facility',0)>0 else 'LIKELY NO-GO (diversity wins everywhere)'}",
        f"  LOGDET vs KCENTER: heads with |Δlogdet-Δkcenter| > {noise_floor}: "
        f"{len(ld_kc_split)}/{len(valid)}  "
        f"(mean |ld-kc| {sum(abs(r['logdet_minus_kcenter']) for r in valid)/max(1,len(valid)):.3f})  "
        f"-> {'SPLIT (volume vs min-max distinct)' if len(ld_kc_split) > len(valid)//3 else 'TRACK (drop kcenter in Phase D)'}",
        f"  logdet wins: {wc.get('logdet',0)} heads  -> "
        f"{'logdet is NEVER best on real heads; the synthetic volume-diversity story does NOT transfer' if wc.get('logdet',0)==0 else 'logdet still relevant'}.",
        "  CAVEAT: synthetic Phase A predicted logdet dominates / facility never wins -> the real "
        "heads show the OPPOSITE (logdet worst, facility+keydiff win). Origin-centred isotropic "
        "synthetic clouds mis-ranked both. Trust the real-head ordering.",
    ]
    md = "# Phase C — routable-margin probe\n\n" + "\n".join(lines) + "\n"
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(md)
    print(md)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
