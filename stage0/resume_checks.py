"""Resume checks (a) + (b) — CPU, run after re-emit+verify PASS.

(a) LOGDET CONDITIONING.  Phase C: logdet wins 0/39 heads and sits at the random
    floor (normalised Δ ~ 1.0).  Is that real, or an ill-conditioned d x d dual?
    The current logdet_order uses a fixed ridge eps = 1e-3 * mean||k||^2.  Here we:
      - measure the K_pre second-moment spectrum per head (cond number lambda_max/lambda_min,
        effective rank) so we can see how ill-conditioned the dual actually is;
      - re-run greedy logdet with the marginal gain log(1 + k^T (M + eps I)^{-1} k) for a
        SWEEP of eps drawn from that spectrum (lambda_min, lambda_median, lambda_mean,
        lambda_max*1e-3) plus the current default;
      - report whether logdet's normalised Δ moves OFF the ~1.0 random floor for ANY eps.
    If it never moves -> logdet is genuinely dead on real heads (drop it in Phase D).
    If the right eps pulls it toward keydiff/facility -> it was a conditioning artifact.

(b) KCENTER / FACILITY OVERLAP.  Winner labels are an argmin partition, so the win
    SETS are disjoint by construction -- the real question is whether kcenter carries
    signal facility does not: on kcenter-win heads is facility also near-best (redundant)
    or far off (complementary)?  We report, per kcenter-win head, facility's gap+rank
    (and vice versa), plus the across-head correlation of the two normalised Δ columns.

Reuses phase_c's EXACT subsampling so numbers line up with outputs/phase_c_probe.md.

Run:  python -m stage0.resume_checks
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from . import selectors
from .common import rng
from .phase_c import _delta, _oracle_set, SEL_NAMES, measure_head

CALIB = Path(__file__).resolve().parent.parent / "outputs" / "calib"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "resume_checks.md"
F32 = torch.float32


# ----------------------------------------------------------------- logdet, absolute ridge
def _logdet_order_eps(K, eps_abs):
    """Greedy log-det order with an ABSOLUTE ridge eps (M = eps I + sum k k^T).

    Identical mechanics to selectors.logdet_order but eps is given directly so we can
    sweep it across the K_pre spectrum instead of the fixed 1e-3*mean-energy default.
    """
    n, d = K.shape
    Minv = torch.eye(d, dtype=K.dtype) / eps_abs
    avail = torch.ones(n, dtype=torch.bool)
    order = []
    for _ in range(n):
        lev = (K @ Minv * K).sum(1)
        gain = torch.log1p(lev)
        gain[~avail] = float("-inf")
        u = int(gain.argmax())
        order.append(u)
        avail[u] = False
        k = K[u]
        Mk = Minv @ k
        Minv = Minv - torch.outer(Mk, Mk) / (1.0 + float(k @ Mk))
    return torch.tensor(order, dtype=torch.long)


def _logdet_select_eps(K, b, eps_abs):
    b = min(b, K.shape[0])
    return torch.sort(_logdet_order_eps(K, eps_abs)[:b]).values


def _spectrum(K):
    """Eigenspectrum of the d x d second-moment M = sum_i k_i k_i^T (= K^T K)."""
    M = K.T @ K
    ev = torch.linalg.eigvalsh(M).clamp_min(0.0)        # ascending, (d,)
    lam_min = float(ev[ev > 0].min()) if (ev > 0).any() else 0.0
    lam_max = float(ev[-1])
    lam_med = float(ev.median())
    lam_mean = float(ev.mean())
    eff_rank = float(ev.sum() / (lam_max + 1e-30))       # trace/lambda_max in [1,d]
    cond = lam_max / (lam_min + 1e-30)
    return {"lam_min": lam_min, "lam_max": lam_max, "lam_med": lam_med,
            "lam_mean": lam_mean, "cond": cond, "eff_rank": eff_rank}


# ----------------------------------------------------------------- driver
def run(layers=(0, 8, 16, 24, 31), keep=0.03, n_cap=1024, n_seq_avg=2,
        bw_pct=0.10, noise_floor=0.05):
    meta = json.loads((CALIB / "meta.json").read_text())
    n_kv, group = meta["n_kv_heads"], meta["gqa_group"]

    # eps variants, named; values resolved per-head from its spectrum
    EPS_KEYS = ["default", "lam_min", "lam_med", "lam_mean", "lam_max*1e-3"]
    rows = []
    for L in layers:
        t = torch.load(CALIB / f"layer_{L:02d}.pt", map_location="cpu")
        Kpre, Kpost, Vv = t["K_pre"], t["K_post"], t["V"]
        Qor, Qev = t["Q_oracle"], t["Q_eval"]
        P = Kpre.shape[2]
        for h in range(n_kv):
            base_acc = {nm: [] for nm in SEL_NAMES}
            ld_acc = {k: [] for k in EPS_KEYS}
            spec_acc = []
            d_or_l, d_rd_l = [], []
            for s in range(n_seq_avg):
                g = rng(1000 + L * 97 + h * 7 + s)          # EXACT phase_c subsample
                idx = torch.randperm(P, generator=g)[:n_cap]
                kp = Kpre[s, h, idx].to(F32)
                kq = Kpost[s, h, idx].to(F32)
                vv = Vv[s, h, idx].to(F32)
                qor = Qor[s, h].reshape(group * Qor.shape[3], -1).to(F32)
                qev = Qev[s, h].reshape(group * Qev.shape[3], -1).to(F32)

                # baseline normalised Δ (matches phase_c)
                m = measure_head(kp, kq, vv, qor, qev, keep, bw_pct, seed=int(idx[0]))
                for nm in SEL_NAMES:
                    base_acc[nm].append(m["norm"][nm])
                d_or, d_rand = m["d_oracle"], m["d_random"]
                d_or_l.append(d_or); d_rd_l.append(d_rand)
                span = (d_rand - d_or) + 1e-30
                b = max(1, round(keep * kp.shape[0]))

                # spectrum + logdet eps sweep on the SAME subsample
                sp = _spectrum(kp)
                spec_acc.append(sp)
                mean_energy = float((kp * kp).sum(1).mean())
                eps_vals = {
                    "default": 1e-3 * mean_energy,
                    "lam_min": max(sp["lam_min"], 1e-12),
                    "lam_med": max(sp["lam_med"], 1e-12),
                    "lam_mean": max(sp["lam_mean"], 1e-12),
                    "lam_max*1e-3": max(sp["lam_max"] * 1e-3, 1e-12),
                }
                for kkey, ev in eps_vals.items():
                    S = _logdet_select_eps(kp, b, ev)
                    d_ld = _delta(qev, kq, vv, S)
                    ld_acc[kkey].append((d_ld - d_or) / span)

            base = {nm: sum(v) / len(v) for nm, v in base_acc.items()}
            ld = {k: sum(v) / len(v) for k, v in ld_acc.items()}
            spec = {k: sum(d[k] for d in spec_acc) / len(spec_acc) for k in spec_acc[0]}
            winner = min(SEL_NAMES, key=lambda nm: base[nm])
            diffuse = min(base.values()) < -0.02
            rows.append({"layer": L, "kv": h, "base": base, "ld": ld, "spec": spec,
                         "winner": winner, "diffuse": diffuse})
        del t
        print(f"  [resume] layer {L} done", flush=True)
    return rows, noise_floor


def report(rows, noise_floor):
    valid = [r for r in rows if not r["diffuse"]]
    L = ["# Resume checks (a) logdet conditioning + (b) kcenter/facility overlap", ""]

    # ---------- (a) logdet conditioning ----------
    L += ["## (a) Logdet conditioning — is logdet=0-wins a bad ridge?", "",
          "Per valid head: K_pre second-moment cond number & effective rank, then logdet's",
          "normalised Δ (oracle=0, random=1) under a sweep of ridge eps drawn from the spectrum.",
          "A value near 1.0 = at the random floor; < ~0.85 = meaningfully better than random.", ""]
    hdr = (f"  {'L':>2} {'kv':>2} | {'cond':>10} {'effrank':>7} | "
           + " ".join(f"{k:>11}" for k in ["default", "lam_min", "lam_med", "lam_mean", "lmax*1e-3"])
           + f" | {'keydiff':>8} {'facility':>8}")
    L.append(hdr)
    best_off_floor = 0
    floor_default = 0
    for r in valid:
        ld = r["ld"]
        if ld["default"] > 0.85:
            floor_default += 1
        if min(ld.values()) < 0.85:
            best_off_floor += 1
        L.append(f"  {r['layer']:>2} {r['kv']:>2} | {r['spec']['cond']:>10.2e} "
                 f"{r['spec']['eff_rank']:>7.1f} | "
                 + " ".join(f"{ld[k]:>11.3f}" for k in
                            ["default", "lam_min", "lam_med", "lam_mean", "lam_max*1e-3"])
                 + f" | {r['base']['keydiff']:>8.3f} {r['base']['facility']:>8.3f}")
    n = len(valid)
    # best eps per head and how often each eps is the best
    from collections import Counter
    best_eps = Counter(min(r["ld"], key=lambda k: r["ld"][k]) for r in valid)
    mean_default = sum(r["ld"]["default"] for r in valid) / n
    mean_best = sum(min(r["ld"].values()) for r in valid) / n
    mean_cond = sum(r["spec"]["cond"] for r in valid) / n
    mean_effrank = sum(r["spec"]["eff_rank"] for r in valid) / n
    L += ["",
          f"  mean K_pre cond = {mean_cond:.2e}, mean eff-rank = {mean_effrank:.1f} / 128",
          f"  logdet mean normalised Δ: default eps = {mean_default:.3f}, "
          f"best-eps-per-head = {mean_best:.3f}  (oracle 0, random 1)",
          f"  heads where default logdet is at/above random floor (Δ>0.85): {floor_default}/{n}",
          f"  heads pulled OFF the floor (Δ<0.85) by the BEST spectrum eps: {best_off_floor}/{n}",
          f"  best-eps choice tally: {dict(best_eps)}",
          f"  -> VERDICT: {'logdet moves off the floor with a spectrum ridge -> revisit in Phase D' if mean_best < 0.85 else 'logdet stays at the random floor under EVERY eps -> conditioning is NOT the cause; logdet is genuinely dead on real heads (drop it).'}",
          ""]

    # ---------- (b) kcenter / facility overlap ----------
    kc_wins = [r for r in valid if r["winner"] == "kcenter"]
    fac_wins = [r for r in valid if r["winner"] == "facility"]

    def rank_of(base, nm):  # 1 = best (lowest Δ)
        return 1 + sum(1 for x in base.values() if x < base[nm])

    L += ["## (b) kcenter vs facility — disjoint, subset, or complementary?", "",
          "Winner labels are an argmin partition, so win-SETS are disjoint by construction.",
          "Real question: on kcenter-win heads, is facility near-best (redundant) or far off",
          "(complementary)? rank: 1=best of the 4 selectors.", "",
          f"  kcenter wins {len(kc_wins)} valid heads; facility wins {len(fac_wins)}.", "",
          "  kcenter-win heads -> how facility does there:"]
    for r in kc_wins:
        b = r["base"]
        L.append(f"    L{r['layer']:>2} kv{r['kv']} : kcenter Δ={b['kcenter']:.3f}  "
                 f"facility Δ={b['facility']:.3f} (rank {rank_of(b,'facility')})  "
                 f"gap fac-kc={b['facility']-b['kcenter']:+.3f}")
    L += ["", "  facility-win heads -> how kcenter does there:"]
    for r in fac_wins:
        b = r["base"]
        L.append(f"    L{r['layer']:>2} kv{r['kv']} : facility Δ={b['facility']:.3f}  "
                 f"kcenter Δ={b['kcenter']:.3f} (rank {rank_of(b,'kcenter')})  "
                 f"gap kc-fac={b['kcenter']-b['facility']:+.3f}")

    # across-head correlation of the two columns
    kc = torch.tensor([r["base"]["kcenter"] for r in valid])
    fc = torch.tensor([r["base"]["facility"] for r in valid])
    corr = float(((kc - kc.mean()) * (fc - fc.mean())).mean()
                 / (kc.std(unbiased=False) * fc.std(unbiased=False) + 1e-30))
    # complementarity: kcenter-win heads where facility is NOT near-best (rank>2 or gap big)
    kc_complementary = [r for r in kc_wins
                        if rank_of(r["base"], "facility") > 2
                        or (r["base"]["facility"] - r["base"]["kcenter"]) > 0.30]
    L += ["",
          f"  across-head corr(kcenter Δ, facility Δ) over {len(valid)} heads = {corr:+.3f}",
          f"  kcenter-win heads where facility is FAR off (rank>2 or gap>0.30): "
          f"{len(kc_complementary)}/{len(kc_wins)}",
          f"  -> {'kcenter is COMPLEMENTARY (wins heads facility cannot) -> keep both in Phase D' if kc_complementary else 'kcenter wins only where facility is also near-best -> redundant, fold into facility'}",
          ""]

    md = "\n".join(L) + "\n"
    OUT.write_text(md)
    print(md)
    print(f"Wrote {OUT}")


def main():
    rows, nf = run()
    report(rows, nf)


if __name__ == "__main__":
    main()
