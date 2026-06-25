"""Phase D STEP 1 (full measurement) + STEP 2 (geometry g_h) — GPU, batched.

Fixes Phase C's noise: ALL 256 (layer, KV-head), FULL cloud (all 7936 keys, no
N_cap), multiple sequences with a held-out split for winner stability, budget SWEPT.

Per the user's STEP-0b decision (deploy-vs-content fork resolved):
  ROUTING label (Stage-2 target) = argmin over {facility, kcenter} (offset-INVARIANT).
  Reported (NOT routing candidates):
    keydiff_unc  -- KeyDiff on RAW keys: the deployable BASELINE to beat.
    keydiff_cen  -- KeyDiff de-meaned: reference (sink-collapse).
    logdet_cen   -- logdet de-meaned: reference (diversity stays weak de-sinked).
  Also a 3-way sensitivity winner argmin{facility,kcenter,keydiff_cen}.
Hard rules: selectors/geometry on K_pre; Δ + oracle ALWAYS on real uncentered K_post/V;
GQA group 4; geometry stats on CENTERED K_pre. Oracle is a yardstick only.

Selectors are run BATCHED over the n_seq independent clouds of each (layer, head):
facility (submodular greedy), kcenter (farthest-point), logdet (D-optimal dual),
keydiff (cosine). Batched orders are asserted identical to the per-cloud selectors
(stage0.selectors) on seq 0 in --smoke.

  python -m stage0.phase_d_step1 --smoke      # 2 layers x 2 heads, 4 seqs + equivalence check
  python -m stage0.phase_d_step1              # full: 32 layers x 8 KV, 16 seqs
Writes outputs/phase_d_raw.pt (report+plots: stage0.phase_d_report).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from . import selectors
from .common import rng

CALIB = Path(__file__).resolve().parent.parent / "outputs" / "calib"
RAW = Path(__file__).resolve().parent.parent / "outputs" / "phase_d_raw.pt"
BUDGETS = [0.03, 0.10, 0.25, 0.50]
PRIMARY = 0.03
COLS = ["facility", "kcenter", "keydiff_cen", "keydiff_unc", "logdet_cen"]
ROUTE = ["facility", "kcenter"]


# ----------------------------------------------------------------- batched geometry
def _bpairwise(K):                       # (B,n,d) -> (B,n,n)
    sq = (K * K).sum(2)
    d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * torch.bmm(K, K.transpose(1, 2))
    return d2.clamp_min_(0.0)


def _bbandwidth(d2, pct):                # (B,n,n) -> (B,)
    B, n, _ = d2.shape
    iu = torch.triu_indices(n, n, offset=1, device=d2.device)
    dist = d2[:, iu[0], iu[1]].sqrt()    # (B, npairs)
    cap = 1 << 22
    if dist.shape[1] > cap:
        g = torch.Generator(device="cpu").manual_seed(0)
        sel = torch.randperm(dist.shape[1], generator=g)[:cap].to(d2.device)
        dist = dist[:, sel]
    return torch.quantile(dist, pct, dim=1).clamp_min(1e-6)


# ----------------------------------------------------------------- batched selectors
def _coverage_batched(K, d2, pct, k):    # (B,n,d),(B,n,n) -> order (B,k)
    B, n, _ = K.shape
    bw = _bbandwidth(d2, pct)
    sim = torch.exp(-d2 / (2.0 * (bw * bw))[:, None, None])
    covered = torch.zeros(B, n, device=K.device)
    avail = torch.ones(B, n, dtype=torch.bool, device=K.device)
    order = torch.empty(B, k, dtype=torch.long, device=K.device)
    ar = torch.arange(B, device=K.device)
    for i in range(k):
        gain = torch.clamp(sim - covered[:, :, None], min=0.0).sum(1)   # (B,n)
        gain = gain.masked_fill(~avail, -1.0)
        u = gain.argmax(1)
        order[:, i] = u
        avail[ar, u] = False
        covered = torch.maximum(covered, sim[ar, :, u])
    return order


def _kcenter_batched(K, d2, k):          # farthest-point + medoid seed
    B, n, _ = K.shape
    ar = torch.arange(B, device=K.device)
    start = d2.sum(2).argmin(1)
    order = torch.empty(B, k, dtype=torch.long, device=K.device)
    order[:, 0] = start
    min_d = d2[ar, start, :].clone()
    min_d[ar, start] = -1.0
    for i in range(1, k):
        nxt = min_d.argmax(1)
        order[:, i] = nxt
        min_d = torch.minimum(min_d, d2[ar, nxt, :])
        min_d[ar, nxt] = -1.0
    return order


def _logdet_batched(K, k, eps_frac=1e-3):
    B, n, d = K.shape
    ar = torch.arange(B, device=K.device)
    eps = eps_frac * (K * K).sum(2).mean(1)                # (B,)
    Minv = (torch.eye(d, device=K.device)[None] / eps[:, None, None]).clone()
    avail = torch.ones(B, n, dtype=torch.bool, device=K.device)
    order = torch.empty(B, k, dtype=torch.long, device=K.device)
    for i in range(k):
        KM = torch.bmm(K, Minv)                            # (B,n,d)
        lev = (KM * K).sum(2)                              # (B,n)
        gain = torch.log1p(lev).masked_fill(~avail, float("-inf"))
        u = gain.argmax(1)
        order[:, i] = u
        avail[ar, u] = False
        kv = K[ar, u, :]                                   # (B,d)
        Mk = torch.bmm(Minv, kv[:, :, None])[:, :, 0]      # (B,d)
        denom = 1.0 + (kv * Mk).sum(1)
        Minv = Minv - torch.bmm(Mk[:, :, None], Mk[:, None, :]) / denom[:, None, None]
    return order


def _keydiff_batched(K):                 # cosine-to-mean-direction; full order (B,n)
    Kn = K / K.norm(dim=2, keepdim=True).clamp_min(1e-12)
    anchor = Kn.mean(1)                                    # (B,d)
    cos = (Kn * anchor[:, None, :]).sum(2)                 # (B,n)
    return torch.argsort(cos, dim=1)


# ----------------------------------------------------------------- geometry stats g_h
def _geom_batched(kc):                   # centered (B,n,d) -> dict of (B,)
    B, n, d = kc.shape
    C = torch.bmm(kc.transpose(1, 2), kc) / n              # (B,d,d)
    evals, evecs = torch.linalg.eigh(C)                    # ascending
    ev = evals.clamp_min(0.0)
    s1, s2 = ev.sum(1), (ev * ev).sum(1)
    PR = (s1 * s1) / (s2 + 1e-30)
    spectral_tail = 1.0 - ev[:, -1] / (s1 + 1e-30)
    r = kc.norm(dim=2)                                     # dist to centroid (0)
    med = r.median(1).values
    mad = (r - med[:, None]).abs().median(1).values * 1.4826 + 1e-12
    outlier_fraction = (r > (med + 3.0 * mad)[:, None]).float().mean(1)
    clusteredness = _two_means_vr_batched(kc, evecs[:, :, -1])
    return {"PR": PR, "spectral_tail": spectral_tail,
            "outlier_fraction": outlier_fraction, "clusteredness": clusteredness}


def _two_means_vr_batched(X, vtop):      # X (B,n,d), vtop (B,d)
    B, n, d = X.shape
    ar = torch.arange(B, device=X.device)
    proj = (X * vtop[:, None, :]).sum(2)                   # (B,n)
    assign = proj > proj.median(1).values[:, None]
    inertia1 = (X * X).sum((1, 2))
    for _ in range(10):
        cnt1 = assign.sum(1).clamp_min(1)
        cnt0 = (~assign).sum(1).clamp_min(1)
        c1 = (X * assign[:, :, None]).sum(1) / cnt1[:, None]
        c0 = (X * (~assign)[:, :, None]).sum(1) / cnt0[:, None]
        d1 = ((X - c1[:, None, :]) ** 2).sum(2)
        d0 = ((X - c0[:, None, :]) ** 2).sum(2)
        assign = d1 < d0
    cnt1 = assign.sum(1).clamp_min(1); cnt0 = (~assign).sum(1).clamp_min(1)
    c1 = (X * assign[:, :, None]).sum(1) / cnt1[:, None]
    c0 = (X * (~assign)[:, :, None]).sum(1) / cnt0[:, None]
    w1 = (((X - c1[:, None, :]) ** 2).sum(2) * assign).sum(1)
    w0 = (((X - c0[:, None, :]) ** 2).sum(2) * (~assign)).sum(1)
    degenerate = (assign.all(1) | (~assign).all(1))
    vr = (1.0 - (w1 + w0) / (inertia1 + 1e-30)).clamp_min(0.0)
    return torch.where(degenerate, torch.zeros_like(vr), vr)


# ----------------------------------------------------------------- Δ machinery (per seq)
def _full(Q, Kpost, V):
    return torch.softmax(Q @ Kpost.T / math.sqrt(Kpost.shape[1]), dim=1) @ V


def _delta(full, Q, Kpost, V, S):
    sc = Q @ Kpost[S].T / math.sqrt(Kpost.shape[1])
    restr = torch.softmax(sc, dim=1) @ V[S]
    return float((full - restr).norm(dim=1).mean())


def _oracle_set(Qor, Kpost, b):
    mass = torch.softmax(Qor @ Kpost.T / math.sqrt(Kpost.shape[1]), dim=1).sum(0)
    return torch.sort(torch.topk(mass, min(b, Kpost.shape[0])).indices).values


def _equiv_check(kp0, kc0, bmax, bw_pct, fac, kc, kdu, kdc, ld):
    """Assert batched orders match per-cloud selectors on seq 0 (first 64 picks)."""
    m = 64
    ref = {
        "facility": selectors.coverage_order(kp0, bandwidth_pct=bw_pct, k=bmax)[:m],
        "kcenter": selectors.diversity_order(kp0, k=bmax)[:m],
        "keydiff_unc": selectors.keydiff_order(kp0)[:m],
        "keydiff_cen": selectors.keydiff_order(kc0)[:m],
        "logdet_cen": selectors.logdet_order(kc0, k=bmax)[:m],
    }
    got = {"facility": fac[0, :m], "kcenter": kc[0, :m], "keydiff_unc": kdu[0, :m],
           "keydiff_cen": kdc[0, :m], "logdet_cen": ld[0, :m]}
    for name in ref:
        eq = bool((ref[name].to(got[name].device) == got[name]).all())
        print(f"    equiv {name}: {eq}")
        assert eq, f"batched != per-cloud for {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-seq", type=int, default=16)
    ap.add_argument("--bw-pct", type=float, default=0.10)
    ap.add_argument("--layers", type=str, default="all")
    args = ap.parse_args()
    layers = list(range(32)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if args.smoke:
        layers, args.n_seq = [0, 16], 4

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((CALIB / "meta.json").read_text())
    n_kv, group = meta["n_kv_heads"], meta["gqa_group"]
    nseq = min(args.n_seq, meta["n_seq"])
    split = nseq // 2
    t0 = time.time()
    out = RAW if not args.smoke else RAW.with_name("phase_d_raw_smoke.pt")

    # resume: skip layers already present in a partial checkpoint (spot-instance safety)
    rows = []
    done = set()
    if not args.smoke and out.exists():
        prev = torch.load(out)
        if prev.get("n_seq") == nseq and prev.get("budgets") == BUDGETS:
            rows = prev["rows"]
            done = {r["layer"] for r in rows}
            print(f"resume: {len(done)} layers already done {sorted(done)}", flush=True)
    print(f"device={dev} layers={len(layers)} kv={n_kv} n_seq={nseq} split={split}/{nseq-split} "
          f"budgets={BUDGETS} cols={COLS}", flush=True)

    for L in layers:
        if L in done:
            continue
        t = torch.load(CALIB / f"layer_{L:02d}.pt", map_location="cpu")
        Kpre, Kpost, Vv = t["K_pre"], t["K_post"], t["V"]
        Qor, Qev = t["Q_oracle"], t["Q_eval"]
        P = Kpre.shape[2]
        bmax = max(max(1, round(bf * P)) for bf in BUDGETS)
        hcount = 2 if args.smoke else n_kv
        for h in range(hcount):
            kp = Kpre[:nseq, h].to(dev, torch.float32)            # (B,n,d)
            kc = kp - kp.mean(1, keepdim=True)
            d2 = _bpairwise(kp)                                   # invariant under centering
            fac = _coverage_batched(kp, d2, args.bw_pct, bmax)
            kcen = _kcenter_batched(kp, d2, bmax)
            del d2
            kdu = _keydiff_batched(kp)
            kdc = _keydiff_batched(kc)
            ld = _logdet_batched(kc, bmax)
            geom = _geom_batched(kc)
            if args.smoke:
                _equiv_check(kp[0], kc[0], bmax, args.bw_pct, fac, kcen, kdu, kdc, ld)
            orders = {"facility": fac, "kcenter": kcen, "keydiff_unc": kdu,
                      "keydiff_cen": kdc, "logdet_cen": ld}

            per_seq = {bf: {col: [] for col in COLS} for bf in BUDGETS}
            for s in range(nseq):
                kq = Kpost[s, h].to(dev, torch.float32)
                vv = Vv[s, h].to(dev, torch.float32)
                qor = Qor[s, h].reshape(group * Qor.shape[3], -1).to(dev, torch.float32)
                qev = Qev[s, h].reshape(group * Qev.shape[3], -1).to(dev, torch.float32)
                full = _full(qev, kq, vv)                        # precompute once per seq
                for bf in BUDGETS:
                    b = max(1, round(bf * P))
                    d_or = _delta(full, qev, kq, vv, _oracle_set(qor, kq, b))
                    rand = [_delta(full, qev, kq, vv,
                                   torch.randperm(P, generator=rng(1000 + L * 97 + h * 7 + s + 1 + i))[:b].to(dev))
                            for i in range(4)]
                    d_rand = sum(rand) / len(rand)
                    span = (d_rand - d_or) + 1e-30
                    for col in COLS:
                        S = torch.sort(orders[col][s, :b]).values
                        per_seq[bf][col].append((_delta(full, qev, kq, vv, S) - d_or) / span)
            geom_h = {k: float(geom[k].mean()) for k in geom}
            rows.append({"layer": L, "kv": h, "n_seq": nseq, "split": split,
                         "per_seq": per_seq, "geom": geom_h})
        del t
        payload = {"rows": rows, "budgets": BUDGETS, "primary": PRIMARY, "cols": COLS,
                   "route": ROUTE, "n_seq": nseq, "split": split, "bw_pct": args.bw_pct,
                   "smoke": args.smoke}
        torch.save(payload, out)                                  # per-layer checkpoint
        print(f"  [{time.time()-t0:.0f}s] layer {L} done ({hcount} heads) -> checkpoint "
              f"{len(rows)} heads", flush=True)

    print(f"[{time.time()-t0:.0f}s] WROTE {out}  ({len(rows)} heads)", flush=True)


if __name__ == "__main__":
    main()
