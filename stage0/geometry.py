"""Query-free geometry statistics g_h of a key cloud (Stage 0 / A.2).

Four cheap candidates, all computed from keys alone (query-free) and -- in Stage 1
-- pre-RoPE, so "distinctiveness" reflects content geometry, not position:

  participation_ratio : effective number of spectral modes of the key covariance,
                        PR = (sum lam)^2 / sum lam^2. Low => few modes
                        (coverage-friendly); high => spread (diversity matters).
  spectral_tail       : fraction of spectral energy beyond the spectral knee.
                        Heavy tail => isolated directions => needles present.
  outlier_fraction    : share of keys whose distance to their k-th nearest
                        neighbour is far above the cloud's typical kNN distance.
                        The most directly causal needle detector.
  clusteredness       : between- vs within-cluster variance ratio from a cheap
                        k-means (Lloyd, numpy-free). High => crisp clusters.

Each returns a python float. None peeks at queries.
"""
from __future__ import annotations

import torch

from .common import DTYPE, pairwise_sq_dist, pairwise_sq_dist_xy, rng


def _cov_eigvals(K: torch.Tensor) -> torch.Tensor:
    Kc = K - K.mean(0, keepdim=True)
    n = K.shape[0]
    cov = (Kc.T @ Kc) / (n - 1)
    lam = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    return lam  # ascending


def participation_ratio(K: torch.Tensor) -> float:
    lam = _cov_eigvals(K)
    pr = (lam.sum() ** 2) / (lam.pow(2).sum() + 1e-30)
    return float(pr)


def spectral_tail(K: torch.Tensor, knee_frac: float = 0.90) -> float:
    """Energy fraction beyond the knee (where cumulative energy passes knee_frac)."""
    lam = _cov_eigvals(K).flip(0)            # descending
    total = lam.sum() + 1e-30
    cum = torch.cumsum(lam, 0) / total
    knee = int(torch.searchsorted(cum, torch.tensor(knee_frac, dtype=lam.dtype))) + 1
    knee = min(knee, lam.numel())
    tail = lam[knee:].sum() / total
    return float(tail)


def outlier_fraction(K: torch.Tensor, k: int = 5, mult: float = 3.0) -> float:
    """Fraction of keys whose k-NN distance exceeds mult * median k-NN distance."""
    n = K.shape[0]
    d2 = pairwise_sq_dist(K).clone()
    d2.fill_diagonal_(float("inf"))
    kk = min(k, n - 1)
    knn = d2.topk(kk, dim=1, largest=False).values[:, -1].sqrt()  # dist to k-th NN
    med = knn.median().clamp_min(1e-12)
    return float((knn > mult * med).double().mean())


def clusteredness(K: torch.Tensor, k: int = 8, iters: int = 30, seed: int = 0) -> float:
    """Between/within variance ratio from k-means (Calinski-Harabasz-style)."""
    n, _ = K.shape
    g = rng(seed)
    idx = torch.randperm(n, generator=g)[:k]
    C = K[idx].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        d2 = pairwise_sq_dist_xy(K, C)
        assign = d2.argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                C[j] = K[m].mean(0)
    gmean = K.mean(0)
    within = torch.zeros((), dtype=DTYPE)
    between = torch.zeros((), dtype=DTYPE)
    for j in range(k):
        m = assign == j
        if m.any():
            within += ((K[m] - C[j]) ** 2).sum()
            between += m.sum() * ((C[j] - gmean) ** 2).sum()
    return float(between / (within + 1e-30))


def all_stats(K: torch.Tensor, seed: int = 0) -> dict:
    return {
        "participation_ratio": participation_ratio(K),
        "spectral_tail": spectral_tail(K),
        "outlier_fraction": outlier_fraction(K),
        "clusteredness": clusteredness(K, seed=seed),
    }
