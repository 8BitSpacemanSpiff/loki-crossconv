"""The two query-free greedy selectors (Stage 0 / A.3).

Both pick a retained set S from a key cloud K, using ONLY the keys (no queries --
Part A is query-free). They sit at opposite poles:

  diversity_*  : greedy farthest-point sampling. Keeps geometric outliers
                 (needles) and spreads over the cloud. The KeyDiff / k-DPP-MAP
                 pole. Picks outliers FIRST.
  coverage_*   : greedy facility location, f(S) = sum_v max_{u in S} sim(v,u)
                 with a Gaussian kernel at a FINE bandwidth. Tiles dense regions
                 with representatives; an isolated needle only ever covers itself
                 (marginal gain ~1), so it is the LAST thing greedy picks once the
                 bulk is represented. The facility-location / max-coverage pole.

Each pole exposes two entry points:
  *_order(K)      -> full greedy pick order (permutation of 0..n-1)
  *_select(K, b)  -> sorted retained indices of size b (= first b of the order)

The pick ORDER is the budget-free signature of each objective (needle priority);
membership at a given budget is just its first-b prefix.

Determinism: diversity seeds from the medoid (most central key); coverage is
deterministic given the bandwidth.
"""
from __future__ import annotations

import torch

from .common import pairwise_sq_dist


def diversity_order(K: torch.Tensor, k: int | None = None) -> torch.Tensor:
    """Greedy farthest-point order over all keys. Returns a permutation (n,).

    k: if given, stop after k picks (the first-k prefix is identical to the full
    order's prefix, so this only saves work -- the result is unchanged for budgets <= k).
    """
    n = K.shape[0]
    k = n if k is None else min(k, n)
    d2 = pairwise_sq_dist(K)
    start = int(d2.sum(1).argmin())          # medoid: deterministic seed
    order = [start]
    min_d = d2[start].clone()
    min_d[start] = -1.0
    for _ in range(k - 1):
        nxt = int(min_d.argmax())
        order.append(nxt)
        min_d = torch.minimum(min_d, d2[nxt])
        min_d[nxt] = -1.0
    return torch.tensor(order, dtype=torch.long, device=K.device)


def _coverage_bandwidth(d2: torch.Tensor, percentile: float) -> torch.Tensor:
    n = d2.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=d2.device)
    dist = d2[iu[0], iu[1]].sqrt()
    # torch.quantile caps input at 2**24 elements; for large clouds the upper-triangle
    # exceeds it. Subsample deterministically -- a quantile of pairwise distances is
    # essentially unchanged by sampling millions of them. Exact for small clouds.
    cap = 1 << 23
    if dist.numel() > cap:
        g = torch.Generator(device="cpu").manual_seed(0)
        sel = torch.randperm(dist.numel(), generator=g)[:cap].to(dist.device)
        dist = dist[sel]
    return torch.quantile(dist, percentile).clamp_min(1e-6)


def coverage_order(K: torch.Tensor, bandwidth_pct: float = 0.10,
                   k: int | None = None) -> torch.Tensor:
    """Greedy facility-location order over all keys. Returns a permutation (n,).

    bandwidth_pct: percentile of pairwise distances used as the Gaussian kernel
    width. A FINE bandwidth makes the objective reward tiling dense regions, which
    is what pushes isolated needles to the END of the order under budget pressure.
    k: if given, stop after k picks (prefix identical to the full order's prefix).
    """
    n = K.shape[0]
    k = n if k is None else min(k, n)
    d2 = pairwise_sq_dist(K)
    bw = _coverage_bandwidth(d2, bandwidth_pct)
    sim = torch.exp(-d2 / (2.0 * bw * bw))   # (n, n) in (0, 1]
    covered = torch.zeros(n, dtype=K.dtype, device=K.device)
    avail = torch.ones(n, dtype=torch.bool, device=K.device)
    order = []
    for _ in range(k):
        gain = torch.clamp(sim - covered[:, None], min=0.0).sum(0)
        gain[~avail] = -1.0
        u = int(gain.argmax())
        order.append(u)
        avail[u] = False
        covered = torch.maximum(covered, sim[:, u])
    return torch.tensor(order, dtype=torch.long, device=K.device)


# Alias: farthest-point sampling is k-CENTER (min-max coverage), a coverage
# *reference*, NOT the diversity pole. Kept under both names for clarity.
kcenter_order = diversity_order


def logdet_order(K: torch.Tensor, eps_frac: float = 1e-3,
                 k: int | None = None) -> torch.Tensor:
    """Greedy log-det maximisation (k-DPP MAP) -- the TRUE diversity pole.

    Maximises log det(K_S K_S^T + eps I) greedily. Implemented via the d x d dual:
    keep M = eps I + sum_{s in S} k_s k_s^T and pick the key with the largest
    marginal gain log(1 + k^T M^{-1} k) (regularised leverage / D-optimal design).
    This is well-defined for |S| > d (unlike the literal Gram volume, which
    saturates at rank d) and grabs distinctive/spanning directions, ignoring
    density. Returns a full pick order (n,).
    k: if given, stop after k picks (prefix identical to the full order's prefix).
    """
    n, d = K.shape
    kk = n if k is None else min(k, n)
    eps = eps_frac * float((K * K).sum(1).mean())   # scale-aware ridge
    Minv = torch.eye(d, dtype=K.dtype, device=K.device) / eps
    avail = torch.ones(n, dtype=torch.bool, device=K.device)
    order = []
    for _ in range(kk):
        lev = (K @ Minv * K).sum(1)              # k_i^T M^{-1} k_i
        gain = torch.log1p(lev)
        gain[~avail] = float("-inf")
        u = int(gain.argmax())
        order.append(u)
        avail[u] = False
        kv = K[u]
        Mk = Minv @ kv                           # Sherman-Morrison rank-1 update
        Minv = Minv - torch.outer(Mk, Mk) / (1.0 + float(kv @ Mk))
    return torch.tensor(order, dtype=torch.long, device=K.device)


def keydiff_order(K: torch.Tensor) -> torch.Tensor:
    """KeyDiff-proper order -- the Stage 3 baseline diversity selector.

    Anchor = mean of L2-normalised keys. KeyDiff evicts the keys most aligned with
    the anchor (highest cosine = most redundant) and RETAINS the most distinctive
    (lowest cosine). Returns a pick order (n,) from most distinctive to least.
    Direction-based (cosine), query-free, content-only.
    """
    Kn = K / K.norm(dim=1, keepdim=True).clamp_min(1e-12)
    anchor = Kn.mean(0)
    cos = Kn @ anchor
    return torch.argsort(cos)                    # ascending cosine = distinctive first


def diversity_select(K: torch.Tensor, b: int) -> torch.Tensor:
    b = min(b, K.shape[0])
    return torch.sort(diversity_order(K)[:b]).values


def kcenter_select(K: torch.Tensor, b: int) -> torch.Tensor:
    return diversity_select(K, b)


def logdet_select(K: torch.Tensor, b: int, eps_frac: float = 1e-3) -> torch.Tensor:
    b = min(b, K.shape[0])
    return torch.sort(logdet_order(K, eps_frac)[:b]).values


def keydiff_select(K: torch.Tensor, b: int) -> torch.Tensor:
    b = min(b, K.shape[0])
    return torch.sort(keydiff_order(K)[:b]).values


def coverage_select(K: torch.Tensor, b: int, bandwidth_pct: float = 0.10) -> torch.Tensor:
    b = min(b, K.shape[0])
    return torch.sort(coverage_order(K, bandwidth_pct)[:b]).values
