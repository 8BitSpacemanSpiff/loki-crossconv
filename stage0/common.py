"""Shared tensor utilities for Stage 0 (synthetic instrument validation).

Pure torch, CPU, float64 for numerically stable eigen / leverage. Deterministic:
every randomized routine takes an explicit integer seed. These helpers are written
so the selectors and the attention-output metric can be reused unchanged on real
Mistral keys/queries in Stage 1 (just pass GPU tensors).
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


def rng(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def pairwise_sq_dist(X: torch.Tensor) -> torch.Tensor:
    """(n, n) squared Euclidean distances, clamped to be non-negative."""
    sq = (X * X).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return d2.clamp_min_(0.0)


def pairwise_sq_dist_xy(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """(n, m) squared distances between rows of X (n,d) and Y (m,d)."""
    sx = (X * X).sum(1)
    sy = (Y * Y).sum(1)
    d2 = sx[:, None] + sy[None, :] - 2.0 * (X @ Y.T)
    return d2.clamp_min_(0.0)
