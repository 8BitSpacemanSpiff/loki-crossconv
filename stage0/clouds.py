"""Synthetic key clouds with KNOWN ground truth (Stage 0, A.1 table).

Three structures, matching the validation table in the spec:

  1. gaussian_blobs   - k tight Gaussian blobs, no outliers
  2. blob_with_needles- 1 spread blob + planted needle outliers (indices known)
  3. uniform_shell    - points on a sphere shell, no cluster structure

Each generator returns a dict with at least:
  K        : (n, d) float64 key cloud
  needles  : (n,) bool mask of planted needle outliers (all-False if none)
  labels   : (n,) long cluster id, or None
  kind     : str

No RoPE, no model, no dataset. Geometry here is pure content (Stage 0 is the
pre-RoPE regime by construction).
"""
from __future__ import annotations

import torch

from .common import DTYPE, rng


def gaussian_blobs(n=256, d=16, k=8, blob_std=0.15, sep=4.0, seed=0):
    g = rng(seed)
    centers = torch.randn(k, d, generator=g, dtype=DTYPE) * sep
    labels = torch.randint(0, k, (n,), generator=g)
    K = centers[labels] + torch.randn(n, d, generator=g, dtype=DTYPE) * blob_std
    return {
        "K": K,
        "labels": labels,
        "needles": torch.zeros(n, dtype=torch.bool),
        "kind": "gaussian_blobs",
    }


def blob_with_needles(n=256, d=16, n_needles=8, blob_std=0.3, needle_radius=10.0, seed=0):
    g = rng(seed)
    K = torch.randn(n, d, generator=g, dtype=DTYPE) * blob_std  # one spread blob at origin
    needle_idx = torch.arange(n - n_needles, n)
    dirs = torch.randn(n_needles, d, generator=g, dtype=DTYPE)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    K[needle_idx] = dirs * needle_radius  # far out, mutually distant directions
    needles = torch.zeros(n, dtype=torch.bool)
    needles[needle_idx] = True
    labels = torch.zeros(n, dtype=torch.long)
    labels[needle_idx] = 1
    return {"K": K, "labels": labels, "needles": needles, "kind": "blob_with_needles"}


def uniform_shell(n=256, d=16, radius=1.0, seed=0):
    g = rng(seed)
    K = torch.randn(n, d, generator=g, dtype=DTYPE)
    K = K / K.norm(dim=1, keepdim=True) * radius
    return {
        "K": K,
        "labels": None,
        "needles": torch.zeros(n, dtype=torch.bool),
        "kind": "uniform_shell",
    }


def all_clouds(seed=0):
    """The three A.1 clouds at matched (n, d)."""
    return [
        gaussian_blobs(seed=seed),
        blob_with_needles(seed=seed),
        uniform_shell(seed=seed),
    ]
