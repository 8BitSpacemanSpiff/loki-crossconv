"""Attention-output-error metric Δ (CAOTE-style) + oracle/random yardsticks.

HARD RULE 1: the verdict metric is the closed-form change in attention output
under eviction, NOT relL2. HARD RULE 2: the oracle peeks at eval-query attention
and is a yardstick only -- never reported as a deployable method.

  attention_output_error(Q, K, V, S):
      || softmax(Q K^T) V  -  softmax(Q K_S^T) V_S ||_2 , averaged over queries,
      with the restricted softmax RE-NORMALISED over S only.

  oracle_set : top-b keys by true held-out attention mass (the yardstick floor).
  random_set : random b-subset, averaged outside (the ceiling).
  normalize  : map a raw Δ into [oracle=0, random=1].

Stage 1 reuses these with real keys/queries/values; Stage 0 feeds them the
synthetic head below, whose best-retention set is known.
"""
from __future__ import annotations

import torch

from .common import DTYPE, rng


def attention_output(Q, K, V):
    A = torch.softmax(Q @ K.T, dim=1)
    return A @ V


def attention_output_error(Q, K, V, S) -> float:
    full = attention_output(Q, K, V)
    A_s = torch.softmax(Q @ K[S].T, dim=1)   # re-normalised over S
    restr = A_s @ V[S]
    return float((full - restr).norm(dim=1).mean())


def oracle_set(Q, K, b) -> torch.Tensor:
    """Top-b keys by true held-out attention mass. Yardstick only (HARD RULE 2)."""
    mass = torch.softmax(Q @ K.T, dim=1).sum(0)
    idx = torch.topk(mass, min(b, K.shape[0])).indices
    return torch.sort(idx).values


def random_set(n, b, seed) -> torch.Tensor:
    g = rng(seed)
    return torch.sort(torch.randperm(n, generator=g)[: min(b, n)]).values


def random_delta(Q, K, V, b, n_draws=16, seed0=1000) -> float:
    n = K.shape[0]
    vals = [attention_output_error(Q, K, V, random_set(n, b, seed0 + i)) for i in range(n_draws)]
    return float(sum(vals) / len(vals))


def normalize_delta(delta, d_oracle, d_random) -> float:
    return float((delta - d_oracle) / (d_random - d_oracle + 1e-30))


def make_mixed_head(n_per_mode=40, k_modes=6, d=16, n_needles=8,
                    blob_std=0.25, blob_sep=4.0, needle_radius=12.0,
                    needle_query_scale=6.0, mode_query_scale=6.0,
                    needle_mass_boost=2.0, seed=0):
    """A synthetic head with a KNOWN best-retention set, for metric criterion (iii).

    Keys: k tight sub-blobs (bulk) + planted needles (far outliers). Values = keys.
    Queries: some point at needle directions (retrieval mass, the dominant mass),
    the rest point at sub-blob centres (bulk mass). This makes a strict ordering
    realisable:
        oracle  : keeps the high-mass needles AND sub-blob centres
        diversity: keeps the needles (dominant mass) + spreads  -> low Δ
        coverage : keeps sub-blob centres, DROPS needles        -> medium Δ
        random   : keeps neither well                           -> high Δ
    """
    g = rng(seed)
    centers = torch.randn(k_modes, d, generator=g, dtype=DTYPE) * blob_sep
    n_bulk = n_per_mode * k_modes
    labels = torch.arange(n_bulk) % k_modes
    K_bulk = centers[labels] + torch.randn(n_bulk, d, generator=g, dtype=DTYPE) * blob_std
    dirs = torch.randn(n_needles, d, generator=g, dtype=DTYPE)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    K_needle = dirs * needle_radius
    K = torch.cat([K_bulk, K_needle], 0)
    n = K.shape[0]
    needles = torch.zeros(n, dtype=torch.bool)
    needles[n_bulk:] = True
    V = K.clone()

    # queries: one per needle (scaled up + boosted -> dominant mass) + one per mode
    Q_needle = (dirs * needle_radius)
    Q_needle = Q_needle / Q_needle.norm(dim=1, keepdim=True) * needle_query_scale * needle_mass_boost
    Q_mode = centers / centers.norm(dim=1, keepdim=True) * mode_query_scale * blob_sep
    Q = torch.cat([Q_needle, Q_mode], 0)
    return {"K": K, "Q": Q, "V": V, "needles": needles, "kind": "mixed_head"}


def make_skewed_head(d=16, n=320, skew=4.0, qscale=0.08, seed=0):
    """Skewed bulk head where COVERAGE wins at an AGGRESSIVE budget (Phase A).

    A single heavy one-sided-skewed cloud under near-uniform attention. With flat
    attention, Δ(S) ~ || mean(V) - mean(V_S) ||: the diversity selector over-weights
    the skew tail (lonely keys that carry little mass), biasing its retained mean;
    the coverage selector stays representative. Robust only at aggressive budgets --
    at moderate budgets farthest-point sampling is itself near-optimal and the two
    selectors converge (Stage 0 Finding 1).
    """
    g = rng(seed)
    base = torch.randn(n, d, generator=g, dtype=DTYPE)
    u = torch.randn(d, generator=g, dtype=DTYPE)
    u = u / u.norm()
    mag = (torch.exp(torch.randn(n, generator=g, dtype=DTYPE)) - 1.0) * skew
    K = base + u * mag[:, None]
    V = K.clone()
    qi = torch.randperm(n, generator=g)[:64]
    Q = K[qi] * qscale  # low temperature -> near-uniform attention
    needles = torch.zeros(n, dtype=torch.bool)
    return {"K": K, "Q": Q, "V": V, "needles": needles, "kind": "skewed_head"}


def make_bulk_head(d=16, k_modes=6, n=300, blob_std=0.4, blob_sep=4.0,
                   qscale=1.5, seed=0):
    """No-needle dense multi-blob head with PEAKED attention (taxonomy study).

    k tight Gaussian blobs, no outliers. Queries are a density-weighted key sample
    at a temperature that keeps attention peaked (so the top-b-by-mass oracle is a
    genuine floor, unlike the near-uniform skewed head). Used for the decisive
    moderate-budget test: do log-det / KeyDiff diverge from facility-location where
    k-center (FPS) collapsed onto it?
    """
    g = rng(seed)
    centers = torch.randn(k_modes, d, generator=g, dtype=DTYPE) * blob_sep
    labels = torch.arange(n) % k_modes
    K = centers[labels] + torch.randn(n, d, generator=g, dtype=DTYPE) * blob_std
    V = K.clone()
    qi = torch.randperm(n, generator=g)[:64]
    Q = (K[qi] + torch.randn(64, d, generator=g, dtype=DTYPE) * blob_std) * qscale
    needles = torch.zeros(n, dtype=torch.bool)
    return {"K": K, "Q": Q, "V": V, "needles": needles, "kind": "bulk_head"}
