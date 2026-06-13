# CrossCov extensions: KV compression and KV eviction

Two runtime modes added on top of the existing CrossCov-U path. Both reuse the
same per-head basis from `pca_analysis/crosscov.py`; neither changes the offline
basis construction. They are selected with `--crosscov-mode`.

```
--use-crosscov --crosscov-mode select     # existing: full cache, latent top-k selection
--use-crosscov --crosscov-mode compress   # rank-r cache, selective reconstruction (SALS-style)
--use-crosscov --crosscov-mode evict      # static query-weighted score-energy eviction
```

## compress
Selection happens in the r-dim latent space (cheap), but the gathered attention
scores are recomputed from the **reconstructed** keys `K_hat = K U_r U_r^T`, not
the exact keys. This is the faithful accuracy cost of storing only the rank-r
cache, so the PPL/recall numbers reflect a real compressed deployment rather than
a full-cache upper bound. Logs `kv_compress_ratio = r/head_dim`. Works with the
existing `--log-recall` / `--log-mass-recall` diagnostics.

Positioning: this is the four-way basis story made runnable — Loki (per-head KᵀK),
SALS (joint-head KᵀK), SWAN (QᵀQ+KᵀK), CrossCov (QᵀK). Against SALS specifically,
the difference is the basis, not the pipeline.

## evict
Each token gets a query-**agnostic** importance `s_j = k_j^T Rq k_j`, the expected
query energy of its score, with `Rq = E[q qᵀ]` pooled over the GQA group. Computed
in latent coords as `k~_j^T G k~_j`, `G = U_rᵀ Rq U_r`. A static retained set keeps
`--sink-tokens` initial + `--recent-window` final tokens and the highest-score
middle tokens up to `1 - --evict-ratio`; evicted columns are masked for all query
rows (irreversible). Logs `evict_mass_kept` (true attention mass retained).

This is the zero-bit limit of the query-weighted score-energy allocation in the
derivation note — the same `Rq`-weighted statistic that drives mixed-precision
allocation, here used as a keep/drop decision. It is differentiated from KeyDiff
(query-agnostic key *geometry*, no `Rq`) and from H2O/SnapKV (need observed
attention scores). Eviction is irreversible, so expect it to lag `select`/`compress`
on retrieval-heavy tasks (cf. SALS RULER MK2 at aggressive ratios) — best used as
a complement that drops the clearly-dead tail.

## Offline: emit Rq for eviction
Eviction needs the per-head query second moment. Add `--emit-rq` to the existing
calibration call; it writes `<out>/key/rq_gram/rq_gram_layer_{l}.pt`
(one pooled `head_dim x head_dim` `Rq` per KV head). If absent at runtime the evict
path falls back to an identity gram (pure latent key-norm eviction) with a warning.

```
python pca_analysis/crosscov.py <num_layers> <tensor_root> <out_dir> --pool-gqa --emit-rq
```

## What is verified vs pending
- **Verified here** (`tests/test_crosscov_ext.py`, numpy, no GPU): rank-r
  reconstruction error equals discarded singular energy; latent score-energy
  `k~ᵀ G k~` equals reconstructed-key energy `(Pk)ᵀ Rq (Pk)` and equals exact
  `kᵀ Rq k` at full rank; eviction ranking is invariant to latent coords; GQA
  pool equals the stacked-cross-cov left singular basis; Loki collapse (K=Q ⇒ U=V).
- **Pending on the H100**: PPL / LongBench / recall numbers for both modes on
  Mistral-7B, and the per-head `ρ_h` vs. relL2 correlation (experiment E2 from the
  derivation note). The torch paths are wired to the existing harness and compile,
  but have not been run against a model in this environment.

## Suggested first runs
1. `compress` at the established operating point (r=8 / kf=0.125), compare PPL to
   `select` at the same r — quantifies the reconstruction cost the compression adds.
2. `evict` at `--evict-ratio 0.3` with sinks+recent protection, log `evict_mass_kept`
   across layers — cheap sanity that the `Rq` scorer keeps high-mass tokens.
