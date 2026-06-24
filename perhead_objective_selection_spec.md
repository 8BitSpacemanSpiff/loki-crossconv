# Per-Head Objective Selection for Query-Free KV Eviction
### Go/No-Go experiment spec + query-aware improvement catalogue
**Track:** KV-cache eviction (kept separate from the CrossCov-U selection paper)
**Substrate:** Mistral-7B, existing per-head calibration data, B=128 block processing (SimBin-compatible)

---

## Part A — Per-head objective-selection experiment (query-free)

### A.0 The claim being tested

Two query-free clustering objectives sit at opposite poles, and **each is half-right**:

- **Diversity** (KeyDiff / log det(KKᵀ) / k-DPP MAP / farthest-point): keeps geometric
  outliers, drops bulk. Retains needles; over-weights lonely keys that cover nothing.
- **Coverage** (facility location / submodular max-coverage): keeps cluster centers so every
  evicted key has a near retained substitute. Minimizes average reconstruction; **drops needles**
  — the same failure your R_q analysis already flagged ("rewards bulk-aligned keys, penalizes
  selective needles, the inverse of what irreversible eviction needs").

Neither single global objective is correct because the right objective is **head-dependent**:
retrieval/needle heads want diversity, local/bulk heads want coverage. The novel, query-free,
not-yet-occupied move is therefore **not a new clustering algorithm** but a *selector*: pick the
objective per (layer, head) from a cheap query-free statistic of the key cloud.

> **H1 (predictiveness):** A query-free geometry statistic g_h of the key cloud predicts which
> objective (diversity vs coverage) incurs lower attention-output error on head h.
>
> **H2 (exploitability):** Routing each head to its predicted objective beats *both* fixed
> objectives at matched budget, and closes a non-trivial fraction of the gap to the oracle.

If H1 fails, the adaptive story is dead and you stop — cheaply, before any harness build.

---

### A.1 Metric and instruments (validate before trusting real heads)

**Metric — attention-output error, NOT relL2.** relL2 already failed the KeyDiff-beats-H2O
sanity check; do not reuse it for the per-head verdict. Use the closed-form change in attention
output under eviction (CAOTE-style), which is the quantity eviction actually perturbs:

```
For head h, retained set S at budget b, held-out eval query q:
    Δ_h(S, q) = || softmax(q Kᵀ)·V  −  softmax(q K_Sᵀ)·V_S ||₂      (S re-normalized over softmax)
    Δ_h(S)    = E_q[ Δ_h(S, q) ]   over held-out eval queries
```

**Oracle is ground-truth ONLY, never a method.** Define the oracle retained set S* per head as
the budget-b set minimizing Δ_h (or, tractably, top-b keys by true held-out attention mass). The
oracle *peeks at eval-query attention*, so it is a yardstick for evaluating the two query-free
selectors — it is **not** a deployable baseline and must never be reported as one (same trap as
"KeyDiff+MASS"). Report every selector as Δ_h(selector) normalized into [Δ_h(oracle),
Δ_h(random)] so "win" means measurable distance closed toward the oracle, not a raw number.

**Stage 0 — instrument validation on synthetic ground truth (gate before real data).**
Build synthetic key clouds with *known* structure and confirm each instrument behaves as designed:

| Synthetic cloud | Diversity selector should… | Coverage selector should… | g_h should read… |
|---|---|---|---|
| k tight Gaussian blobs, no outliers | keep ≥1 per blob, waste budget on within-blob spread | keep blob centers, evict bulk efficiently | low effective rank / low outlier frac |
| 1 dense blob + planted needle outliers | retain the needles | drop the needles | high outlier frac / heavy spectral tail |
| uniform shell (no structure) | ≈ random | ≈ random | mid, low clusteredness |

Pass criteria: (i) the two greedy selectors split the planted needles in the predicted
directions; (ii) g_h takes separable values across the three clouds; (iii) Δ on a synthetic head
with a *known* best-retention set ranks oracle < better-objective < worse-objective < random.
**If the instruments don't pass synthetic ground truth, fix them before touching Mistral.**

---

### A.2 Candidate query-free statistics g_h

Compute several cheap candidates per head; let Stage 2 choose the most predictive **on held-out
heads** (don't hand-pick — that overfits the statistic choice):

- **Effective rank / participation ratio** of the key Gram spectrum:
  `PR = (Σλ_i)² / Σλ_i²`. Low ⇒ few modes (coverage-friendly); high ⇒ spread (diversity matters).
- **Spectral-tail heaviness:** mass in the eigenvalue tail beyond the knee. Heavy tail ⇒ isolated
  directions ⇒ needles present.
- **Outlier fraction:** share of keys with large k-NN distance / high leverage score
  (`h_ii` of the key hat matrix). This is the most directly causal candidate for needle presence.
- **Clusteredness:** between- vs within-cluster variance ratio from a cheap k-means / gap statistic.

All four are query-free and computable from existing calibration keys. Compute **pre-RoPE** so
"distinctiveness" reflects content geometry, not position (carry over from the prior KeyDiff notes).

---

### A.3 Stage 1 — per-head regime measurement

For each (layer, head) on real Mistral, at a fixed budget (start 50% eviction; sweep later):

1. Compute g_h (all candidates).
2. Run the **diversity** selector (greedy farthest-point / max-Δlog-det) → S_div.
3. Run the **coverage** selector (greedy facility location, `f(S)=Σ_v max_{u∈S} sim(v,u)`) → S_cov.
4. Score Δ_h(S_div), Δ_h(S_cov), Δ_h(oracle), Δ_h(random) on held-out eval queries.
5. Label the head: `winner_h = argmin(Δ_h(S_div), Δ_h(S_cov))`, plus the margin
   `m_h = |Δ_h(S_div) − Δ_h(S_cov)| / (Δ_h(random) − Δ_h(oracle))` (normalized separation).

Output: a per-head table {g_h candidates, winner_h, m_h, normalized Δ for each selector}.

---

### A.4 Stage 2 — predictiveness gate (the GO/NO-GO)

Fit a threshold/logistic map `g_h → winner_h` on **half** the heads, evaluate on the held-out
half. Two numbers decide everything:

- **AUC** of g_h → winner_h on held-out heads.
- **Adaptive vs fixed gap:** mean normalized Δ for {always-diversity, always-coverage,
  per-head-adaptive(predicted), per-head-adaptive(oracle-routed)} aggregated over heads.

```
GO   if  AUC ≥ 0.70 on held-out heads
     AND  Δ̄(adaptive-predicted) < Δ̄(better fixed objective) by a margin outside head-resampling noise
     AND  Δ̄(adaptive-oracle-routed) shows real headroom (the ceiling is worth chasing)

NO-GO if  AUC ≈ 0.5 (no query-free statistic separates the regimes)            → adaptive story dead
      OR  most heads have small m_h (the two selectors agree → nothing to route) → no exploitable regime
      OR  adaptive-predicted ≥ better fixed objective (prediction can't be cashed in)
```

Any NO-GO branch kills the direction with one cheap pass over existing calibration data and no
harness work. That's the point of gating here.

---

### A.5 Stage 3 — downstream validation (only if Stage 2 = GO)

Implement per-head adaptive eviction in the cold-compress harness and compare at **matched
budgets** against KeyDiff and plain SimBin:

- **Needle-sensitive:** RULER (multi-needle), Musique — where diversity-win heads should matter.
- **Bulk/uniform:** a long-context summarization or LongBench-style task — coverage-win heads.
- Report per-budget curves; the win condition is adaptive ≥ max(KeyDiff, SimBin) on the union and
  strictly better in at least one regime, with the per-head routing explaining *why*.

---

### A.6 Design fork — do not delete your own novelty

The apportionment layer (Webster / Sainte-Laguë across buckets) is the part of SimBin unlike
anything in the survey. A greedy facility-location / coverage selector picks the global budget
directly and **does not need cross-bucket apportionment** — so a naive clustering swap can dissolve
the one distinctive component you have. Decide the paper's identity up front:

- **Path 1 — "apportionment over clusters":** keep bucketing as the unit, make the *objective per
  bucket* and the *apportionment rule* the contribution. Adaptive selection lives inside/above
  apportionment. Preserves SimBin's distinctive layer.
- **Path 2 — "greedy adaptive coverage/diversity eviction":** drop apportionment, route per head
  between two greedy selectors. Cleaner story, but more crowded base and you lose the apportionment
  wedge.

Pick before building; they are different papers.

---

### A.7 Reuse / cost

Stages 0–2 reuse the existing Mistral calibration keys and the pending per-head ρ_h vs relL2
scatter scaffolding (same data, swap relL2 → attention-output error, add the four g_h statistics
and the two greedy selectors). No new model runs until Stage 3. H100 + 2TB is ample.

---

## Part B — Query-aware improvement catalogue (from the attached prior discussion)

These are the **query-information** levers raised earlier. They are the *complement* to Part A:
Part A maps the query-free axis; Part B maps the query-aware axis. Listed for completeness and so
the two threads stay legible. (Q) = uses query information; (—) = query-free, included for context.

### B.1 KeyDiff family

1. **(Q) Query-conditioned diversity — the headline hybrid.** One line on KeyDiff's score:
   `evict argmax_k [ cos(k, anchor) − λ · relevance(k, q̄) ]`.
   λ=0 recovers KeyDiff; λ>0 only evicts keys that are *both* redundant *and* query-irrelevant,
   using HashEvict's cheap SimHash term for the relevance part to stay near attention-free. Targets
   the **union** of KeyDiff's and HashEvict's documented failure regimes — build the benchmark
   deliberately (distinctive-answer tasks ∪ non-distinctive-answer tasks). Most defensible because
   it attacks a *documented* weakness rather than hoping for uniform gain.

2. **(—) Multi-anchor KeyDiff = SimBin ∘ KeyDiff.** Bucket first (SimBin), apply KeyDiff's
   anchor-diversity *within each bucket*. Fixes the single-mean-anchor weakness when keys are
   multimodal; KeyDiff's optimality argument then holds per-cluster. Composition, not query info.

3. **(—) Magnitude / asymmetric-LSH (MIPS→cosine).** Make Hamming distance estimate dot-product,
   not angle. Treat as a hypothesis with a finding either way: KeyDiff normalizes keys and still
   nearly matches baseline, so magnitude may not buy much. Cheap ablation, not a sure win.

   *Smaller swings:* per-head eviction budgets; pre-RoPE distinctiveness; value-side redundancy
   (two near-identical *values* are redundant regardless of keys); learned relevance projection
   (powerful but **breaks the training-free property** — different category of paper).

### B.2 SimBin family (borrowed from CTkvr)

1. **(Q) Query-aligned projection planes instead of random ones — the headline.** Derive the
   SimHash planes from the query subspace (top principal directions of recent queries, or q̄ and its
   spread) so `sign(K · q_principal)` buckets keys by how they'd actually score against queries.
   Data-dependent LSH; keeps all of SimBin's machinery (bit-packing, O(1) assignment, within-bucket
   eviction), just points the planes somewhere useful. **Note:** you forfeit SimHash's clean
   collision guarantee (1−θ/π) once planes go data-dependent — the paper then has to win empirically.

2. **(Q) Adjacent-query smoothness → EMA plane updates.** The objection to #1 is "recompute planes
   every step." CTkvr's adjacent-query similarity (and the centroid-drift result) says queries drift
   smoothly, so update plane directions with a slow EMA, not per step — importing relevance without
   HashEvict's per-step scan cost.

3. **(partly Q) Two-stage coarse→fine within-bucket decision.** The 8-bit bucket is the coarse
   (diversity) stage; *within* a retained bucket replace SimBin's evict-oldest with a fine criterion
   — extra hash bits, key magnitude, or a small query-relevance term. "Evict oldest" is the weakest
   part of SimBin, so this alone likely moves the needle.

4. **(—) Magnitude as within-bucket tiebreak.** Keep per-key ‖k‖, evict low-norm first (or full
   asymmetric-LSH). Same caveat as B.1.3 — cheap ablation, test don't assume.

### B.3 How Part A and Part B relate

- **Part A (query-free, per-head objective routing)** and **B.1.1 / B.2.1 (query-aware)** are the
  two independent axes. They are composable: a per-head router (A) could pick not just diversity vs
  coverage but also *whether to switch on* the query-aware term (B), giving a 2×2 of objectives.
- The honest boundary from the prior survey still holds: query-aligned planes + fine within-bucket
  relevance is the slope toward "this is just CTkvr." What keeps any of this distinct is that you
  remain in the **eviction** setting (permanent drop, low memory, KeyDiff's regime), whereas CTkvr
  is **retrieval** (keep-and-recall). Don't let the borrows erode that boundary.
