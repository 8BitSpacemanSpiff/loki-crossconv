# CLAUDE.md — KV-Cache Eviction (Per-Head Objective Selection)

## RESUME (read this FIRST on a cold instance — 2026-06-25)

State: Stage 0 + Phase A/B/C + resume-checks + **Phase D (STEP 0, 0b, STEP 1+2)** are done and
committed. We are **STOPPED at the Stage-2 gate** (the per-head routing AUC fit) awaiting a GO or a
reframe decision — see "Phase D verdict" below. The 49GB calibration artifact is **GONE** (spot
instance, never pushed — too large); re-emit it first. Everything else (code, results `.md`/`.csv`/
`.png`, the small `outputs/phase_d_raw.pt` (452K), decisions) IS pushed.

**FIRST STEP on resume — re-emit the artifact, then re-validate before trusting any Δ:**
1. `python -m stage0.emit_calibration --seq-len 8192 --n-seq 32 --out-dir outputs/calib`
   (~3 min on a warm model cache, ~25 min cold incl. the ~14GB download; c4, 8192, 32 seqs → ~49GB).
2. Contract: `python -m stage0.verify_calibration --dir outputs/calib` → `CONTRACT OK across 32 layers`.
3. Faithfulness: `python -m stage0.validate_emit` → `VERDICT: PASS` (weights/outputs ~fp16,
   artifact-tie ≈ 0). Do NOT trust any Phase C/D Δ or oracle label until 2 AND 3 pass.

**ENV / HARDWARE (this session was an A100-80GB, NOT the H100/H200 named below):**
- `.venv/` is gitignored; rebuild per `RUNBOOK.md`. This box had **torch 2.1.0+cu121,
  transformers 4.40.2**, needs **numpy<2** + sentencepiece. Because torch<2.1.1, emit + validate
  use **`attn_implementation="eager"`** (not sdpa) — provably inert (captured K/Q/V come from
  pre-attention hooks; artifact-tie was 0.00e+00). Always run as `python -m stage0.X` from repo root
  (a bare `python stage0/x.py` shadows stdlib `selectors` with `stage0/selectors.py`).
- Model `mistral-community/Mistral-7B-v0.2` (ungated; mistralai never published a base v0.2 repo).

**CENTERING IS THE STAGE-0 STORY (supersedes the old "rank-1 / no volume" framing):**
- The apparent rank-1 K_pre (eff-rank ~1.4) was a **shared mean / attention-sink DC offset**, NOT
  intrinsic. Centered rank is ~9–27 (layer 0 is the only genuine near-rank-1). The "logdet dies
  because there's no volume to maximize" mechanism is **DEAD** (`phase_d_step0_centered_rank.md`).
- Euclidean coverage selectors (`facility`, `kcenter`) are **translation-invariant** → centered ==
  uncentered; their wins are real. Only the inner-product selectors move under de-meaning: uncentered
  `keydiff` rode the sink (collapses 17→2 wins when centered), `logdet` was buried by it (recovers
  0→5). See `phase_d_step0b_centering_ablation.md`.

**MENU / ROLES (settled in STEP 0b; do not re-litigate):**
- Routing label (Stage-2 target) = `argmin{facility, kcenter}` on de-meaned content geometry.
- `keydiff_unc` (raw keys) = the deployable **BASELINE to beat**, reported as a column, not routed.
- `keydiff_cen` + `logdet_cen` = **reference columns** (document the sink-collapse / diversity-stays-
  weak), not routing candidates. `logdet` is **dropped** as a candidate.

**PHASE D VERDICT (full 256-head measurement; `phase_d_step1_measurement.md`, `..._step2_geometry.md`,
`phase_d_perhead.csv`, 3 PNGs) — the data does NOT cleanly support per-head geometry routing:**
- Routing is **budget-driven**, not per-head: @3% kcenter 232 / facility 23 (kcenter 91%);
  crossover to facility as budget grows (10%: 113/140; 25%: 161/90; 50%: 163/89).
- No budget has both balanced classes AND stable labels: @3% stable (22/255 flip) but DEGENERATE
  (always-kcenter ≈ 91%); @10% classes balance but UNSTABLE (83/253 flip across the held-out split).
- **Baseline dominates:** routed `min{fac,kc}` beats `keydiff_unc` on only 30/255 heads @3%
  (8/252 @50%) — coverage loses to deployable KeyDiff on ~88% of heads.
- `g_h` (PR, spectral_tail, outlier_fraction, clusteredness on centered K_pre) separate the winner
  classes only weakly / overlapping on a 23-vs-232 imbalance. No clean boundary in the scatter PNGs.
- **Open decision for the human (do NOT start Stage 2 without it):** (a) reframe to the budget-
  dependent kcenter→facility crossover (what the data supports); (b) reckon with KeyDiff beating the
  coverage menu on 88% of heads; or (c) fit the AUC only at 10% (the one balanced budget) while
  owning the 33% label instability.

**STALE-SPEC WARNING — older notes below are superseded by the above:**
- The spec/hard-rules reference a repo `--emit-rq` forward pass. **It does not exist**; emit was
  built from scratch (`stage0/emit_calibration.py`). Ignore `--emit-rq`.
- Phase A *synthetic* selector rankings are STALE for real data (see CENTERING + VERDICT above).
- Phase C (`phase_c_probe.md`, uncentered keydiff 17 / facility 15 / kcenter 7, logdet 0) is
  **superseded** by Phase D STEP 0b/STEP 1: the uncentered keydiff strength was the sink artifact.

## What this project is
Research track: **KV-cache eviction**. This is *separate* from the CrossCov-U sparse-attention
selection paper — do not pull ideas, code, or framing across the two. If a task starts drifting
toward selection/CrossCov-U, stop and flag it.

## Source of truth
`perhead_objective_selection_spec.md` (in this repo) is the spec. Follow its staged structure and
its **go/no-go gates** literally. Part A (query-free per-head objective routing) is the experiment.
Part B (query-aware levers) is a *catalogue for reference only* — it is NOT part of this build.

## Hard rules (do not violate, do not "improve around")
1. **Metric = attention-output error** (closed-form Δ attention output, CAOTE-style).
   **Never use relL2** for any verdict — it failed the KeyDiff-beats-H2O sanity check and is
   unreliable here.
2. **The oracle is a yardstick, never a method.** The oracle peeks at eval-query attention; use it
   only to normalize Δ into [oracle, random]. Never report it as a deployable baseline (the
   "KeyDiff+MASS" trap).
3. **Validate instruments on synthetic ground truth first (Stage 0).** No conclusions from real
   Mistral data until the two greedy selectors, the geometry statistics, and the error metric all
   pass the synthetic checks in A.1.
4. **Part A stays query-free.** No q̄, no query-derived projection planes here. Those are Part B,
   a different thread.
5. **Compute geometry statistics pre-RoPE.** Distinctiveness must reflect content, not position.
6. **Decide the apportionment fork (A.6) before writing the selector.** Path 1 keeps the
   Webster/Sainte-Laguë apportionment layer; Path 2 drops it for greedy selection. Do not silently
   delete apportionment — it is the most distinctive component. Ask which path if unspecified.
7. **Gates are stop points.** At the Stage 2 AUC gate, STOP and report numbers. Do not start Stage 3
   harness work without an explicit GO.

## Working discipline
- Self-test every instrument against known synthetic ground truth before trusting it on real data.
- Use held-out split validation (fit thresholds on half the heads, test on the other half).
- Stage 0 is **synthetic** — no model, no dataset. Run it on the bare VM immediately.
- Stages 1–2 need the per-head Mistral calibration artifact (one `--emit-rq` forward pass to collect
  keys/queries); after that the analysis is offline.
- **No expensive eval-harness runs (RULER/Musique) until Stage 3** — that is the costly part the gates
  exist to protect.
- Write deliverables (results tables, plots, configs) to the repo `outputs/` dir.
- Report at each gate with the actual numbers (AUC, per-selector normalized Δ, margins), not prose
  summaries.

## Environment
- Hardware: H100, 2TB system RAM. Claude Code itself needs no GPU; the experiment code uses the H100.
- **Fresh VM: model and dataset are NOT downloaded yet.** Bootstrap is required before Stage 1 — but
  NOT before Stage 0 (Stage 0 is synthetic and runs on a bare Python/torch env).
- Model: `mistralai/Mistral-7B-v0.2`. Rotary: prerotary. Calibration/transform dataset: c4.
  Eval: wikitext-test. B=128 blocks (SimBin-compatible).
- Calibration: per-head keys/queries come from the repo's `--emit-rq` forward pass. If that artifact
  is missing, scoring silently falls back to key-norm — and key-norm is a dead proxy here
  (corr ≈ −0.21), so confirm the artifact exists before trusting any Stage 1 number.
- Repo: loki-crossconv fork (this checkout).

## Decisions

**Apportionment fork (A.6) — decided: decouple measurement from productization.**
- Stages 0–2 (measurement): run the two objectives as **global greedy selectors per head**,
  apportionment-agnostic. This isolates the routing hypothesis (does per-head geometry predict
  diversity-vs-coverage?) without entangling it with apportionment mechanics — the cleanest diagnostic.
- Stage 3 (productization, only on a GO): **Path 1** — build the method *over* SimBin's buckets and
  keep the Webster/Sainte-Laguë apportionment layer. Do NOT collapse to global greedy (Path 2); that
  silently deletes the most distinctive component. Override only with an explicit, stated reason.

**Calibration artifact — emitted+validated Phase B, then LOST with the spot instance. NOT saved.**
- **Re-emit first on resume** (see RESUME block at top). It is gitignored (`outputs/calib/`, ~49GB)
  and was never pushed. Path when present: `/home/loki-crossconv/outputs/calib/`.
- Built from scratch via hooks in `stage0/emit_calibration.py` (monkeypatch `apply_rotary_pos_emb`
  + `v_proj` hooks); no `--emit-rq` path exists. No scoring in emit → key-norm fallback is
  structurally impossible.
- Contents per (layer, KV-head): `K_pre`, `K_post`, `V` over the cloud `keys[0:P]` (P=7936), and
  `Q_oracle`/`Q_eval` = disjoint halves of the last 256 query positions for the 4 GQA-group heads.
  c4, seq_len=8192, n_seq=32, fp16. Verified by `verify_calibration.py` + `validate_emit.py`.
- **Model:** `mistral-community/Mistral-7B-v0.2` (ungated canonical base conversion; mistralai never
  published a base v0.2 repo — identical weights: rope_theta 1e6, no sliding window, 32L/32H/8KV/d128).

**Phase C result (real-head probe, `outputs/phase_c_probe.md`) — supersedes synthetic Phase A.**
- Canonical facility check PASS (coverage selector sound at 10th-pct bandwidth).
- Real heads: logdet 0 wins, keydiff 17, facility 15, kcenter 7 (of 39 valid). Routable=YES.
- Phase D menu UNDECIDED: carry keydiff+facility+kcenter, drop logdet; needs fuller averaging + g_h.
