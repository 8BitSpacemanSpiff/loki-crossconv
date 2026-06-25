# CLAUDE.md — KV-Cache Eviction (Per-Head Objective Selection)

## RESUME (read this FIRST on a cold instance — 2026-06-25)

State: Stage 0 + Phase A/B/C + resume-checks + **Phase D (STEP 0, 0b, STEP 1+2) + sink ablation**
are done and committed. **Routing-as-a-coverage-axis is DEAD** (per-head geometry routing failed;
sink ablation confirmed coverage is not competitive even sink-protected — see verdict below). The
live axis is **distinctiveness (keydiff) vs coverage**, pending ONE decisive check tomorrow:
**content-excluded Δ** (see CRITICAL OPEN ITEM). We are NOT at the Stage-2 AUC gate anymore — that
fork is closed. The 49GB calibration artifact is **GONE** (spot instance, never pushed); re-emit
first, and tomorrow re-emit it **WITH the massive-activation capture folded in**
(`phase_d_emit_addition_massive_activations.md`). Everything else (code, results, `phase_d_raw.pt`
452K, `phase_d_sink_raw.pt`, decisions) IS pushed.

**FIRST STEP on resume — re-emit (with the MA addition tomorrow), then re-validate before any Δ:**
1. `python -m stage0.emit_calibration --seq-len 8192 --n-seq 32 --out-dir outputs/calib`
   (~3 min warm cache, ~25 min cold incl. ~14GB download; c4, 8192, 32 seqs → ~49GB).
   **TOMORROW: first fold in massive-activation capture** (residual top-8 + per-dim max-abs) per
   `phase_d_emit_addition_massive_activations.md`, then re-emit.
2. Contract: `python -m stage0.verify_calibration --dir outputs/calib` → `CONTRACT OK across 32 layers`.
3. Faithfulness: `python -m stage0.validate_emit` → `VERDICT: PASS` (weights/outputs ~fp16,
   artifact-tie ≈ 0). Do NOT trust any Δ or oracle label until 2 AND 3 pass.

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

**MENU / ROLES (settled in STEP 0b):**
- `keydiff_unc` (raw keys) = the deployable **BASELINE to beat**.
- coverage menu = `facility`, `kcenter` (offset-invariant). `logdet` **dropped**; `keydiff_cen` +
  `logdet_cen` are reference columns. RULE GOING FORWARD: **every selector comparison MUST
  sink-protect ALL selectors** (always-keep pos 0-3) or facility is handicapped (it evicts sinks).

**PHASE D VERDICT — per-head geometry routing FAILED (`phase_d_step1_measurement.md`, `..._step2_geometry.md`,
`phase_d_perhead.csv`, 3 PNGs):**
- Routing is budget-driven not per-head: @3% kcenter 232 / facility 23; crossover to facility as
  budget grows (10%: 113/140; 25%: 161/90; 50%: 163/89). No budget is both balanced AND stable
  (@3% stable but degenerate ~91% kcenter; @10% balanced but 33% split-flip). `g_h` barely separates
  winner classes. Coverage routing is not a viable axis.

**SINK ABLATION VERDICT — DONE, 256 heads (`phase_d_sink_ablation.md`, `phase_d_sink_raw.pt`):**
- Protecting pos 0-3 equally: coverage beats `keydiff_unc` @3% only 35/256 (14%) → 64/256 (25%)
  sink-protected (@25%: 30→36). Sink-protection ~DOUBLES facility's wins but **KeyDiff still beats
  the coverage menu on 192/256 (75%) sink-protected** → routing-as-coverage-axis is DEAD; coverage
  is not competitive even after the sink fix.
- Mechanism (KeyDiff-win heads): mean **sink_mass ~0.54** (eval attn on pos 0-3); KeyDiff parks
  **~0.68** of retained mass on sinks; sinks kept of 4 = **kd ~3.0 / fac ~0.3 / kc ~1.4** (facility
  throws away the high-mass sinks). KeyDiff's edge is **sink-AUGMENTED, not sink-DEPENDENT**.

**>>> CRITICAL OPEN ITEM — DO THIS FIRST TOMORROW (before any reframe is "earned"): <<<**
The 75% is on **FULL Δ**, which still rewards the free sink mass BOTH methods keep. Re-run
`keydiff_unc` vs routed `min{facility,kcenter}` on **CONTENT-EXCLUDED Δ**: softmax renormalized over
NON-sink positions only (drop 0-3 from BOTH the softmax denominator AND the output sum), @3% and
@25%, **broken out by layer band L0-1 / L2-14 / L15+**. This is THE fork:
- gap HOLDS on content-excluded Δ → "reframe toward distinctiveness" is earned (real, not bookkeeping).
- gap DROPS to parity → the 75% was sink bookkeeping → fold eviction back in; distinctiveness story dies.
(All selectors still sink-protected. Δ stays on real uncentered K_post/V.)

**SECOND OPEN ITEM (possibly the most publishable):** the @25% norm reversal under sink-protection —
facility 165→93, kcenter 91→163. Unexplained budget × coverage-normalization interaction; needs a
clean look once the content-Δ fork is resolved.

**STALE-SPEC WARNING — older notes below are superseded by the above:**
- The whole `perhead_objective_selection_spec.md` per-head ROUTING premise is **DEAD**: logdet dead
  (Phase C), routing dead (Phase D + sink ablation). The only LIVE axis is **distinctiveness
  (keydiff) vs coverage**, and even that is provisional pending the content-excluded Δ check.
- The spec/hard-rules reference a repo `--emit-rq` forward pass. **It does not exist**; emit was
  built from scratch (`stage0/emit_calibration.py`). Ignore `--emit-rq`.
- Phase A *synthetic* selector rankings are STALE for real data (see CENTERING + VERDICT above).
- Phase C (`phase_c_probe.md`, uncentered keydiff 17 / facility 15 / kcenter 7, logdet 0) is
  **superseded** by Phase D STEP 0b/STEP 1: the uncentered keydiff strength was the sink artifact.

## SimBin > KeyDiff — tomorrow's second thread (do AFTER content-excluded Δ)

Hypothesis: sinks/massive-activation tokens explain why SimBin beats KeyDiff — but via MODES, not
buckets. Corrected mechanism:
- KeyDiff already KEEPS sinks (3.02/4) and won 75% partly because of it — so sink-retention is
  common to both and can't be SimBin's edge. SimBin's advantage must live in CONTENT tokens.
- "One leader per bucket" is wrong (too few outliers for that; massive-activation tokens share
  fixed-dim sign patterns so they collapse INTO a few buckets, not one-per-bucket). Right framing:
  KeyDiff's single GLOBAL mean anchor mis-ranks multimodal key clouds; SimBin keeps one rep per
  MODE. Sinks are the sharpest mode the anchor mishandles, not the whole story.

PREMISE-CHECK FIRST (don't theorize on a confounded delta): the SimBin>KeyDiff result may have been
measured on relL2 (flagged unreliable) or sink-mass Δ. Re-establish SimBin > KeyDiff under
sink-protection + content-excluded Δ BEFORE explaining it.

Two-part test on the re-emitted artifact (nearly free once massive-activation capture lands):
  (a) Are SimBin's retained tokens MORE enriched for sink/massive-activation tokens than KeyDiff's?
  (b) On content-excluded Δ, does SimBin still beat KeyDiff, and is it via better per-mode content
      coverage (not sink enrichment)?
  survives content-exclusion & not just sink enrichment -> multimodal-coverage story is right.
  edge IS the sink enrichment -> original sink intuition wins, KeyDiff's sink-keeping worse than it looks.

META: sinks/massive-activations are now load-bearing in THREE places (KeyDiff's win, the metric
confound, SimBin). Candidate real paper: "how each eviction method implicitly handles
massive-activation tokens, and why that drives the whole ranking" — subsumes all three comparisons,
sits in the outlier-dimension/quantization wheelhouse.

## KVEvict persistent storage (ops — set up tomorrow, FIRST, to stop re-emitting)
This box is a **Jarvislabs.ai** instance. The user created a persistent **Filesystem "KVEvict"**
(survives pause/resume/destroy) — but it was NOT attached today, so the 49GB calib still dies on
stop. Tomorrow:
1. Launch/resume the instance **WITH KVEvict attached** (control-plane; user's action):
   `jl create --gpu <type> --fs-id <KVEvict_id>` (CLI) or `Instance.create(..., fs_id=...)`. It
   mounts at **`/home/jl_fs/`**. (No `jl` CLI was installed on the box; attach can't be done from
   inside a running instance.)
2. Once `/home/jl_fs/` exists: re-emit calib ONCE into it (or `mv outputs/calib /home/jl_fs/calib`
   + symlink back), and stash the HF cache (`~/.cache/huggingface`) there too → future instances
   that attach KVEvict skip BOTH the ~25-min emit and the 14GB model download.
3. Update RUNBOOK so resume loads calib from `/home/jl_fs/calib` instead of re-emitting.
Docs: https://docs.jarvislabs.ai/cli , https://docs.jarvislabs.ai/vm/

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
