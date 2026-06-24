# CLAUDE.md — KV-Cache Eviction (Per-Head Objective Selection)

## RESUME (read this FIRST on a cold instance — 2026-06-24)

State: Stage 0 + Phase A + Phase B + Phase C are **done and committed**. The 49GB calibration
artifact was emitted, validated against the model (fp16-faithful), and probed — **but it is GONE**:
it lived only on a spot instance and was NOT pushed (too large). Code, results `.md`, and decisions
ARE pushed. See `RUNBOOK.md` for exact rebuild commands.

**FIRST STEP on resume — re-emit the artifact, then re-validate before trusting anything:**
1. `python -m stage0.emit_calibration --seq-len 8192 --n-seq 32 --out-dir outputs/calib`
   (~25 min on the GPU including the ~14GB model download; c4, seq_len 8192, 32 seqs → ~49GB).
2. Resume check (a) — contract: `python -m stage0.verify_calibration --dir outputs/calib`
   must print `CONTRACT OK across 32 layers` (RoPE applied, disjoint oracle/eval, attention peaked).
3. Resume check (b) — model faithfulness: `python -m stage0.validate_emit`
   must print `VERDICT: PASS` (recon-vs-true weights/outputs ~fp16, artifact-tie ≈ 0).
   Do NOT trust any Phase C/D Δ or oracle label until BOTH (a) and (b) pass on the re-emitted data.

**STALE-SPEC WARNING — do not follow these blindly:**
- The spec/hard-rules below reference a repo `--emit-rq` forward pass. **It does not exist** in this
  checkout; emit was built from scratch (`stage0/emit_calibration.py`). Ignore `--emit-rq`.
- The Phase A *synthetic* selector recommendation is **STALE/WRONG for real data**. On real Mistral
  heads (Phase C, `outputs/phase_c_probe.md`): **logdet is DEAD** — it won 0/39 heads; the
  volume-diversity story does NOT transfer. keydiff (also the Stage 3 baseline) was strongest
  (17/39), facility/coverage is sound and wins 15/39, kcenter 7/39.
- **Routability = YES** (38/39 heads separate beyond noise) — NOT the synthetic NO-GO.
- **Phase D selector menu is UNDECIDED.** Candidates: `keydiff` + `facility` + `kcenter`; drop
  `logdet`. Phase D also needs fuller averaging (more seqs, larger N) + per-head geometry `g_h`.
- Hardware is actually an **H200 (143GB)**, not the H100 named below. Model is
  `mistral-community/Mistral-7B-v0.2` (ungated; mistralai never published a base v0.2 repo).
- Env: a `.venv/` (torch 2.12+cu130, transformers 4.44.2, sentencepiece, protobuf) — gitignored,
  rebuild per `RUNBOOK.md`. Always run modules as `python -m stage0.X` from the repo root (a bare
  `python stage0/x.py` shadows stdlib `selectors` with `stage0/selectors.py`).

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
