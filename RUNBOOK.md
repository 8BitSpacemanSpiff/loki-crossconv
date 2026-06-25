# RUNBOOK — rebuild & resume on a fresh instance

This repo's per-head KV-eviction work (`stage0/` package) is fully tracked, but the **49GB
calibration artifact is NOT** (gitignored, rebuilt each session). A cold instance with no memory of
prior sessions can resume from a fresh clone using the steps below.

Hardware assumed: 1 GPU (H200/H100 class, ~80GB+), CPU with many cores, ~100GB free disk.

## 1. Clone + branch
```bash
git clone https://github.com/8BitSpacemanSpiff/loki-crossconv.git
cd loki-crossconv
git checkout evict-perhead
```

## 2. Environment / deps
Create a venv and install the stack (CUDA torch already matches this box's driver; pin transformers
to a Mistral-compatible version). The emit tokenizer needs sentencepiece + protobuf.
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu130   # match the box's CUDA
pip install "transformers==4.44.2" "accelerate>=0.30" "datasets>=2.17" \
            "safetensors>=0.4.3" "huggingface-hub>=0.23" sentencepiece protobuf
pip install "numpy<2" matplotlib                                       # see caveats below
```
Sanity: `python -c "import torch,transformers,sentencepiece,protobuf; print(torch.cuda.is_available())"` → `True`.

> **Caveats seen on a fresh A100 box (2026-06-25):** the prebuilt torch wheel needed **numpy<2**
> (numpy 2.x crashed the import) and **sentencepiece** was missing. matplotlib is needed for the
> Phase D plots. If the box's torch is **< 2.1.1**, transformers refuses `attn_implementation="sdpa"`;
> `emit_calibration.py` / `validate_emit.py` therefore use **`eager`** — inert for what we capture
> (K/Q/V come from pre-attention hooks; the artifact-tie was 0.00e+00).

> ALWAYS run modules as `python -m stage0.X` from the repo root. Running `python stage0/x.py`
> directly puts `stage0/` on `sys.path[0]`, so `stage0/selectors.py` shadows the stdlib `selectors`
> module and breaks huggingface_hub/urllib imports.

## 3. Stage 0 — synthetic instruments (no GPU, seconds; optional re-confirm)
```bash
python -m stage0.validate            # 3 A.1 criteria, all PASS across 8 seeds
python -m stage0.phase_a             # bidirectional metric (Δ) check
python -m stage0.phase_a_taxonomy    # selector taxonomy (NOTE: synthetic ranking is STALE vs real)
```

## 4. Re-emit the calibration artifact (GPU, ~25 min incl. model download)
Model `mistral-community/Mistral-7B-v0.2` (ungated) is fetched automatically on first load.
```bash
python -m stage0.emit_calibration --seq-len 8192 --n-seq 32 --out-dir outputs/calib
```
Writes `outputs/calib/layer_00..31.pt` (~49GB) + `meta.json`. (Smoke first if unsure:
`python -m stage0.emit_calibration --smoke`.)

## 5. Resume checks — MUST pass before trusting any Δ
```bash
python -m stage0.verify_calibration --dir outputs/calib   # (a) expect: CONTRACT OK across 32 layers
python -m stage0.validate_emit                            # (b) expect: VERDICT: PASS
```
(a) confirms RoPE applied, disjoint oracle/eval, GQA group 4, attention peaked.
(b) reconstructs attention from saved tensors and matches the live model to fp16 noise.

## 6. Phase C — real-head routable-margin probe (CPU, ~90s)
```bash
python -m stage0.phase_c             # -> outputs/phase_c_probe.md  (SUPERSEDED by Phase D, see below)
```

## 6b. Resume checks (CPU, ~2 min) — logdet conditioning + kcenter/facility overlap
```bash
python -m stage0.resume_checks       # -> outputs/resume_checks.md
```

## 6c. Phase D — STEP 0 / 0b / STEP 1+2 (the per-head routing experiment)
```bash
# STEP 0  (CPU, ~1 min): is the rank-1 K_pre a shared sink offset? -> YES, centered rank ~9-27
python -m stage0.step0_centered_rank          # -> outputs/phase_d_step0_centered_rank.md
# STEP 0b (CPU, ~1 min): does the sink offset distort selection? -> only inner-product selectors
python -m stage0.step0b_centering_ablation    # -> outputs/phase_d_step0b_centering_ablation.md
# STEP 1+2 (GPU, ~2 h on A100): full 256-head measurement + geometry g_h. Per-layer checkpoint;
#   re-running resumes from outputs/phase_d_raw.pt. facility is the cost (~3.7s/cloud, exact).
python -m stage0.phase_d_step1 --n-seq 8      # -> outputs/phase_d_raw.pt  (452K, IS pushed)
python -m stage0.phase_d_report               # -> measurement.md, geometry.md, perhead.csv, 3 PNGs
# SINK ABLATION (GPU, ~1 h): protect pos 0-3 for all selectors @3%/@25%. DONE.
python -m stage0.phase_d_sink_ablation --n-seq 8   # -> phase_d_sink_ablation.md, phase_d_sink_raw.pt
```
Smoke first if unsure: `python -m stage0.phase_d_step1 --smoke` (asserts batched==per-cloud selectors).

## 6d. TOMORROW (in order) — massive activations + the content-excluded Δ fork
The per-head ROUTING experiment is DEAD (Phase D + sink ablation). The live axis is
distinctiveness vs coverage, and it hinges on ONE check. Sequence:
1. **Re-emit WITH massive-activation capture folded in** (~25 min H100/A100). Implement per
   `outputs/phase_d_emit_addition_massive_activations.md` (residual top-8 + per-dim max-abs), then
   `emit_calibration` → `verify_calibration` → `validate_emit` (must PASS). **Artifact is NOT saved
   (spot) — re-emit first.**
2. **Content-excluded Δ (CPU)** — the decisive fork: `keydiff_unc` vs routed `min{facility,kcenter}`
   (all sink-protected), Δ renormalized over NON-sink positions (drop 0-3 from softmax denominator
   AND output sum), @3% and @25%, broken out by layer band **L0-1 / L2-14 / L15+**. Holds → reframe
   toward distinctiveness is earned; drops to parity → gap was sink bookkeeping, fold eviction.
3. **Three-way mechanism check** (massive-activation dims ↔ sink positions ↔ retaining selector).

## 7. Resume the research
Point Claude Code at the **RESUME block at the top of `CLAUDE.md`** — current state, the centering
story, the settled roles, the Phase D + sink-ablation verdicts, and the CRITICAL OPEN ITEM
(content-excluded Δ). Routing is dead; do step 6d FIRST. Rule: every selector comparison MUST
sink-protect all selectors or facility is handicapped.
