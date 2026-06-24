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
```
Sanity: `python -c "import torch,transformers,sentencepiece,protobuf; print(torch.cuda.is_available())"` → `True`.

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
python -m stage0.phase_c             # -> outputs/phase_c_probe.md
```

## 7. Resume the research
Point Claude Code at the **RESUME block at the top of `CLAUDE.md`** — it carries the current state,
the stale-spec warnings (logdet is dead on real heads; Phase D selector menu undecided), and the
next decisions. Phase D has NOT been started and needs an explicit GO.
