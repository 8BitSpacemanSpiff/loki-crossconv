# KeyDiff comparison on Llama-3.1-8B-Instruct — what to run right now

This compares CrossCov R_q-weighted eviction (`evict`) against KeyDiff (`keydiff`) and
the key-norm control (`keynorm`) at a **matched cache budget**, with identical sink/recent
protection. KeyDiff and key-norm need **no calibration** — they run on the live keys.
Only `evict` needs the one-time CrossCov basis + R_q for Llama.

Model facts: Llama-3.1-8B-Instruct is 32 layers, 32 query heads, 8 KV heads (GQA),
head_dim 128. Use `--model-type llama --pool-gqa`.

Compatibility note: this repo pins `transformers==4.40.2`, which cannot parse the
new Llama-3.1 `rope_scaling: {"rope_type": "llama3", ...}` config. For the 8K-only
WikiText/C4 runs below, patch the local model config once:
```
python tools/llama31_8k_compat_config.py /home/models/Llama-3.1-8B-Instruct
```
This backs up the original config and removes the unsupported extended-RoPE entry.
Restore with `--restore` before any >8K experiment.

Dataset note: `dataset=c4` first checks `LOKI_C4_PATH` and the old lab path. If no
local dataset exists, it streams C4 validation from Hugging Face. Set
`LOKI_C4_STREAM_SAMPLES` if you want more or fewer calibration/evaluation samples.

## Tier 0 — runs immediately, no calibration (KeyDiff vs key-norm)
You can sanity the baselines this second:
```
python evaluate_tasks.py --model-id meta-llama/Llama-3.1-8B-Instruct --model-type llama \
  --sequence-length 8192 --dataset wikitext-test \
  --use-pca-topk --use-crosscov --crosscov-mode keydiff \
  --keep-tokens 2048 --sink-tokens 16 --recent-window 64 \
  --quiet-diagnostics --log-mass-recall
```
Swap `--crosscov-mode keynorm` for the control. This already tells you how much true
attention mass KeyDiff retains at a 2048-token budget. But it does NOT include your
method yet — for that you need Tier 1.

## Tier 1 — the real comparison (one calibration, then the driver)
**Step 1 — calibrate Llama (once).** Same flow you used for Mistral, pointed at Llama:
1. Save pre-RoPE Q/K tensors from C4 (the `--save-tensors` path, `--rotary-type prerotary`).
2. Build the GQA-pooled basis + pooled R_q:
   ```
   python pca_analysis/crosscov.py 32 <tensor_root> <out_dir> --pool-gqa --emit-rq
   ```
   This writes the CrossCov-U basis and `<out_dir>/key/rq_gram/...` for all 32 layers.

**Step 2 — run the four-way comparison at a matched budget:**
```
python run_evict_comparison.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct --model-type llama \
  --dataset wikitext-test --sequence-length 8192 \
  --top-r 32 --keep-tokens 2048 \
  --rotary-type prerotary --transform-dataset c4 \
  --sink-tokens 16 --recent-window 64 \
  --modes evict,keynorm,keydiff \
  --out evict_llama31_2048.json
```
It prints a table of **mass_kept** and **ppl** per method at the shared budget.

## The metric that is the go/no-go
`mass_kept` = fraction of true attention mass captured by the retained set. It is the
thing eviction is supposed to maximize, it is query-faithful, and it runs in a single
forward (cheap). Read it first:
- `evict` > `keydiff` and `evict` > `keynorm`  → query-weighting helps; commit GPU-hours to LongBench.
- `evict` ≈ `keynorm`  → the win is just magnitude, not the query distribution. Honest negative; KeyDiff's diversity view may be the better object.
- `evict` < `keydiff`  → don't build the eviction paper; redirect to selection/compression.

## Budget sweep
Repeat with `--keep-tokens` in {4096, 2048, 1024, 512} at `--sequence-length 8192`. KeyDiff
is near-baseline at mild budgets; methods separate at tight budgets, so the 512–1024 points
are where the comparison is decided.

## To match KeyDiff's published LongBench table (follow-up, not "right now")
KeyDiff reports LongBench accuracy at an 8K absolute budget on Llama-3.1-8B. To reproduce
that head-to-head you need LongBench wired into the harness (check whether `--lm-harness-eval`
covers it; LongBench may need adding). Then run each `--crosscov-mode` with `--keep-tokens 8192`
on LongBench. The mass_kept result above is the cheap predictor of whether that run is worth it.

## Fairness checklist (already enforced by the shared code path)
- Identical sink/recent protection across all methods (`_static_evict_mask`).
- Identical budget (`--keep-tokens`, the KeyDiff protocol).
- Both KeyDiff and `evict` are attention-free (R_q is offline) → both FlashAttention-compatible;
  state this so KeyDiff's attention-free property isn't a differentiator.
- R_q calibrated on C4, disjoint from the eval text.
- Static-KeyDiff (anchor over the full sequence) matches the single-forward harness; the
  streaming-anchor version is a separate efficiency axis, not an accuracy difference.
- Structural edge worth measuring separately: `evict` scores from r=32 latent dims;
  KeyDiff needs full d=128 keys for cosine — at iso-accuracy your scoring path touches less memory.
