# Integration instructions (hand to the integrating LLM)

You are integrating two new runtime modes — **KV compression** and **KV eviction** —
into the `loki-crossconv` repo (a Loki fork with a CrossCov-U sparse-attention path).
Both are sub-modes of the existing `--use-crosscov` path. Do **not** change the
offline basis construction or the existing `select` behaviour.

There are two ways to do this. **Prefer Path A.** Use Path B only if the patch will
not apply cleanly (e.g. the target tree has diverged).

---

## Path A — apply the patch (preferred)

A git patch is provided: `crosscov_ext.patch`. From the repo root:

```bash
git checkout -b crosscov-ext
git am < crosscov_ext.patch          # preserves the commit
# if git am fails:
git apply --reject --whitespace=fix crosscov_ext.patch   # then resolve any .rej
```

Then jump to **Verification** below. Do not make any other edits.

---

## Path B — apply the changes manually

Make exactly these edits. Each is additive; preserve all existing code paths.

### B1. New file `methods/pca_topk/crosscov_ext.py`
Copy it verbatim from the patch (or from the provided file). It defines three
functions: `mask_attn_crosscov_compress`, `mask_attn_crosscov_evict`, and
`get_rq_gram`. It imports `methods` and `from .utils import PCA_DATA_PATH`, matching
the sibling `crosscov_utils.py`. Do not rewrite its logic.

### B2. `methods/pca_topk/modify_mistral.py` and `methods/pca_topk/modify_llama.py`
Identical three edits in **both** files:

1. **Imports.** After the line
   `from .crosscov_utils import mask_attn_crosscov, get_crosscov_components`
   add:
   ```python
   from .crosscov_ext import (
       mask_attn_crosscov_compress,
       mask_attn_crosscov_evict,
       get_rq_gram,
   )
   ```

2. **Mode read + lazy gram load.** Immediately after
   `use_crosscov = getattr(args, "use_crosscov", False)` add
   `crosscov_mode = getattr(args, "crosscov_mode", "select")`.
   Inside `if use_crosscov:`, after the `get_crosscov_components(...)` block that
   sets `self.cc_key_r`, add:
   ```python
   if crosscov_mode == "evict" and not hasattr(self, "cc_rq_gram"):
       self.cc_rq_gram = get_rq_gram(
           args, self.layer_idx, self.head_dim, args.top_r,
           self.num_key_value_groups, repeat_kv, self.cc_key_r)
   ```

3. **Dispatch.** Find the single call:
   ```python
   attn_weights, alpha = mask_attn_crosscov(
       args, self.layer_idx, attn_weights, attention_mask, query_states, key_states,
       self.cc_key_r, self.cc_query_r, args.top_r, topk)
   ```
   Replace it with an if/elif on `crosscov_mode` that keeps the above for `"select"`,
   calls `mask_attn_crosscov_compress(...)` (same args) for `"compress"`, and calls
   `mask_attn_crosscov_evict(args, self.layer_idx, attn_weights, attention_mask,
   query_states, key_states, self.cc_key_r, self.cc_rq_gram, args.top_r,
   getattr(args,"evict_ratio",0.5), getattr(args,"sink_tokens",16),
   getattr(args,"recent_window",64))` for `"evict"`, else `raise ValueError`.
   (Copy the exact block from the patch to avoid signature drift.)

### B3. `pca_analysis/crosscov.py` — offline Rq emission
Add three things (copy from the patch):
- functions `pooled_gqa_rq(Q_all, kv_head, group)` and `save_rq(out_side_dir, layer_id, rq)`
  (place them just above `save_layer`);
- parse `emit_rq = "--emit-rq" in sys.argv[4:]` next to the existing flag parsing;
- after the two `save_layer(...)` calls in the per-layer loop, an `if emit_rq:` block
  that builds one pooled `Rq` per KV head and calls
  `save_rq(f"{output_dir}/key", layer_id, rq)`.
This writes `<out>/key/rq_gram/rq_gram_layer_{l}.pt`. It changes nothing when the flag is absent.

### B4. `evaluate_tasks.py` — CLI flags
After the existing `--use-crosscov` argument, add:
```python
parser.add_argument("--crosscov-mode", type=str, default="select",
                    choices=["select", "compress", "evict"])
parser.add_argument("--evict-ratio", type=float, default=0.5)
parser.add_argument("--sink-tokens", type=int, default=16)
parser.add_argument("--recent-window", type=int, default=64)
```

### B5. New files (copy verbatim, no integration needed)
- `methods/pca_topk/crosscov_ext.py` (B1)
- `tests/test_crosscov_ext.py` (numpy self-test)
- `run_three_variants.py` (driver)
- `CROSSCOV_EXT.md` (docs)

---

## Verification (run after either path)

```bash
# 1. everything compiles
python -m py_compile methods/pca_topk/crosscov_ext.py \
  methods/pca_topk/modify_mistral.py methods/pca_topk/modify_llama.py \
  pca_analysis/crosscov.py evaluate_tasks.py run_three_variants.py

# 2. the math behind both modes (numpy only, no torch/GPU)
python tests/test_crosscov_ext.py          # must print "ALL CHECKS PASSED"

# 3. the driver builds the right commands without running them
python run_three_variants.py --model-id X --model-type mistral \
  --top-r 8 --keep 0.125 --dry-run
```

All three must succeed before running anything on a GPU.

---

## Critical gotchas — do not get these wrong

1. **Both flags are required** for any CrossCov run: `--use-pca-topk` selects the
   `pca_topk` modifier, and `--use-crosscov` switches to the CrossCov branch inside it.
   Neither alone is enough. The driver already passes both.
2. **Eviction needs an offline step.** Run the calibration once with `--emit-rq` to
   produce the `rq_gram` files. Without them, `evict` mode silently falls back to an
   identity gram (pure key-norm eviction) and prints a warning — the numbers will not
   reflect the query-weighted method.
3. **Do not "fix" the compress path to use exact keys.** It scores selected tokens
   from *reconstructed* rank-r keys on purpose; that is the whole point (it measures
   the real accuracy cost of the compressed cache). Using exact keys would silently
   turn it back into `select`.
4. **Scope of these modes:** they are faithful for the single-forward PPL/LongBench
   accuracy harness. They are **not** an incremental decode cache — `compress` does not
   realize a memory saving in the forward (it logs the ratio), and `evict` is a static
   whole-sequence mask, not evict-on-write. Do not add throughput/latency claims from
   these paths; that needs separate cache-storage engineering.
5. **post-RoPE vs pre-RoPE.** Both modes operate on `key_states` as the modify files
   present them, which is post-RoPE. A pre-RoPE compression variant (to match the SALS
   overlap-score argument) requires capturing pre-RoPE keys before the RoPE line — that
   is a deliberate future step, not part of this change. Do not silently introduce it.
6. **Don't touch** `mask_attn_crosscov`, `get_crosscov_components`, or anything under
   `pca_analysis/` other than the additive `--emit-rq` block.
```
