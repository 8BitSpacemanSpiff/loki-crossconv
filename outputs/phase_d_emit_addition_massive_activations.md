# Phase D — emit addition: massive-activation capture (TOMORROW's first task)

> Status: **SPEC, not yet implemented.** Created at end of 2026-06-25 wind-up because the file
> was referenced but did not exist. The calibration artifact is NOT saved (spot instance) — the
> emit must be re-run anyway, so fold this capture in BEFORE re-emitting.

## Why
The entire Phase D / sink-ablation story is about **attention sinks** on positions 0-3 (mean
`sink_mass ~0.54` of eval-query attention; KeyDiff parks ~0.68 of its retained mass there). Attention
sinks are produced by **massive activations** in the residual stream (a few hidden dims with huge
magnitude at a few positions, esp. BOS). To characterize/handle them we need to capture them at emit
time, not infer them after the fact.

## What to capture (per layer, fold into `stage0/emit_calibration.py`)
On the **residual-stream hidden state** `h` of each decoder layer (shape `(seq, d_model=4096)`):
1. **per-dim max-abs**: `h.abs().amax(dim=seq)` → `(d_model,)` — which hidden dims blow up, over all
   positions. Identifies the massive-activation dimensions.
2. **residual top-8**: the 8 largest-|value| `(position, dim, value)` entries of `h` — the actual
   massive activations (expect them concentrated at positions 0-3 / specific dims).

Capture mechanism mirrors the existing emit hooks (monkeypatch `apply_rotary_pos_emb` call-counter +
`v_proj` forward hooks): add a forward hook on each `model.model.layers[i]` (decoder layer) output to
grab the residual stream, reduce on-GPU to the two summaries above, store per (layer, sequence).
Keep fp16; tiny vs the 49GB K/Q/V. Extend `verify_calibration` with a sanity line (sinks/massive
dims present and stable across sequences).

## Then, in order (do NOT skip the re-validate)
1. Re-emit WITH this capture → `verify_calibration` (`CONTRACT OK`) + `validate_emit` (`VERDICT: PASS`).
2. **CONTENT-EXCLUDED Δ (CPU) — the decisive fork.** Re-run `keydiff_unc` vs routed
   `min{facility,kcenter}`, **all selectors sink-protected (always-keep pos 0-3)**, but score on Δ
   with positions 0-3 **excluded from BOTH the softmax denominator AND the output sum** (renormalize
   over non-sink keys only). Report @3% and @25%, **broken out by layer band L0-1 / L2-14 / L15+**.
   - gap HOLDS → "reframe toward distinctiveness" is earned (real, not sink bookkeeping).
   - gap DROPS to parity → the 75% was sink bookkeeping → fold eviction back in; distinctiveness dies.
   Δ stays on real uncentered K_post/V.
3. **Three-way mechanism check** tying massive-activation dims ↔ sink positions ↔ which selector
   retains them, using the captured summaries (per-dim max-abs / residual top-8).

## Carry-over rules
- Every selector comparison MUST sink-protect all selectors, or facility is handicapped.
- Per-head routing (coverage axis) is DEAD; the only live axis is distinctiveness vs coverage,
  pending step 2 above.
- Second open item (after the fork): the @25% sink-protection norm reversal (facility 165→93,
  kcenter 91→163) — unexplained budget × coverage-normalization interaction.
