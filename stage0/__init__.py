"""Stage 0 — synthetic instrument validation for per-head objective selection.

Synthetic only: no model, no dataset, no calibration artifact. Validates the two
greedy selectors, the geometry statistics, and the attention-output-error metric
against known ground truth before any real Mistral data is touched (CLAUDE.md
hard rule 3).
"""
