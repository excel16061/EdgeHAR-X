# Phase 2B Deviations and Clarifications Log

Rule: every entry records the date, what changed, why, and whether any
confirmatory result had been seen.

## Pre-inference clarifications (2026-11-01; no confirmatory clip had been processed)

These operationalize under-specified points of PREREGISTRATION.md v1.0. They
do not change hypotheses, monitors, metrics, data or decision rules.

| # | Item | Clarification | Results seen? |
|---|---|---|---|
| C1 | S3 family decision rule | Same rule as the primary family: 95% cluster-bootstrap CI excludes 0 **and** Holm-adjusted (within S3) jackknife p < 0.05. The same rule is applied descriptively to S2/S4 and exploratory comparisons. | No |
| C2 | Class-composition share (S1) | share = mean ΔAURC under within-class label permutation ÷ observed ΔAURC. Reported only if the 95% cluster-bootstrap CI of the observed ΔAURC excludes 0. | No |
| C3 | S7 flip AUROC | Among clips correct on `clean_adaptive`: positive = still correct on the disrupted view, negative = flipped to wrong. The score is the clean-view monitor. | No |
| C4 | S8 warm-up | "First 3 clips of each run" = the first 3 clips of P in manifest order. End-to-end ms = shared decode + Σ (transform + sampling + preprocess + model) over the monitor's views. | No |
| C5 | S9 tied boundary | If a monitor's UCF101-calibration scores tie at the 70% boundary, its threshold is N/A. | No |
| C6 | Probabilities | Stored as float64 softmax of the exported float32 logits in `results/predictions_consolidated.npz`. The exporter itself stores logits only, as frozen in Phase 2A. | No |
| C7 | Degenerate jackknife | `baseline_metrics.jackknife_t` returns p = 0 when SE = 0. The analysis wrapper sets p = 1 when SE = 0 and estimate = null (p = 0 otherwise). The frozen Phase 2A file is unchanged. | No |
| C8 | Non-estimable endpoints | If a subset has no within-class (correct, incorrect) pairs, the endpoint is reported as `not_estimable` (no CI, not rejected) instead of crashing. It is excluded from that family's Holm set. | No |
| C9 | Memory | Not recorded by the frozen exporter. Measured separately after the exports by a small profiling run on the 2 Phase 1 smoke clips (exploratory; not confirmatory data). | No |
| D1 | Extraction command path (PHASE2B_EXECUTION_PLAN Step 1) | The first `unar` call extracted the inner class archives into `datasets/HMDB51/extracted/_archives/hmdb51_org/`, not `_archives/`, so the documented loop found nothing. It was re-run with `_archives/hmdb51_org/<class>.rar`, still `-s` (never overwrite). All 1,132 split-file clips of the 6 classes were extracted, 0 missing. Procedural only; no effect on data or analysis. | No |

## Deviations during/after inference

| # | Item | Change | Results seen? |
|---|---|---|---|
| D2 | Figure 2 subtitle (`make_figures.py`, not a frozen artifact) | The first rendering (`figures/`) said the pooled-within gap "comes from class difficulty". The results show only about 54-56% (Guardian-Full, TG0) and about 0% (max-logit) of the pooled ΔAURC is class composition, so that wording overstated the data. Re-rendered to `figures_final/` with a definitional subtitle. `figures/` is kept (no deletion) and superseded. No statistic changed. | Yes (wording only) |
