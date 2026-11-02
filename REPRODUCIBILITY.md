# Reproducibility Guide

Three levels of reproduction are possible. This guide says which apply to
each result.

| Level | Needs | Status in this public release |
|---|---|---|
| **R0: inspect** | this repository only | ✅ All aggregate results, figures, protocols, reports |
| **R1: re-test code** | this repository plus Python | ✅ Partly: some unit tests run on synthetic data (§2) |
| **R2: re-run analyses from model outputs** | per-clip model outputs (logits for 19 views × 101 classes per clip) | ❌ **Not published**: per-clip records are withheld (§3) |
| **R3: full re-run from video** | UCF101 + HMDB51 videos, X-CLIP weights, and the pipeline modules | ❌ **Not possible from this reduced release.** The prediction exporter and its pipeline dependencies are omitted while their redistribution rights are verified. The selection rules and protocols are documented (§4). |

## 1. Environment used for the reported results

- **Python and packages:** Python 3.12.16, torch 2.15.0, transformers
  4.57.6, numpy 1.26.4, scipy 1.17.1, matplotlib 3.11.2, Pillow 11.3.0,
  opencv-python 4.10.0.84, pytest 9.2.0.
- **Hardware:** macOS on Apple silicon; inference on MPS.
- **Not yet tested:** `requirements.txt` gives the pinned versions, but has
  not been tested in a fresh environment. The original environment also had
  `opencv-contrib-python` 4.11 installed alongside `opencv-python`.

## 2. Tests

Run each file separately:

```bash
python -m pytest -q -p no:cacheprovider <file>
```

| Measure | Result in a simulated fresh public checkout (offline, empty model cache) |
|---|---|
| Test files / tests | 1 / 27 |
| **PASS** | **21** |
| **BLOCKED** (missing unpublished input; not passed) | **6** |
| FAILED for another reason | 0 |

Per-file details and every missing input: [`TEST_STATUS.md`](TEST_STATUS.md).

## 3. What is withheld, and why

| Withheld | Reason | Effect |
|---|---|---|
| Video files, dataset archives | Third-party content; the redistribution terms are the data providers' | R3 requires obtaining the datasets |
| Per-clip manifests, per-clip predictions, per-clip score tables (`phase2b/predictions/`, `per_clip_table.csv`, Phase 1 `per_clip_scores.csv`, Guardian per-video CSVs) | They contain dataset file names, some of which include personal names from source videos; redistribution terms are unverified | R2 is not possible from this repository; reported aggregates must be taken as published |
| Phase 3 surrogate, validation harness, per-clip validation records, freeze files | The robustness evaluation is unapproved and incomplete; freeze files embed clip paths | Phase 3 numbers are published as an aggregate extract (`research_analysis/phase3/phase3_validation_summary.json`) with the frozen-record SHA256 values for later verification |
| Run logs, local manifests with machine paths | Machine-specific | None on the results |

The phase reports in this repository are copies of the frozen originals,
so they still cite some of the withheld files.

## 3a. Public copies that differ from the frozen originals

Only these files were changed, and only in the public copies. The frozen
originals are unchanged in the private repository, and the SHA256 of each
original is recorded privately.

| File | Change | Reason |
|---|---|---|
| `research_analysis/phase3/PHASE3_PARITY_VALIDATION.md` | One UCF101 clip ID replaced by `[clip id redacted]` | Clip-level identifier |
| `research_analysis/phase3/feasibility/p3_diagnostics/P3_DIAGNOSTIC_REPORT.md` | One UCF101 clip ID replaced by `[clip id redacted]` | Clip-level identifier |
| `research_analysis/phase2b/figures_final/*.pdf`, `results/guardian_v3_official/risk_coverage/*.pdf` | PDF `CreationDate`: time of day and timezone removed (date kept; same byte length) | Personal metadata |

No number, table value or conclusion was changed.

## 4. Re-deriving the data selection (R3)

- **Phase 2B categories and mapping:**
  `research_analysis/phase2a/DATASET_MAPPING.md`, the 12 TA3N
  UCF↔HMDB categories.
- **Phase 2B sets:**
  - Set P is the HMDB51 split-1 *train* partition of those categories.
  - Set S is 6 further categories, split-1 *test*.
  - Exclusion rules are in `PREREGISTRATION.md`.
- **Phase 1:** all split-1 test clips of fencing, punch, pushup, pullup,
  ride_bike and ride_horse (180 clips).
- **Model outputs:** produced by the prediction exporter, which is
  omitted from this reduced release. Its configuration is included
  (`research_analysis/phase2a/config/phase2b_config.json`).
- **Statistics:** `research_analysis/phase2b/phase2b_analysis.py`, with
  seeds recorded in `STATISTICAL_RESULTS.json` → `config`.

## 5. Integrity references (frozen records held privately; SHA256)

| Record | SHA256 |
|---|---|
| Guardian v3 frozen models (`results/guardian_v3_official/frozen_models_v2.json`, **included**) | `e56fececf1f980a2eafbc9fc882ac4157b3829802a8c492b9a31994db1022cd1` |
| Phase 2A preregistration v1.0 (`research_analysis/phase2a/PREREGISTRATION.md`, **included**, byte-identical) | see `phase3_validation_summary.json` → `included_file_sha256` |
| Phase 3 parity criteria freeze (`CRITERIA_FREEZE.json`, withheld) | see `phase3_validation_summary.json` → `frozen_record_sha256` |
| P3-v2 freeze (`P3V2_FREEZE.json`, withheld) | see `phase3_validation_summary.json` → `frozen_record_sha256` |
