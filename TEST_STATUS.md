# Test Status (release candidate, 2026-11-02)

- **Environment:** a simulated fresh public checkout of this repository.
- **Model access:** offline, with an empty Hugging Face cache.
- **Command:** each file run separately with
  `python -m pytest -q -p no:cacheprovider <file>`.
- **Status meanings:** **PASS** means the test passed. **BLOCKED** means the
  test failed or errored because an input file is not in this repository.
  Blocked tests have **not** passed.

| Test file | Tests | PASS | BLOCKED |
|---|---|---|---|
| `research_analysis/cross_dataset_hmdb51/phase1_audit/test_statistical_audit.py` | 27 | 21 | 6 |
| **Total** | **27** | **21** | **6** |

## Missing inputs

| Missing input | Blocked tests | What it is |
|---|---|---|
| `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` | 4 | Phase 1 per-clip predictions (withheld: clip-level records) |
| `research_analysis/cross_dataset_hmdb51/hmdb51_mapped_test_manifest.csv` | 1 | Phase 1 per-clip manifest (withheld: clip-level records) |
| `results/guardian_v3_official/evaluation_balanced.csv` | 1 | Guardian v3 per-video evaluation scores (withheld: clip-level records) |

Blocked tests whose cause is not a missing file: **0**.

## Every blocked test

| Test | Outcome | Missing input |
|---|---|---|
| `test_statistical_audit.py::test_manifest_official_split` | fail | `research_analysis/cross_dataset_hmdb51/hmdb51_mapped_test_manifest.csv` |
| `test_statistical_audit.py::test_group_structure` | error | `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` |
| `test_statistical_audit.py::test_cluster_counts_constant_within_group` | error | `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` |
| `test_statistical_audit.py::test_phase1_point_estimates` | error | `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` |
| `test_statistical_audit.py::test_phase1_bootstrap_reproduced_exactly` | error | `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` |
| `test_statistical_audit.py::test_end_to_end_and_no_overwrite` | fail | `results/guardian_v3_official/evaluation_balanced.csv` |

## Scope of this reduced release

- **Omitted test suites:** `tests/test_edgehar.py`, `tests/test_guardian.py`,
  `test_phase2a.py`, `test_phase2b_analysis.py` and
  `test_hmdb51_validation_v2.py`, together with the modules they test
  or import. They are omitted while redistribution rights are verified.
- **The full release** has 116 tests.

