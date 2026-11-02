# Preregistration — TemporalGuard Phase 2B (confirmatory)

Version 1.0, drafted 2026-11-01. **Status: DRAFT, awaiting owner approval.**
It becomes binding when the owner records approval and the SHA256 of this file
in `phase2a_manifest.json` is committed by the owner. No confirmatory
prediction exists at the time of writing.

**Integrity statement.**
- No HMDB51 confirmatory clip has been passed through any model.
- The only inference in Phase 2A was a smoke test on 2 Phase 1 clips, which
  are exploratory and excluded.
- Protocol design used only:
  - the exploratory Phase 1 results;
  - the Phase 1 audit;
  - published literature;
  - synthetic data.

---

## 1. Study design

- **Type:** confirmatory evaluation of frozen and zero-parameter
  failure-detection monitors under cross-dataset shift (UCF101 → HMDB51) for a
  zero-shot video classifier.
- **Classifier:** `microsoft/xclip-base-patch32`, Hugging Face revision
  `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`, local files only. The SHA256 of
  every snapshot file is recorded in `phase2a_manifest.json`.
  - Prompts: `"a video of a person {label}"` over **101 UCF101 labels**,
    alphabetical order.
  - 8 frames per view, ≤96 decoded frames at width 224.
- **Frozen config:** `config/phase2b_config.json`, SHA256 embedded and
  checked at run time. The exporter refuses edited configs.
- **Frozen monitors:** Guardian V3 (`frozen_models_v2.json`, SHA256
  `e56fecec…1cd1`), used unchanged. No retraining, no recalibration.

## 2. Data

**Mapping:** TA3N 12 categories, verbatim (`DATASET_MAPPING.md`).

**Confirmatory sets** (`manifests/hmdb51_confirmatory_manifest.csv`):

| Set | Definition | n |
|---|---|---|
| **P (primary)** | HMDB51 split-1 train partition, 12 classes | 797 |
| **S (secondary)** | HMDB51 split-1 test partition, 6 classes not used in Phase 1 | 177 |

**Exclusions** (rules fixed; listed in `manifests/hmdb51_excluded.csv`):
- the 180 Phase 1 clips;
- any clip sharing a conservative film group with them (46 so far);
- any clip within perceptual-hash Hamming ≤10 of a Phase 1 clip;
- exact duplicates.

The final manifest is regenerated **once**, after the 6 remaining classes are
extracted (Phase 2B step 1), with the identical code and `--phash`. That is
the last change permitted before inference. Any additional exclusions are
logged with reasons. Counts may only go **down**.

**UCF101 reference data** (`manifests/ucf101_reference_manifest.csv`): the
Guardian development (202) and calibration (202) videos. These are the only
data allowed for any fitting, threshold setting or selection. The UCF101
evaluation split is not used.

**Labels:** correctness is computed only in the analysis step. Exported
prediction files contain no correctness fields.

## 3. Monitors (all label-free; frozen definitions in `baseline_metrics.py`)

Notation: ŷ = top-1 on `clean_adaptive`. pᵥ = softmax of view v.

| Name | Definition | Passes | Role |
|---|---|---|---|
| `msp` | max pᶜˡᵉᵃⁿ | 1 | **reference** |
| `neg_entropy`, `margin`, `energy` (logsumexp), `max_logit` | from clean logits | 1 | secondary baselines |
| `guardian_o`, `guardian_full` | frozen Guardian V3 on clean + severity-3 corruption probes | 2 / 5 | **primary** (G-F); secondary (G-O) |
| `tg0` | mean_{v∈T} pᵥ(ŷ), T = {clean_uniform, seg4_0..3} | 6 | **primary** |
| `tg_min`, `tg_agreement_then_msp`, `tg_neg_js`, `tg_sampling_pred_prob`, `tg_seg2` | see temporal_monitor_design.md | 2–6 | secondary |
| `corruption_tta_pred_prob` | mean over 4 Guardian probes of pᵥ(ŷ) | 5 | **ablation** for TG0 |
| `corruption_agreement_then_msp` | Phase 1 audit rule | 5 | secondary |

**Negative controls:**
- `constant` (AURC must equal the error rate);
- `random` (seeded uniform, seed 20273; within-class AUROC ≈ 0.5);
- `within_class_shuffled_msp` (MSP permuted within class, seed 20274);
- `class_oracle` (class accuracy; **diagnostic only, uses labels**;
  within-class AUROC = 0.5 by construction).

## 4. Endpoints

Primary analysis set **P**, correctness rule **mapped**.

### 4.1 Primary metric

**Pair-weighted within-class AUROC** = P(score of correct > score of incorrect
| same class), ties = ½. Pairs are pooled over classes, so classes with no
errors or no correct predictions contribute no pairs.

### 4.2 Primary hypotheses (family of 3, Holm-corrected)

| ID | Null | Statistic |
|---|---|---|
| H1 | within-AUROC(msp) = 0.5 | within-AUROC(msp) − 0.5 |
| H2 | within-AUROC(guardian_full) = within-AUROC(msp) | Δ |
| H3 | within-AUROC(tg0) = within-AUROC(msp) | Δ |

### 4.3 Secondary endpoints (reported in full; no confirmatory claims)

- **S1.** Pooled AURC, E-AURC, AUGRC, AURC−random, pooled AUROC, and
  risk@{50, 70, 80}% coverage for every monitor. Also the
  **class-composition share**: the fraction of ΔAURC (vs msp) reproduced by
  the within-class label-permutation null (10,000 permutations, seed 20272).
  The share is reported only when the pooled ΔAURC CI excludes 0.
- **S2.** tg0 vs corruption_tta_pred_prob (within-AUROC Δ).
- **S3.** Each logit baseline vs msp. Holm within this family of 4.
- **S4.** Each secondary TG variant vs msp.
- **S5.** H1–H3 repeated on set **S**.
- **S6.** H1–H3 repeated with **strict** correctness.
- **S7.** RQ3. For each disruption view:
  - top-1 accuracy (mapped) with a cluster CI;
  - MSP within-AUROC computed on that view's own logits;
  - the flip rate (clean-correct → disrupted-wrong);
  - the clean-view monitors' AUROC for predicting which clips flip.
- **S8.** RQ4:
  - passes per monitor;
  - median and p95 of per-clip summed `model_ms` and end-to-end ms (decode +
    transform + sampling + preprocess + model);
  - ΔAUROC_within per extra pass.
  - The first 3 clips of each run are excluded from timing (warm-up).
- **S9.** Calibration transfer. Thresholds come from UCF101 calibration
  only:
  - frozen thresholds for msp / G-O / G-F;
  - the frozen rule (70% target) for new monitors.

  Report HMDB51 coverage, selective risk, false-alarm rate and miss rate.
- **S10.** Macro within-class AUROC over classes with ≥3 correct and ≥3
  incorrect predictions. Per-class AUROC tables with counts.

## 5. Statistical analysis

- **Unit of resampling:** conservative class-agnostic film group (`film_group`).
- **Primary interval:** unstratified cluster bootstrap over film groups,
  percentile 95% CI. B = 10,000, seed 20270.
- **Co-primary test:** delete-one-film-group cluster jackknife, t with G−1
  df. Holm is applied to the 3 jackknife p-values.
- **Decision rule:** reject H₀ only if **both** hold:
  1. the 95% cluster-bootstrap CI excludes the null value;
  2. the Holm-adjusted jackknife p < 0.05.

  Reason: in the Phase 2A synthetic simulation the percentile bootstrap alone
  rejected a true null 9% of the time at nominal 5%.
- **Generalisation across activity classes:** stated only if the
  hierarchical (class → film group) bootstrap CI (seed 20271) also excludes
  the null. Otherwise results are stated as conditional on these 12 classes.
- **Equivalence:** a difference is "practically negligible" if its 90%
  cluster-bootstrap CI lies within ±0.03. The minimally important
  difference is 0.03 within-class AUROC.
- **Direction:** all tests are two-sided. A "monitor improves on MSP" claim
  requires a positive estimate.
- **Multiplicity:**
  - Holm over H1–H3;
  - Holm within S3;
  - all other secondary results are labelled exploratory, with unadjusted
    CIs.
- **No interim analyses.** The analysis script (`phase2b_analysis.py`, to be
  written and hashed in Phase 2B step 0, before any inference) runs once
  after all exports finish.

## 6. Class imbalance and degenerate classes

- The design is balanced at ≤70 per class (57–70 after exclusions).
- The primary metric weights each class by its number of correct × incorrect
  pairs. This is reported with per-class pair counts.
- Classes with 0 errors or 0 correct predictions are listed and contribute no
  within-class pairs. They are **not** removed from pooled metrics.
- The macro metric (S10) requires ≥3/≥3 per class.
- No class is dropped post hoc for any reason other than the validity rules
  in §8.

## 7. Power (planning estimate, synthetic)

Simulation design (`baseline_metrics.synthetic_benchmark`):
- 12 classes × 66 clips, 27 groups per class;
- group random effect; accuracy about 62% (about 300 errors).

Results (80 replications, B = 300):

| True Δ within-AUROC | Rejection rate (percentile CI) |
|---|---|
| ≈0.02 | 0.33 |
| ≈0.03 | 0.62 |
| ≈0.05 | 0.93 |

Mean CI width ≈0.06. At Δ = 0 the rejection rate was 0.09.

**Interpretation:** the study can detect Δ ≳ 0.04–0.05. It cannot reliably
detect the 0.03 MID, so equivalence statements matter. The real error count
and ICC are unknown, so this is a rough estimate. The sample size is fixed at
all eligible clips: no optional stopping, no adding data.

## 8. Validity and stopping criteria (checked before unblinding results)

Abort confirmatory claims, and report descriptively, if any of these hold:
1. The smoke equivalence check fails: top-1 mismatch, or |Δscore| > 1e-4 vs
   `edgehar.predict` on the Phase 1 smoke clips.
2. More than 2% of manifest clips fail to decode.
3. Any leakage assertion fails, or a perceptual near-duplicate of a Phase 1
   clip is found after extraction without being excluded.
4. Total mapped errors in P < 100, or fewer than 6 classes have ≥5 correct
   and ≥5 incorrect predictions.
5. The config or analysis-script hash differs from the frozen value.
6. Any frozen artifact hash changes.

These checks are computed and printed by the analysis script **before** any
endpoint.

## 9. Deviations

Every deviation is logged in `research_analysis/phase2b/DEVIATIONS.md` with:
- the date;
- what changed;
- why;
- whether results had been seen.

Results affected by a deviation are labelled.

## 10. What will be reported regardless of outcome

- All primary and secondary endpoints and all negative controls.
- Per-class tables.
- The full exclusion list.
- The smoke equivalence report.
- Every deviation.
- Null and negative results with equal prominence.
