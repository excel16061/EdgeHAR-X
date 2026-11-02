# HMDB51 Cross-Dataset Validation V2 (Phase 1)

Date: 2026-11-01. Status: complete, awaiting owner review. Nothing committed.

> **Scope:** 180 HMDB51 clips from **6 classes** mapped to UCF101 labels
> (official split-1 test partition, 50 source groups). This is **not** a
> 51-class HMDB51 evaluation and should not be described as one.

## 1. Protocol (verified against repository code)

| Item | Value | Verified by |
|---|---|---|
| Backbone | `microsoft/xclip-base-patch32`, zero-shot | `model` column of all 900 rows |
| Candidate labels | 101 UCF101 classes (alphabetical `classInd.txt`) | `hmdb51_full_eval.py` |
| Sampling | adaptive, 8 frames, ≤96 decoded frames, width 224 | `strategy` column; `hmdb51_full_eval.py` |
| Class mapping | fencing→Fencing, punch→Punch, pushup→PushUps, pullup→PullUps, ride_bike→Biking, ride_horse→HorseRiding | asserted in script |
| Clips | 180 = **all** split-1 test clips (label `2`) of the 6 classes, 30/class | `verify_manifest()` against `testTrainMulti_7030_splits/*_test_split1.txt` |
| Source groups | 50 (filename prefix before `_<class>_`), 6–10 per class, sizes 1–15 | `source_group()` |
| Conditions | clean + blur/dark/occlusion/frame_drop at severity 3 (900 rows, 0 errors) | `load_clips()` |
| Frozen models | `results/guardian_v3_official/frozen_models_v2.json`, SHA256 `e56fecec…1cd1` ✔ | hashed before and after run |
| Thresholds | From the UCF101 calibration split (target coverage 0.70); `evaluation_data_used: false` | frozen JSON |
| Guardian features | Same definitions as `guardian_v3_official_train.load_features` | reproduced v1 AURC to 1e-6 (tests) |

**No HMDB51 data was used to train, calibrate, or choose thresholds.** The
matched-coverage metrics (risk at 50/70/80%) rank clips within HMDB51. They
evaluate ranking quality and are not deployable thresholds.

## 2. Discrepancies and observations found during the audit

1. **No HMDB51 entry in `research/RESEARCH_LOG.md`.** The protocol was
   reconstructed from code and `CLAUDE.md`. Both agree.
2. **Pilot overlap.** The 30 pilot clips (`hmdb51_pilot_predictions.csv`) are
   a subset of the 180 evaluation clips. Their predicted labels match exactly,
   which suggests inference is deterministic. The pilot was used to check the
   pipeline before the full run, so the evaluation set is not perfectly
   "untouched". There is no evidence the pilot was used to tune anything.
3. **Cosmetic issues in `hmdb51_full_eval.py`.** It prints "Pilot videos:"
   and comments about "an earlier pilot" even though it is the full run. The
   v1 fixed-threshold results were printed to stdout only and never saved.
   V2 saves them.
4. **The v1 bootstraps differ in settings.** The i.i.d. bootstrap used seed
   42 with B=5000. The cluster bootstrap used seed 42 with B=5000. The
   UCF101 final evaluation used seed 2027 with B=10000. V2 uses seed 2027,
   B=10000, and a class-stratified cluster bootstrap. Both v1 scripts
   reproduce exactly (re-run on 2026-11-01).
5. **Label-mapping ambiguity.** 19/30 HMDB `punch` clips are predicted as
   UCF101 `BoxingPunchingBag`. 14/30 `fencing` clips are predicted as
   `TaiChi`. Part of the measured "error" is caused by the one-to-one
   mapping, not by recognition failure.
6. **Classes are balanced** at 30 clips each. Pooled accuracy therefore
   equals class-balanced accuracy (67.78%), and class reweighting has no
   effect. The real class issue is **between-class confounding** (§4.3).

## 3. Statistical methods

- **Unit of resampling:** source group, resampled with replacement *within
  class* (stratified cluster bootstrap). Percentile 95% CIs, B=10,000,
  seed 2027. Models and thresholds are held fixed.
- **Primary hypotheses (pre-specified in this script, before running it):**
  AURC(Guardian-O) − AURC(baseline) and AURC(Guardian-Full) − AURC(baseline).
  These use a two-sided bootstrap p-value with Holm correction over the 2 tests.
- **Secondary / exploratory:** pooled AUROC (correct vs incorrect), the
  within-class **stratified AUROC** (only same-class correct/incorrect pairs),
  macro within-class AUROC, matched-coverage risk, fixed-threshold
  coverage/risk, per-class metrics, and leave-one-class-out (LOCO) sensitivity.
- Clip-level Wilson intervals are also reported. They ignore clustering and
  are too narrow (compare the risk CIs in §4.1).

Caveat: the primary hypotheses were fixed after the v1 exploratory AURC
results were already known (§6).

## 4. Results

Clean top-1 accuracy: **67.78%** (122/180). Cluster 95% CI [55.2%, 79.9%];
the clip-level Wilson CI is [60.6%, 74.2%].
Per-class accuracy: pullup 100%, ride_bike 96.7%, pushup 90%, ride_horse
73.3%, fencing 36.7%, punch 10%.

### 4.1 Fixed UCF101-calibrated thresholds (the deployable setting)

| Model | Accepted | Coverage [cluster CI] | Selective risk [cluster CI] | Wrong accepted | Recorded ms/clip |
|---|---|---|---|---|---|
| Baseline | 142 | 78.9% [69.0, 87.3] | 27.5% [11.9, 43.2] | 39 | 126 |
| Guardian-O | 146 | 81.1% [72.0, 88.7] | 26.0% [10.6, 41.8] | 38 | 249 |
| Guardian-Full | 151 | 83.9% [75.9, 90.8] | 25.8% [11.4, 40.9] | 39 | 619 |

On UCF101 these thresholds produced about 70–72% coverage and 19–20% risk.
On HMDB51, **every model over-accepts** (79–84% coverage) and has higher risk.
The calibration does not transfer under dataset shift.
Risk differences against the baseline are not significant (Guardian-O
−1.4 pp [−3.6, +0.5]; Guardian-Full −1.6 pp [−6.8, +3.1]). The models also
accept different numbers of clips, so risks at a fixed threshold are not
coverage-matched.

### 4.2 Pooled ranking metrics (primary)

| Model | AURC [cluster CI] | Pooled AUROC | Stratified (within-class) AUROC |
|---|---|---|---|
| Baseline | 0.362 [0.085, 0.577] | 0.527 | 0.601 |
| Guardian-O | 0.359 [0.080, 0.575] | 0.538 | 0.611 |
| Guardian-Full | 0.294 [0.074, 0.491] | 0.589 | 0.611 |

| Comparison | ΔAURC [95% cluster CI] | p (raw) | p (Holm) |
|---|---|---|---|
| Guardian-O − baseline | −0.0034 [−0.0105, +0.0001] | 0.147 | 0.147 |
| Guardian-Full − baseline | −0.0686 [−0.1085, −0.0043] | 0.024 | 0.048 |

Guardian-Full improves pooled AURC. The result is nominally significant
after Holm correction, but the CI upper bound (−0.004) is close to zero.
Guardian-O shows no detectable effect. The baseline's pooled AUROC of 0.527
means that, pooled across classes, the raw clean score barely beats chance
at separating correct from incorrect predictions on HMDB51.

### 4.3 Within-class reliability (negative finding)

| Comparison | ΔStratified AUROC [95% cluster CI] | p |
|---|---|---|
| Guardian-O − baseline | +0.010 [−0.015, +0.060] | 0.62 |
| Guardian-Full − baseline | +0.010 [−0.086, +0.216] | 0.79 |

Per-class AUROC for baseline → Guardian-Full:
- fencing: 0.42 → 0.34
- punch: 0.11 → 0.05
- ride_horse: 0.79 → 0.94
- pushup and ride_bike: 1.0 → 1.0 (ride_bike has a single error)
- pullup: undefined (no errors)

**There is no evidence that Guardian-Full ranks clips better within a
class.** Most of its pooled AURC gain comes from scoring easy classes
(pullup, pushup, ride_bike) higher overall than hard classes (fencing,
punch). That is a between-class effect. On the two hard classes, all scores
are inversely related to correctness (AUROC < 0.5). Wrong predictions there
are made confidently, and Guardian probes make this slightly worse.

### 4.4 Leave-one-class-out sensitivity

Guardian-Full ΔAURC stays negative whichever class is removed (range
−0.046 to −0.083). Guardian-O stays between −0.001 and −0.005. The pooled
direction is stable, but LOCO cannot separate between-class from within-class
effects, so it does not change the conclusion in §4.3.

### 4.5 Cost (recorded model inference time only)

Guardian-O costs about 2.0× the baseline and Guardian-Full about 4.9×. These
are sums of the recorded X-CLIP `inference_ms` values. They exclude
decoding, corruption, and feature overhead, and no energy was measured.

## 5. Bottom line

- The frozen UCF101 thresholds **do not keep their calibrated coverage or
  risk** on HMDB51 (coverage +9 to +14 pp, risk +7 pp).
- **Guardian-O:** no measurable benefit on HMDB51 at any metric.
- **Guardian-Full:** a modest, borderline pooled AURC gain (Holm p = 0.048)
  at about 4.9× inference cost. The gain is **not supported** by within-class
  analysis. Fixed-threshold risk is not significantly better.
- **Recommended wording:** *"On a 6-class, 180-clip HMDB51 transfer subset,
  Guardian-Full showed a borderline improvement in pooled AURC that appears
  driven by between-class score differences rather than improved
  within-class error detection."*

## 6. Limitations

1. 6 of 51 classes and 180 clips. Each class has only 6–10 independent
   source groups, so CIs are wide and power is low.
2. One-to-one label mapping inflates errors (punch ↔ BoxingPunchingBag,
   fencing ↔ TaiChi). Using a mapping-tolerant correctness label would change
   accuracy and may change reliability conclusions. It was not evaluated, to
   avoid post-hoc redefinition.
3. Pullup has zero errors and ride_bike one, so within-class AUROC rests
   mainly on fencing, punch, ride_horse, and pushup.
4. The v1 AURC results were seen before V2's primary hypotheses were
   written. The pilot clips overlap the evaluation set.
5. Source groups are inferred from filenames. Recordings of the same actor or
   scene under different prefixes would not be detected.
6. Possible pretraining overlap was not checked. X-CLIP was trained on
   Kinetics (YouTube), and HMDB51 contains some YouTube-derived clips.
7. Bootstrap p-values are percentile-inversion approximations. With 50
   clusters, percentile intervals may under-cover. BCa or a permutation test
   could serve as a robustness check (not done).
8. Corruptions are synthetic and use a single seed per clip/condition.
   X-CLIP inference was **not re-run** for V2. V2 re-analyses the existing
   900-row prediction file.
9. Latency is recorded model time on one Apple-silicon (MPS) machine. It is
   not end-to-end.

## 7. Reproduction

```bash
# from repository root
.venv312/bin/python research_analysis/cross_dataset_hmdb51/validation_v2/hmdb51_validation_v2.py
.venv312/bin/python -m pytest research_analysis/cross_dataset_hmdb51/validation_v2/test_hmdb51_validation_v2.py -q -p no:cacheprovider
```

The script refuses to overwrite existing outputs. Use `--output-dir` for
re-runs. Runtime is about 7 s. Python 3.12.16, numpy 1.26.4. No new packages
were installed.

## 8. Files

| File | Content |
|---|---|
| `hmdb51_validation_v2.py` | Analysis script (read-only on inputs) |
| `test_hmdb51_validation_v2.py` | 10 tests: hashes, manifest, v1 reproduction, metric unit tests, no-overwrite |
| `outputs/validation_v2_results.json` | All metrics, CIs, p-values, Holm |
| `outputs/per_clip_scores.csv` | 180 clips: source group, correctness, all three scores, features, ms |
| `outputs/per_class_metrics.csv` | Per-class accuracy, AUROC, AURC, fixed-threshold acceptance |
| `outputs/leave_one_class_out.csv` | LOCO AURC sensitivity |
| `outputs/risk_coverage_curves.csv` | Full risk-coverage curves (for Phase 5 figures) |
| `outputs/run_manifest.json` | Command, versions, input SHA256s, seed, no-retraining flags |
