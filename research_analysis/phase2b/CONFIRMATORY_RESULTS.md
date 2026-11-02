# Phase 2B Confirmatory Results — TemporalGuard

Date: 2026-11-01.
- **Protocol:** `research_analysis/phase2a/PREREGISTRATION.md` v1.0 (SHA256
  `9cb1fe94…`, frozen in `FREEZE.json` before any confirmatory inference).
- **Analysis:** `phase2b_analysis.py` (SHA256 `3fa41398…`), run **once**
  (`logs/analysis_run.log`, exit 0, 9 min 19 s).
- **Raw outputs:** `results/`.
- **Figures:** `figures_final/`.

> Scope: X-CLIP B/32 zero-shot over 101 UCF101 labels, evaluated on HMDB51
> clips from the 12 TA3N UCF↔HMDB categories. This is **not** a 51-class HMDB51
> evaluation.

## 1. Data actually evaluated

| Set | Clips | Classes | Film groups | Mapped errors | Clean accuracy (mapped) |
|---|---|---|---|---|---|
| **P** (primary: HMDB51 split-1 train partition) | **797** | 12 | 326 (whole manifest) | **189** | **76.3%** |
| **S** (secondary: split-1 test partition, 6 new classes) | **177** | 6 | — | 50 (45 of them in walk and kick_ball) | 71.8% |
| UCF101 reference (development + calibration, thresholds only) | 404 | 101 | — | — | — |

- **Total exported:** 1,378 clips × 19 views = 26,182 forward passes, with
  0 missing and 0 corrupt.
- **Validity checks (PREREGISTRATION §8): all 6 passed**, so confirmatory
  claims are permitted:
  - smoke equivalence: max |Δ| = 2.75e-8;
  - 0 decode failures;
  - leakage: 1,154 videos perceptually hashed, 0 near-duplicates of Phase 1
    clips;
  - 189 errors (≥100 required) and 7 classes with ≥5 correct and ≥5 incorrect
    (≥6 required);
  - frozen hashes intact;
  - protected artifacts unchanged.

## 2. Preregistered primary hypotheses (set P, mapped correctness)

**Decision rule:** reject H₀ only if the 95% film-group cluster-bootstrap CI
excludes the null **and** the Holm-adjusted cluster jackknife p < 0.05 (Holm
over H1–H3). B = 10,000.

| Hypothesis | Estimate | 95% cluster CI | Jackknife p | **Holm p** | 95% hierarchical CI | Decision |
|---|---|---|---|---|---|---|
| **H1** within-AUROC(MSP) = 0.5 | **0.626** | [0.514, 0.726] | 0.0185 | **0.055** | [0.431, 0.848] | **Not rejected** (narrow miss; CI excludes 0.5 but Holm p > 0.05) |
| **H2** Δ within-AUROC (Guardian-Full − MSP) = 0 | **+0.028** | [−0.037, +0.098] | 0.401 | **0.401** | [−0.076, +0.143] | **Not rejected**; not equivalent (90% CI [−0.027, +0.085] exceeds ±0.03) |
| **H3** Δ within-AUROC (TG0 − MSP) = 0 | **+0.035** | [−0.007, +0.080] | 0.074 | **0.149** | [−0.023, +0.114] | **Not rejected**; not equivalent (90% CI [−0.001, +0.072]) |

**No primary hypothesis was rejected.** H2 and H3 are *inconclusive*, not
"no effect":
- the point estimates are small and positive;
- equivalence within ±0.03 could not be established;
- the study was powered for Δ ≈ 0.04–0.05 (PREREGISTRATION §7).

## 3. Secondary endpoints (no confirmatory claims; descriptive)

### 3.1 Replications

| Replication | H1 (MSP vs 0.5) | H2 (G-F − MSP) | H3 (TG0 − MSP) |
|---|---|---|---|
| **S5: set S** (177 clips) | 0.535 [0.378, 0.750], not rejected | **+0.219 [+0.108, +0.339], Holm p = 0.0002, rejected; hierarchical CI [+0.027, +0.406]** | +0.114 [+0.020, +0.245], Holm p = 0.068, not rejected |
| **S6: strict correctness** (P) | 0.604 [0.486, 0.715], Holm p = 0.23 | +0.045 [−0.018, +0.110], Holm p = 0.23 | +0.032 [−0.006, +0.076], Holm p = 0.23 |

**Caution on S5.** Set S has 50 errors, and 45 of them fall in walk (27/30
wrong) and kick_ball (18/30 wrong). The climb and shoot_bow subsets have 0
errors, golf 1, and shoot_ball 4. The large Guardian-Full effect on S
therefore rests on about 2 informative classes, and it does **not** replicate
on the larger primary set P (+0.028). It should be treated as
hypothesis-generating.

### 3.2 Families S2–S4 and other comparisons (Δ within-class AUROC; same decision rule)

| Comparison | Δ | 95% cluster CI | Holm p (in family) | Decision |
|---|---|---|---|---|
| **S2** TG0 − corruption-TTA (does *temporal* structure matter?) | +0.023 | [−0.016, +0.073] | 0.245 | not rejected |
| **S3** neg-entropy − MSP | −0.051 | [−0.092, −0.010] | 0.065 | not rejected (descriptively worse) |
| S3 margin − MSP | +0.012 | [−0.073, +0.088] | 1.0 | not rejected |
| S3 energy − MSP | +0.009 | [−0.091, +0.102] | 1.0 | not rejected |
| S3 max-logit − MSP | +0.009 | [−0.088, +0.097] | 1.0 | not rejected |
| **S4** TG-min − MSP | **+0.075** | **[+0.011, +0.141]** | 0.090 | not rejected (hierarchical CI [−0.005, +0.179]) |
| S4 TG-agreement − MSP | +0.075 | [−0.013, +0.170] | 0.42 | not rejected |
| S4 TG-JS − MSP | +0.022 | [−0.079, +0.124] | 0.66 | not rejected |
| S4 TG-sampling − MSP | +0.025 | [−0.012, +0.063] | 0.42 | not rejected |
| S4 TG-seg2 − MSP | +0.026 | [−0.007, +0.062] | 0.42 | not rejected |
| expl. Guardian-O − MSP | −0.007 | [−0.046, +0.032] | 1.0 | not rejected |
| expl. TG0 − Guardian-Full | +0.007 | [−0.042, +0.057] | 1.0 | not rejected |

### 3.3 Within-class AUROC per monitor (set P; 95% cluster CI)

| Monitor | Passes | Within-class AUROC | Macro (9 classes) |
|---|---|---|---|
| MSP | 1 | 0.626 [0.514, 0.726] | 0.693 |
| neg-entropy | 1 | 0.576 [0.456, 0.689] | 0.631 |
| margin | 1 | 0.638 [0.539, 0.733] | 0.709 |
| energy | 1 | 0.635 [0.505, 0.744] | 0.697 |
| max-logit | 1 | 0.635 [0.501, 0.749] | 0.704 |
| Guardian-O | 2 | 0.619 [0.516, 0.710] | 0.684 |
| Guardian-Full | 5 | 0.654 [0.541, 0.749] | 0.728 |
| corruption-TTA | 5 | 0.637 [0.510, 0.743] | 0.717 |
| **TG0** | 6 | 0.661 [0.550, 0.758] | 0.738 |
| TG-min | 6 | **0.701** [0.597, 0.788] | **0.761** |
| TG-agreement | 6 | 0.701 [0.594, 0.785] | 0.762 |
| TG-sampling | 2 | 0.651 [0.542, 0.751] | 0.716 |
| TG-seg2 | 3 | 0.652 [0.540, 0.753] | 0.727 |
| *random (ctrl)* | — | 0.490 | 0.479 |
| *shuffled MSP (ctrl)* | — | 0.521 | 0.516 |
| *constant / class oracle (ctrl)* | — | 0.500 / 0.500 | — |

**Negative controls behaved as designed:**
- constant AURC = error rate (0.237);
- random and shuffled within-class AUROC ≈ 0.5;
- class oracle within-class AUROC = 0.5, pooled AUROC 0.884.

### 3.4 Pooled reliability (S1), reported separately from within-class

| Monitor | AURC [95% CI] | AURC − random [CI] | Pooled AUROC | ΔAURC vs MSP [CI] | Class-composition share |
|---|---|---|---|---|---|
| MSP | 0.100 [0.062, 0.152] | −0.137 [−0.173, −0.098] | 0.806 | — | — |
| Guardian-Full | 0.089 [0.056, 0.135] | −0.148 | 0.825 | −0.011 [−0.022, −0.001] | 0.56 |
| **TG0** | 0.082 [0.048, 0.128] | −0.155 | 0.845 | **−0.018 [−0.028, −0.011]** | 0.54 |
| TG-min | 0.076 [0.045, 0.117] | −0.162 | 0.850 | −0.025 [−0.041, −0.011] | 0.38 |
| max-logit | 0.075 [0.048, 0.110] | −0.162 | 0.849 | −0.025 [−0.058, −0.001] | ≈0 (−0.01) |
| energy | 0.077 [0.050, 0.112] | −0.160 | 0.844 | −0.023 [−0.057, +0.002] | n/a (CI includes 0) |
| neg-entropy | 0.119 | −0.118 | 0.760 | +0.019 [+0.008, +0.033] (worse) | 0.96 |
| class oracle (diag.) | 0.063 | — | 0.884 | — | — |

- **Every monitor ranks errors far better than random** on these 12 classes.
  This contradicts the exploratory 6-class Phase 1 result, where nothing beat
  random.
- **Temporal-view monitors give the clearest pooled gains**, but about half of
  TG0's and Guardian-Full's pooled gain is reproduced by class composition
  alone.
- **Within-class information exists for every monitor.** The
  within-class label-permutation p is 0.005 for MSP and ≤ 0.018 for all
  monitors (approximate; ignores clustering). It is not strong enough to
  pass the preregistered dual test for H1.

## 4. Calibration transfer (S9; thresholds from UCF101 calibration only)

- The frozen thresholds were reproduced exactly from the re-exported UCF101
  calibration clips (|Δ| = 0).
- On HMDB51 set P the error rate when accepting everything is **23.7%**.

| Monitor | HMDB coverage (target 70%) | Selective risk | False-alarm rate | Miss rate |
|---|---|---|---|---|
| MSP | 66.4% | 11.5% | 23.0% | 32.3% |
| Guardian-O | 67.6% | 11.7% | 21.7% | 33.3% |
| Guardian-Full | 68.0% | 9.8% | 19.6% | 28.0% |
| energy | 68.8% | 9.1% | 18.1% | 26.5% |
| max-logit | 67.0% | **8.2%** | 19.4% | 23.3% |
| TG0 | 63.9% | 9.4% | 24.2% | 25.4% |
| TG-min | 58.7% | **6.2%** | 27.8% | **15.3%** |
| TG-agreement | 61.4% | 6.5% | 24.8% | 16.9% |

Unlike the 6-class Phase 1 subset (79–84% coverage), coverage here stayed
within 1–11 points of the 70% target. The temporal monitors reject more
(lower coverage) and catch more errors.

## 5. Temporal disruptions (S7, RQ3)

**No disruption measurably changed top-1 accuracy.** Every Δ was within
±1 pp, and every 95% CI included 0. The widest was delay3, at −0.9 pp
[−3.3, +1.5].

- **Flip rates** (clean-correct → disrupted-wrong) were 1.8–6.3%. The highest
  was delay3 (50% of the clip stale): 38 flips.
- **Clean-view monitors predicted which clips would flip well:** flip AUROC
  0.85–0.98. Flips happen almost only among low-confidence clips.
- **No "silent failure" pattern** (confident errors induced by benign
  disruption) was observed.
- This is consistent with Schiappa et al. (CVPR 2023) on Kinetics-style
  models.

## 6. Cost (S8; measured; details in COMPUTATIONAL_COST.md)

| Monitor | Median model ms | Median end-to-end ms |
|---|---|---|
| MSP | 120 | 799 |
| Guardian-O | 249 | — |
| Guardian-Full | 626 | 1,350 |
| TG0 | 835 | 1,583 |
| TG-seg2 | 406 | — |

Decoding dominates end-to-end latency (median 661 ms).

**Within-class gain per extra pass:**
- TG0: +0.007 [−0.001, +0.016]
- Guardian-Full: +0.007 [−0.009, +0.024]

## 7. Interpretation

1. **RQ1 (errors vs class difficulty).**
   - On 12 classes, confidence scores *do* carry within-class information.
     The point estimate is 0.63, the permutation test is significant, and the
     cluster CI excludes 0.5.
   - About half of the pooled gains of multi-view monitors over MSP comes from
     class composition.
   - The preregistered H1 test **narrowly failed** (Holm p = 0.055), so we
     cannot make the confirmatory claim that MSP detects errors within class
     under shift.
   - The Phase 1 conclusion "nothing beats random" does **not** generalise
     beyond the 6-class subset.
2. **RQ2 (temporal consistency).**
   - **Not confirmed.** TG0 − MSP = +0.035, not significant and not
     equivalent.
   - TG-min (+0.075) is the largest within-class gain, but it is a secondary
     variant that fails the Holm and hierarchical criteria.
   - TG0 did not beat the corruption-view ablation (S2), so there is no
     evidence that *temporal* structure, as opposed to generic multi-view
     averaging, is what helps.
3. **RQ3 (temporal disruption).** X-CLIP B/32 with 8-frame sampling was
   essentially insensitive to frame drop, delay, reversal and segment
   shuffling on these classes. RQ3 is answered by a negative result.
4. **RQ4 (cost).**
   - Multi-view monitors cost 2–7× the model time of MSP for, at best, a few
     points of within-class AUROC that are not significant.
   - In the deployable setting (fixed thresholds) they halve the selective
     risk relative to accepting everything, as MSP does.
   - TG-min reaches the lowest risk (6.2%) at the price of lower coverage
     (58.7%). This is a secondary, uncorrected observation.

## 8. Exploratory observations (not preregistered as tests)

- **Clip length.** Within-class AUROC was ≈0.51–0.52 for clips with fewer
  than 64 decoded frames (n = 169), vs 0.73 (MSP), 0.78 (TG0) and 0.80
  (Guardian-Full) for longer clips. Short clips may be where every monitor
  fails, which is a candidate for follow-up.
- **Inverted classes.** Within *fencing*, confidence is again inversely
  related to correctness (MSP 0.37, G-F 0.26, TG0 0.34). TaiChi confusions
  are confident. *Walk* has 7.5% accuracy because the external mapping
  allows only WalkingWithDog; predictions were TaiChi, BoxingSpeedBag and
  others.
- **Max-logit and energy** have the best pooled AURC among single-pass scores
  with ~0 class-composition share. Their within-class Δ vs MSP is ≈ +0.01.
  Their pooled advantage seems to come from better *cross-class*
  comparability of scores.
