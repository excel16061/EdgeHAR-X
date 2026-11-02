# Phase 1 Independent Audit: HMDB51 Cross-Dataset Validation V2

Date: 2026-11-01. Scope: audit of
`research_analysis/cross_dataset_hmdb51/validation_v2/` (Phase 1). Nothing
was retrained, recalibrated, re-inferred, committed or uploaded. Every
number below comes from `statistical_audit.py` (see `run_manifest.json`
and §9).

> The HMDB51 evaluation is a **6-class, 180-clip subset**: the official
> split-1 test clips of fencing, punch, pushup, pullup, ride_bike and
> ride_horse. It is **not** a 51-class HMDB51 evaluation.

---

## 0. Executive summary

1. **Every Phase 1 number reproduces exactly.** This holds for the per-clip
   scores (max abs diff 2.2e-16), the frozen thresholds, the UCF101 final
   evaluation, the AURCs, the bootstrap CIs (bit-identical), the p-values,
   Holm, and leave-one-class-out. The audit code was written independently
   of the Phase 1 code.
2. **The reported Guardian-Full effect is real as arithmetic but misleading
   as science.** ΔAURC = −0.069. About 76% of it (−0.052) is what you would
   expect from **class composition alone**, i.e. if neither monitor had any
   within-class information. Within-class error detection shows no detectable
   improvement under any method.
3. **On HMDB51, no monitor ranks errors better than chance in AURC.** A
   constant score (random ranking) gets AURC 0.322. MSP gets 0.362 (worse
   than random) and Guardian-Full 0.294. Every monitor's AURC-minus-random
   CI includes 0. Guardian-Full's "improvement" mostly undoes MSP's
   worse-than-random class ordering.
4. **The significance claim (Holm p = 0.048) is fragile.** It holds for
   group-level bootstraps and jackknife with the 6 classes held fixed. It
   fails when:
   - class sampling is acknowledged (hierarchical bootstrap: p = 0.10, CI
     includes 0), or
   - Guardian-Full is compared within the family of simple baselines (Holm
     over 6 comparisons: p = 0.096).
5. **A zero-parameter rule matches Guardian-Full** at the same 5-pass cost.
   The rule is "how many probes agree with the clean label, ties broken by
   MSP". At UCF101-calibrated thresholds on HMDB51 it has lower risk (25.3%
   vs 25.8%) and a lower miss rate. Guardian-O is almost identical to the
   2-pass lexicographic rule "occlusion flips → reject first, else MSP".
6. **Calibration does not transfer.** The thresholds were chosen only on
   UCF101 (verified). On HMDB51:
   - coverage rises from about 70% to 79–84%;
   - the monitors let through 66–67% of errors (vs 44–46% on UCF101);
   - risk reduction vs accepting everything is 4.8 pp (MSP, CI includes 0)
     to 6.4 pp (Guardian-Full).
7. **The source groups are not fully independent.** *Glory* (one film)
   appears in two classes, and three *Return of the King* clips sit in two
   classes. There are 47 class-agnostic film groups (46 if the two LOTR films
   are merged), not 50. This barely changes the CIs.

**Recommendation for Phase 2: REVISE** (§8). The claim "Guardian V3 improves
cross-dataset reliability" is not supported. A narrower, defensible research
question remains open (see `PHASE2_RESEARCH_PLAN.md`): *do pooled
failure-detection metrics under dataset shift measure class difficulty rather
than error detection, and can any lightweight monitor detect errors within a
class?*

---

## 1. Inputs, hashes, environment

| Artifact | SHA256 (prefix) | Role |
|---|---|---|
| `results/guardian_v3_official/frozen_models_v2.json` | `e56fecec…1cd1` ✔ expected | Frozen models and thresholds |
| `results/guardian_v3_official/evaluation_balanced.csv` | see manifest | UCF101 evaluation, 505 videos |
| `results/guardian_v3_official/calibration_balanced.csv` | see manifest | UCF101 calibration, 202 videos |
| `research_analysis/cross_dataset_hmdb51/hmdb51_full_predictions.csv` | `4fbad918…4507` | HMDB51, 180 clips × 5 conditions |
| `research_analysis/cross_dataset_hmdb51/hmdb51_mapped_test_manifest.csv` | `bdc113dc…63d1` | HMDB51 subset manifest |
| Phase 1 outputs (`validation_v2/outputs/*`) | see manifest | Reproduction targets |

- **Environment:** Python 3.12.16, numpy 1.26.4, scipy 1.17.1, macOS arm64.
  No packages were installed.
- **Seeds:** the bootstrap and permutations use seed 2027. The sensitivity
  grid uses seeds {1, 42, 2027, 12345, 99991} × B ∈ {1k, 5k, 10k, 50k}.
- **Unchanged inputs:** all input hashes match before and after the run
  (asserted in code).
- **Model provenance:**
  - Backbone: X-CLIP `microsoft/xclip-base-patch32`, zero-shot over 101
    UCF101 labels, adaptive 8-frame sampling.
  - `score` = max softmax probability (MSP) over the 101 labels
    (`edgehar.py:115-127`).
  - Guardian-O and Guardian-Full are L2-regularised logistic regressions
    trained on 202 UCF101 development videos.
  - Thresholds are the midpoint between the 141st and 142nd ranked
    calibration score (target coverage 70% on 202 UCF101 calibration
    videos). Rule: `guardian_v3_freeze_corrected.py:get_threshold`.
- **UCF101 leakage check:** development, calibration and evaluation are
  group-disjoint. Zero shared `(class, gXX)` groups, verified.

---

## 2. A — AURC and risk-coverage

| Check | Result |
|---|---|
| Definition | AURC = (1/N) Σ_{k=1..N} (errors among top-k)/k, sorted by descending score. Identical in `guardian_v3_risk_coverage.py`, the v1 scripts, Phase 1 and the audit. |
| Ties | **No tied scores** for any monitor on either dataset. Stable-order AURC equals the tie-aware (expected over random tie order) AURC to 1e-12. Ties matter only for the controls, which use tie-aware AURC. |
| Same clips | All monitors are scored on the same 180 HMDB51 clips (505 UCF101), and the bootstrap reuses one multiplicity vector per replicate for every monitor. The comparisons are paired. |
| Minor inconsistency | `guardian_v3_risk_coverage.risk_at_coverage` uses `floor(f·N)`; the HMDB scripts use `round(f·N)`. At N = 180 and f ∈ {0.5, 0.7, 0.8} both give the same k. No impact. |
| UCF101 reproduction | AURC 0.12793 / 0.12738 / 0.12390 (MSP / G-O / G-F) and fixed-threshold coverage and risk match `risk_coverage_report.json` and `final_evaluation.json` exactly. |

### Why −0.069, CI [−0.109, −0.004], Holm p = 0.048

- **Point estimate:** AURC(G-F) 0.29377 − AURC(MSP) 0.36233 = −0.06856.
- **CI:** percentile interval of 10,000 stratified cluster-bootstrap
  replicates (seed 2027). Source groups are resampled with replacement
  within each class.
- **Raw p:** 120 of 10,000 replicates were ≥ 0, so p = 2 × 0.012 = 0.024.
  The Monte-Carlo SE of p is 0.0015.
- **Holm** over the two primary tests:
  - Guardian-Full: 2 × 0.024 = 0.048
  - Guardian-O: max(0.048, 1 × 0.147) = 0.147

  Both reproduce exactly.
- **What drives the number:** class means of each score vs class accuracy.

| Class | Accuracy | Mean MSP | Mean G-O | Mean G-F |
|---|---|---|---|---|
| pullup | 1.000 | 0.612 | 0.791 | 0.820 |
| ride_bike | 0.967 | 0.478 | 0.737 | 0.769 |
| pushup | 0.900 | **0.410** (lowest) | 0.711 | 0.734 |
| ride_horse | 0.733 | 0.454 | 0.734 | 0.746 |
| fencing | 0.367 | 0.445 | 0.700 | 0.682 |
| punch | 0.100 | 0.471 | 0.711 | 0.709 |

  MSP gives the 90%-accurate class (pushup) the lowest confidence and the
  10%-accurate class (punch) above-median confidence. Guardian-Full reorders
  the class means almost into accuracy order (Spearman ρ = 0.89, n = 6,
  p = 0.019; MSP ρ = 0.49). Most of the pooled gain comes from this
  reordering (§4).

---

## 3. B — Cluster-aware uncertainty and alternative inference

### 3.1 Group structure (audit finding)

- **Phase 1 definition:** a group is the class plus the filename prefix
  before `_<class>_`. This gives 50 groups of size 1–15 (10 singletons).
- **Prefixes spanning classes:** *Glory* appears in punch and ride_horse.
  *Return of the King* parts 2, 4 and 6 (and *Two Towers* 3) appear across
  punch and ride_horse. Phase 1 counted these as independent groups.
- **Class-agnostic film grouping:** stripping a trailing part number gives
  47 groups. Also merging the two LOTR films gives 46. The rule is fixed in
  `film_group()` and tested.
- **No byte-identical duplicates:** SHA256 of all 180 clip files, 180
  unique values.
- **Prefixes shared with other HMDB51 classes:** 11 source prefixes in the
  subset (mostly films) also occur in other HMDB51 classes (any partition).
  This does not leak anything, since nothing is trained on HMDB51. It does
  confirm that groups are film-level, not clip-level.
- **Residual risk:** channels or actors that use different prefixes (e.g.
  several "Rapier" sources) cannot be detected from filenames.

### 3.2 ΔAURC(Guardian-Full − MSP) = −0.0686 under alternative methods

All rows use 10,000 replicates or permutations, seed 2027.

| Method | 95% CI | p | Excludes 0? | Assumption / comment |
|---|---|---|---|---|
| Clip bootstrap (ignores groups) | [−0.106, −0.036] | <1e-4 | yes | **Invalid:** treats correlated clips as independent. Shown for contrast. |
| Cluster bootstrap, stratified, percentile (**Phase 1**) | [−0.108, −0.004] | 0.024 | yes | Classes fixed |
| Cluster bootstrap, stratified, BCa | [−0.134, −0.023] | — | yes | z0 = −0.25, a = −0.07 |
| Cluster bootstrap, unstratified (50 groups) | [−0.111, −0.004] | 0.024 | yes | Class mix varies |
| Film-level cluster bootstrap (46 groups) | [−0.111, −0.003] | 0.027 | yes | Cross-class films merged |
| Cluster jackknife-t, 50 groups (df 49) | [−0.133, −0.004] | 0.038 | yes | SE 0.032 |
| Cluster jackknife-t, film groups (df 45) | [−0.133, −0.004] | 0.038 | yes | |
| **Hierarchical bootstrap (class → group)** | **[−0.131, +0.003]** | **0.105** | **no** | Treats the 6 classes as sampled activities. Crude with only 6 classes. |
| Paired swap randomization on rank scores (film groups) | — | 0.020 | — | Heuristic exchangeability (see below) |
| Clustered DeLong/Obuchowski, pooled ΔAUROC = +0.061 | [+0.007, +0.116] | 0.028 | yes | Different metric (AUROC), 50 groups. Film groups give p = 0.030. |

**Within-class ΔAUROC (G-F − MSP) = +0.010:**
- cluster bootstrap CI [−0.086, +0.216], p = 0.79
- cluster jackknife CI [−0.120, +0.141], p = 0.87

### 3.3 Guardian-O (ΔAURC = −0.0034)

- Not significant under the Phase 1 bootstrap (p = 0.147), the unstratified
  and film-level bootstraps (p = 0.16, 0.12), the jackknife (p = 0.21–0.23),
  the hierarchical bootstrap (p = 0.38) or clustered DeLong (p = 0.19–0.22).
- Two methods cross 0.05:
  - **BCa:** the upper bound is −1.2e-5, i.e. it excludes 0 by a hair. The
    bootstrap distribution has a point mass near 0, because G-O only changes
    the ranking of the few clips where occlusion flips the label. BCa is
    known to be unstable for such distributions.
  - **Swap randomization:** p = 0.045.

  Reporting either as evidence would be selecting a favorable method after
  the fact. **Conclusion: no robust evidence for Guardian-O.**

### 3.4 Assessment of methods

- **Percentile bootstrap with 50 clusters:** with 10 singletons and a few
  large groups (12, 15 clips), percentile intervals can under-cover. Here,
  BCa and jackknife-t are *wider* than percentile (lower bound about −0.13
  vs −0.11) and still exclude 0, so the Phase 1 interval is not
  anti-conservative *for the fixed-class question*.
- **The key issue is the estimand, not the interval method.** The Phase 1
  bootstrap conditions on these 6 classes. The effect is mostly between
  classes (§4), so its sign depends on how 6 class means happen to be
  ordered. That is effectively n = 6. Once class sampling is allowed for, the
  interval includes 0. **Any claim beyond "these 6 classes" is unsupported.**
- **Swap randomization** assumes the two monitors' rank scores are
  exchangeable within a source group under H0. Rank normalisation makes the
  marginals equal but not jointly exchangeable, and the null it tests
  (identical joint distribution) is stronger than "equal AURC". Treat it as
  a heuristic.
- **Within-class label permutation (§4)** treats clips as exchangeable
  within a class. Source-group correlation makes its p-values approximate.
  It is used mainly for decomposition.
- **Seed and replicate sensitivity** (`seed_sensitivity.csv`):
  - Guardian-Full: with B ≥ 5,000, p ranges 0.019–0.028 across 5 seeds. The
    CI upper bound ranges −0.0035 to −0.0059 and always excludes 0. At
    B = 1,000, p ranges 0.014–0.04.
  - Guardian-O: never excludes 0 (p 0.136–0.17).
  - Seed choice does not change any conclusion, but the Guardian-Full upper
    bound always sits within about 0.006 of zero.

**Verdict for B:** for *these six classes*, Guardian-Full's pooled AURC is
lower than MSP's under every defensible clustered method. The result does
not generalise beyond them. Within-class detection shows no effect under any
method.

---

## 4. C — Cross-dataset generalisation and decomposition

### 4.1 Subset verification

- The manifest equals exactly the set of label-`2` (test) clips in
  `<class>_test_split1.txt` for each of the 6 classes: 30 per class, i.e.
  all of them, not a sample.
- The predictions cover exactly those 180 clips × 5 conditions with 0
  errors.
- The label mapping is one-to-one into UCF101: Fencing, Punch, PushUps,
  PullUps, Biking, HorseRiding.
- **Mapping problems:**
  - punch → BoxingPunchingBag in 19/30 clips (HMDB "punch" includes film
    fist-fights and boxing).
  - fencing → TaiChi in 14/30 clips.
  - ride_horse → Archery or HorseRace in 7/30 clips.

  Correctness was **not** redefined (no post-hoc change).

### 4.2 Decomposition (`decomposition.csv`)

**Method.** Shuffle the correctness labels within each class (10,000
permutations) while keeping the scores fixed. This keeps each class's
accuracy and each monitor's score distribution, and destroys any
within-class information. The mean AURC under this shuffle is the AURC
explained by class composition alone.

| Dataset | Monitor | AURC | AURC from class composition only | p (within-class info) | Pooled AUROC | Within-class AUROC |
|---|---|---|---|---|---|---|
| HMDB51 | MSP | 0.362 | 0.335 | 0.88 | 0.527 | 0.601 |
| HMDB51 | Guardian-O | 0.359 | 0.333 | 0.86 | 0.538 | 0.611 |
| HMDB51 | Guardian-Full | 0.294 | 0.282 | 0.69 | 0.589 | 0.611 |
| HMDB51 | agreement→MSP | 0.353 | 0.325 | 0.87 | 0.557 | 0.622 |
| HMDB51 | constant (control) | 0.322 | 0.322 | 1.00 | 0.500 | 0.500 |
| UCF101 | MSP | 0.128 | 0.147 | **0.002** | 0.805 | 0.594 |
| UCF101 | Guardian-Full | 0.124 | 0.149 | **0.001** | 0.821 | 0.620 |

**Guardian-Full ΔAURC vs MSP on HMDB51:**

| Part | Value | Share |
|---|---|---|
| Total | −0.0686 | 100% |
| Expected from class composition alone | −0.0524 | ≈ 76% |
| Remaining within-class part | −0.0162 | ≈ 24%; not significant (within-class ΔAUROC p = 0.79) |

- **No monitor has detectable within-class information on HMDB51**
  (p 0.69–0.95). On UCF101 every monitor does (p ≤ 0.031).
- On HMDB51, MSP's AURC is *worse* than its class-composition-only value
  (0.362 vs 0.335). Its within-class ranking actively hurts, because:
  - in punch (AUROC 0.11) and fencing (0.42), confidence is
    **inversely** related to correctness;
  - Guardian-Full makes both worse: punch 0.05, fencing 0.34;
  - Guardian-Full helps ride_horse (0.79 → 0.94);
  - pushup and ride_bike are at ceiling (1.0; ride_bike has a single error);
  - pullup is undefined (no errors).
- **UCF101 contrast:** Guardian-Full ΔAURC = −0.0040 [−0.011, +0.003],
  p = 0.27. Within-class ΔAUROC = +0.026 [−0.062, +0.110]. The −0.069 on
  HMDB51 is 17× the in-distribution effect, which is itself a warning sign
  of a composition effect.

### 4.3 Leave-one-class-out (reproduced exactly)

Guardian-Full ΔAURC stays in [−0.083, −0.046] whichever class is removed.
LOCO shows the result does not depend on any single class. It cannot
separate between-class from within-class effects: with 5 classes left, the
class-mean ordering still drives AURC.

### 4.4 Answer to C

The observed Guardian-Full improvement is **mainly a class-composition
effect specific to this dataset**. Guardian-Full assigns higher scores to
classes that are easier for X-CLIP on this subset. Within a class, it does
not separate correct from incorrect predictions better than MSP. Whether its
class-level ordering generalises to other classes cannot be tested with 6
classes.

---

## 5. D — Calibration transfer (`calibration_transfer.csv`)

- **Thresholds:** for MSP, G-O and G-F the audit recomputed them from
  `calibration_balanced.csv` with the frozen rule, and they match the frozen
  JSON exactly (diff 0.0). `evaluation_data_used: false`. No HMDB51 label
  touched any threshold.
- **Baseline thresholds:** set by the same rule on UCF101 calibration only.
- **Definitions:**
  - *False-alarm rate* = correct predictions rejected / all correct.
  - *Miss rate* = incorrect predictions accepted / all incorrect.
  - *Risk reduction* = error rate when accepting everything − selective
    risk.

| Monitor | UCF101 cov / risk / FAR / miss | HMDB51 cov / risk / FAR / miss | HMDB51 risk reduction [cluster CI] |
|---|---|---|---|
| MSP | 72.1 / 20.1 / 16.1 / 46.2 | 78.9 / 27.5 / 15.6 / 67.2 | +4.8 pp [−1.0, +12.8], p = 0.11 |
| Guardian-O | 70.7 / 19.3 / 17.0 / 43.7 | 81.1 / 26.0 / 11.5 / 65.5 | +6.2 pp [+0.7, +13.9], p = 0.020 |
| Guardian-Full | 71.9 / 19.3 / 15.6 / 44.3 | 83.9 / 25.8 / 8.2 / 67.2 | +6.4 pp [+1.5, +13.3], p = 0.006 |
| agreement→MSP (0 params) | 69.5 / 18.5 / 17.6 / 41.1 | 81.1 / 25.3 / 10.7 / 63.8 | +6.9 pp [+1.7, +14.4], p = 0.005 |
| occ-agreement→MSP (0 params) | 70.5 / 19.1 / 17.0 / 43.0 | 81.1 / 26.0 / 11.5 / 65.5 | +6.2 pp [+0.7, +13.9], p = 0.020 |

All values are percentages. On HMDB51 the error rate when accepting
everything is 32.2%; on UCF101 it is 31.3%.

- Under shift, every monitor **accepts more** and **misses more errors**:
  about two-thirds of HMDB51 errors pass the gate, vs under half on UCF101.
  The false-alarm rate falls, which is the mirror image: the monitors become
  more permissive.
- **Discrimination vs calibration.** Discrimination (AURC/AUROC) degrades
  to chance level on HMDB51 (§4). Calibration (the coverage/risk achieved at
  a fixed threshold) drifts by +9 to +14 pp coverage and +6.5 to +7.4 pp
  risk.
  - Both fail. Fixing calibration with target-domain recalibration would not
    fix discrimination.
  - The modest fixed-threshold risk reductions (4.8–6.9 pp) come mostly from
    rejecting low-scoring clips, whose class mix happens to be harder.
- **Useful without target calibration?** Only marginally. Guardian-Full
  reduces risk by 6.4 pp at about 4.9× the cost. A zero-parameter
  agreement rule at the same cost does the same or slightly better. MSP
  alone (1×) gives 4.8 pp, not significant. G-F's and G-O's
  fixed-threshold risks are not significantly different from MSP's
  (Phase 1: CIs include 0).

---

## 6. Simple-baseline comparison (`baseline_comparison.csv`)

- **Unavailable:** entropy, top-2 margin, energy and any logit- or
  embedding-based score. The saved artifacts contain only the top-1
  label and its probability. These require re-running inference with full
  probability vectors saved. **Not computed, nothing imputed.**
- **Computed:** zero-parameter scores built from the saved views (clean +
  4 severity-3 probes). They were defined in the audit script before
  computing them, but **after** Phase 1 results were known, so they are
  exploratory.

| HMDB51 | Passes | AURC | ΔAURC vs MSP [CI] | Holm (6 comparisons) | Within-class AUROC | ΔAUROC within class vs MSP [CI] |
|---|---|---|---|---|---|---|
| MSP | 1 | 0.362 | — | — | 0.601 | — |
| Guardian-O | 2 | 0.359 | −0.003 [−0.010, +0.000] | 0.29 | 0.611 | +0.010 [−0.015, +0.060] |
| Guardian-Full | 5 | 0.294 | −0.069 [−0.108, −0.004] | **0.096** | 0.611 | +0.010 [−0.086, +0.216] |
| mean view conf | 5 | 0.440 | +0.077 [+0.014, +0.125] | 0.012 (worse) | 0.589 | −0.012 |
| min view conf | 5 | 0.457 | +0.095 [+0.025, +0.161] | 0.002 (worse) | 0.582 | −0.019 |
| agreement→MSP | 5 | 0.353 | −0.009 [−0.029, +0.001] | 0.25 | 0.622 | +0.021 [−0.074, +0.224] |
| occ-agreement→MSP | 2 | 0.359 | −0.004 [−0.011, +0.000] | 0.29 | 0.611 | +0.010 [−0.015, +0.060] |
| *random / constant* | 0 | 0.322 | | | 0.500 | |
| *class oracle (uses labels)* | — | 0.091 | | | 0.500 | |

- **Versus random on HMDB51:** AURC − error rate is +0.040 for MSP,
  −0.028 for Guardian-Full and +0.031 for agreement→MSP. **All CIs include
  0** (e.g. G-F [−0.169, +0.086]).
- **Versus random on UCF101:** every monitor beats random by 0.15–0.19,
  with CIs well away from 0.
- **UCF101 Δ vs MSP:** no monitor improves significantly. Min-view
  confidence is significantly worse.
- **Class oracle (diagnostic only):** it uses HMDB51 labels. A score that
  only knows each clip's class accuracy, with zero within-class
  information, gets AURC 0.091. That is far better than any monitor, which
  shows how much pooled AURC rewards class-level separation.
- **Cost:** MSP takes 126 ms per clip on HMDB51; 2-pass monitors take
  249 ms and 5-pass monitors 619 ms. These are recorded X-CLIP model times
  only. Decoding, corruption and feature computation are excluded, and
  energy was not measured.

---

## 7. Confirmed, corrected and new findings vs the Phase 1 report

| Phase 1 statement | Audit status |
|---|---|
| All numerical results (accuracy, AURCs, CIs, p, Holm, fixed thresholds, LOCO) | **Confirmed**, exact reproduction |
| "Thresholds do not transfer" | **Confirmed**, extended with FAR and miss rate |
| "Guardian-O: no measurable benefit" | **Confirmed** (2 of 10 methods cross 0.05; not robust) |
| "Guardian-Full borderline pooled gain, Holm p = 0.048" | **Confirmed numerically; scope corrected.** Valid only conditional on these 6 classes. p = 0.10 under the hierarchical bootstrap and 0.096 under Holm across the baseline family. |
| "Gain appears driven by between-class differences" | **Confirmed and quantified:** about 76% is class composition, within-class part not significant |
| "50 source groups" | **Corrected:** 50 class-specific groups, of which some share a film across classes (*Glory*, LOTR). 47 or 46 class-agnostic groups. Conclusions unchanged. |
| "Primary hypotheses pre-specified" | **Corrected wording:** they were specified *after* the v1 exploratory AURC results were seen. They are not preregistered. |
| "Guardian probes make this slightly worse" (fencing, punch) | **Corrected:** true for Guardian-Full. Guardian-O is slightly *better* on fencing (0.45 vs 0.42) and slightly worse on punch. |
| (new) MSP and every monitor are no better than random ranking on HMDB51 | **New negative finding** |
| (new) A zero-parameter agreement rule matches or beats Guardian-Full; Guardian-O is about equal to a lexicographic occlusion rule | **New negative finding** |
| (new) Entropy and margin baselines impossible from saved artifacts | **New limitation** |

---

## 8. GO / REVISE / NO-GO

**REVISE.**

- **NO-GO** on "Guardian V3 improves cross-dataset reliability". The
  evidence does not support it.
- **NO-GO** on any security or attack-detection claim. Nothing adversarial
  was tested.
- **GO**, conditional on the revisions in `PHASE2_RESEARCH_PLAN.md` §0, for
  a reframed study: *within-class failure detection for zero-shot video
  recognition under dataset shift, and how pooled metrics confound it*. This
  needs:
  1. a preregistered protocol before new data is touched;
  2. saved full probability vectors, which enables entropy, margin and
     energy baselines;
  3. a larger, externally defined class mapping, e.g. the established
     12-class UCF↔HMDB overlap used in video domain adaptation (verify
     before use);
  4. within-class metrics as primary endpoints;
  5. a literature review before any novelty claim.

---

## 9. Reproduction

```bash
# from repository root (~2 min)
.venv312/bin/python research_analysis/cross_dataset_hmdb51/phase1_audit/statistical_audit.py
.venv312/bin/python -m pytest research_analysis/cross_dataset_hmdb51/phase1_audit/ -q -p no:cacheprovider
```

- **Outputs:** `audit_results.json`, `baseline_comparison.csv`,
  `inference_sensitivity.csv`, `seed_sensitivity.csv`, `decomposition.csv`,
  `calibration_transfer.csv`, `run_manifest.json`.
- **No overwriting:** the script refuses to overwrite any of them. Use
  `--output-dir` to regenerate elsewhere.
- **Metric definitions:** in the docstrings of `aurc`, `aurc_expected`,
  `auroc`, `operating_point`, `within_class_label_permutation`,
  `obuchowski` and `swap_randomization`.
