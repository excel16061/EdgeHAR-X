# Results (v0.1)

- **What "verified" means here:** every number below was re-checked
  against the frozen result files by an automated claim check (70/70 on
  2026-11-02). That check is internal; it is not an external replication.
- **CI convention:** 95% CIs are film-group / source-group cluster
  bootstraps unless stated otherwise.

## 1. UCF101 baseline: Guardian v3 (505 group-disjoint clips, 101 classes)

Source: `results/guardian_v3_official/final_evaluation.json`. Clean top-1
accuracy 68.7%.

| Monitor | Coverage | Selective risk | Model ms per clip (median) |
|---|---|---|---|
| MSP | 72.1% | 20.1% | 125 |
| Guardian-O | 70.7% | 19.3% | 262 |
| Guardian-Full | 71.9% | 19.3% | 669 |

- **Risk differences vs MSP** (paired video-level bootstrap, models and
  thresholds fixed):
  - Guardian-O: −0.7 pp [−2.5, +0.9];
  - Guardian-Full: −0.8 pp [−3.3, +1.7].
- **Neither is significant.**
- **Where the clips came from:** they are group-disjoint clips from the
  official *training* partition. The official test list had been inspected
  earlier, and the probe variants were selected using calibration results.

## 2. Phase 1: HMDB51 cross-dataset check (exploratory)

**Scope:** 180 clips, **6** HMDB51 classes mapped to UCF101, 50 source
groups. **Not a 51-class HMDB51 evaluation.**

Sources: `research_analysis/cross_dataset_hmdb51/validation_v2/outputs/validation_v2_results.json`
and `…/phase1_audit/`.

- **Clean accuracy:** 67.8% (cluster CI [55.2, 79.9]).
- **Pooled AURC, Guardian-Full − MSP:** −0.069. Holm p = 0.048 over 2
  tests.
- **Audit verdict: REVISE.**
  - **About 76%** of that ΔAURC is expected from class composition alone.
  - **No monitor's AURC beats a random ranking**: every ΔAURC-vs-random CI
    includes 0.
  - The significance does **not** survive a hierarchical bootstrap that
    also resamples classes (p = 0.10).
- **Thresholds did not transfer.** The UCF101-calibrated thresholds
  over-accepted on HMDB51: 79–84% coverage against a 70% target.

## 3. Phase 2B: pre-registered confirmatory study (TemporalGuard)

Sources: `research_analysis/phase2b/results/STATISTICAL_RESULTS.json`,
`research_analysis/phase2b/CONFIRMATORY_RESULTS.md`.

- **Data:** set P has 797 clips in 12 categories and 189 mapped errors;
  clean mapped accuracy 76.3%. Set S has 177 clips.
- **All 6 pre-registered validity checks passed**, so confirmatory claims
  were permitted.

### 3.1 Primary hypotheses (set P)

| Hypothesis | Estimate [95% cluster CI] | Holm p | 95% hierarchical CI | Decision |
|---|---|---|---|---|
| H1: within-class AUROC(MSP) vs 0.5 | 0.626 [0.514, 0.726] | 0.055 | [0.431, 0.848] | **Not rejected** (narrow miss) |
| H2: Guardian-Full − MSP | +0.028 [−0.037, +0.098] | 0.401 | [−0.076, +0.143] | **Not rejected**; not equivalent within ±0.03 |
| H3: TG0 − MSP | +0.035 [−0.007, +0.080] | 0.148 | [−0.023, +0.114] | **Not rejected**; not equivalent within ±0.03 |

**Reading the table:**
- **No primary hypothesis was rejected.**
- **H2 and H3 are inconclusive, not "no effect".** The study was powered for
  Δ ≈ 0.04–0.05.

**Erratum:** the frozen report `CONFIRMATORY_RESULTS.md` prints the H3 Holm
p as 0.149. The stored value is 0.14848, which rounds to 0.148. The frozen
report is published unchanged. The decision is unaffected.

### 3.2 Secondary and descriptive findings (no confirmatory claims)

- **Pooled ranking.** Every monitor ranks errors far better than random in
  pooled AURC on these 12 classes. About half of the pooled gain of
  TG0 and Guardian-Full over MSP is reproduced by class composition alone.
- **TG-min had the largest within-class gain:** +0.075 [+0.011, +0.141].
  - It fails the in-family Holm correction and the hierarchical criterion
    (hierarchical CI [−0.005, +0.179]).
  - At fixed thresholds it had the lowest selective risk, 6.2%, at 58.7%
    coverage (target 70%).
- **Temporal structure is not shown to matter.** TG0 did not beat the
  corruption-view ablation (S2: +0.023 [−0.016, +0.073]).
- **Set S replication (S5): treat as hypothesis-generating.** Guardian-Full −
  MSP = +0.219, Holm p = 0.0002. But 45 of set S's 50 errors fall in two
  classes (walk, kick_ball), and the effect does **not** replicate on set P.
- **Benign temporal disruptions** (frame drop, delay, reversal, shuffle)
  changed top-1 accuracy by no more than ±1 pp; every CI includes 0.
- **Cost:**
  - multi-view monitors take 2–7× MSP's model time;
  - decoding dominates end-to-end latency (median 661 ms per clip);
  - within-class gain per extra forward pass is about +0.007, and its CI
    includes 0.
- **Phase 1's "no monitor beats random" does not generalise** beyond the
  6-class subset.

Figures: `research_analysis/phase2b/figures_final/`.

## 4. Phase 3: feasibility and validation (no robustness results exist)

Source: `research_analysis/phase3/phase3_validation_summary.json`, an
aggregate extract with no clip identifiers, plus
`research_analysis/phase3/PHASE3_PARITY_VALIDATION.md` and
`research_analysis/phase3/feasibility/p3_diagnostics/P3_DIAGNOSTIC_REPORT.md`.

| Step | Criterion | Result |
|---|---|---|
| P1 pixel parity (50 clips × 19 views = 950) | max ≤ 1/255; mean ≤ 0.25/255 | **PASS:** 1.2e-7 / 1.4e-8 |
| P2 output parity | top-1 100%; max \|Δp\| ≤ 2e-3; monitor diffs ≤ 5e-3; identical decisions | **PASS:** 100%; 2.3e-6; ≤ 2.6e-5; 100% |
| P6 determinism | identical re-runs | **PASS** |
| **P3 gradient check (original, frozen)** | relative error ≤ 5% in all 25 directions | **FAIL:** 24/25 passed; worst 8.70%. **The failure stands.** |
| P3 diagnostic | — | Near-zero derivative in the failing direction. JVP agrees with autograd; FD numerics are the likely cause. **P3 remains FAILED.** |
| **P3-v2** (separate protocol, prospectively frozen, 5 new clips) | C1 JVP vs autograd; C2s scale-aware FD; C3 validity; C4 and C4b surrogate step vs 20 random steps on the frozen pipeline | **PASS:** C1 60/60 (max 9.9e-13); C2s 60/60; C3 0% not applicable; C4 4/4 losses and C4b 3/3 terms, each on 5/5 clips |
| CPU feasibility timing | — | X-CLIP backward per 8-frame view: 28.2 s on Apple MPS vs 0.13 s on CPU (about 210×) |

**The P3-v2 PASS does not overturn the original P3 FAIL.** Both are reported.
One-step gradient fidelity is validated. Multi-step behaviour is not.

## 5. Earlier EdgeHAR baseline (pre-registration era; not significance-tested)

Source: `research/RESEARCH_LOG.md`.
- **Data:** UCF101, 505 clips.
- **Top-1:** adaptive sampling 66.53% vs uniform 65.35% (net +6 clips).

## 6. Limitations

- **One setting.** One backbone (X-CLIP B/32), zero-shot, 101 prompts,
  8 frames.
- **HMDB51 coverage.** 6 (Phase 1) and 12+6 (Phase 2B) mapped categories
  only.
- **Mapping.** One-to-one label mapping turns some plausible predictions
  into "errors", e.g. HMDB *punch* → UCF101 *BoxingPunchingBag*.
- **Generalisation.** Group-level CIs hold the evaluated classes fixed.
  Hierarchical CIs are wider and often include the null.
- **UCF101 split reuse.** The official test list was inspected in early
  experiments.
- **Phase 3.** Feasibility only. The robustness evaluation is neither
  approved nor run.
- **Measurements.** All come from one Apple-silicon laptop.
- **Verification.** Internal only; no external review.
