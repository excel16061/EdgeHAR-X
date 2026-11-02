# Methodology

This is a consolidated summary. The authoritative protocols are the frozen
documents cited in each section.

## 1. Pipeline (frozen)

- **Decoding:** up to 96 evenly spaced frames, resized to width 224.
- **Frame sampling:** 8 frames, chosen *adaptively*. Per temporal bin, the
  frame with the largest grey-level change (64×64 difference) is taken
  (`edgehar.sample_indices`; that module is not included in
  the reduced release).
- **Model:** X-CLIP B/32, `microsoft/xclip-base-patch32`, revision
  `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.
  - Prompt: "a video of a person {class}".
  - Candidate labels: the 101 UCF101 class names, alphabetical.
- **Views per clip:** 19, defined in
  `research_analysis/phase2a/config/phase2b_config.json`:
  - the clean adaptive view;
  - corruption probes (blur, dark, occlusion, frame drop);
  - a uniform-sampling view and four temporal segments;
  - temporal disruptions.

## 2. Reliability monitors

Definitions: `research_analysis/phase2a/baseline_metrics.py`. All monitors
are label-free, and the accept threshold is fixed on UCF101 calibration
data (70% target coverage).

| Monitor | Forward passes | Definition (short) |
|---|---|---|
| MSP | 1 | max softmax probability of the clean view |
| Energy, max-logit, margin, neg-entropy | 1 | standard single-pass scores |
| Guardian-O / Guardian-Full | 2 / 5 | frozen logistic model on probe-agreement features (`results/guardian_v3_official/frozen_models_v2.json`, SHA256 `e56fecec…1cd1`) |
| TG0 (TemporalGuard, primary) | 6 | mean over {uniform view, 4 segments} of p(ŷ) |
| TG-min | 6 | min over the same views of p(ŷ) |
| Corruption-TTA | 5 | same views as Guardian-Full, averaged (ablation for "temporal vs any multi-view") |

## 3. Data

- **UCF101:**
  - prompts;
  - Guardian development and calibration splits;
  - the Guardian v3 final evaluation: 505 group-disjoint clips from the
    official *training* partition, because the official test list had been
    inspected in earlier experiments.
- **HMDB51, Phase 1:** the 6 classes mapped one-to-one to UCF101; 180
  split-1 test clips; 50 source groups.
- **HMDB51, Phase 2B:** the 12 TA3N UCF↔HMDB categories
  (`research_analysis/phase2a/DATASET_MAPPING.md`).
  - **Set P** (primary): split-1 train partition, 797 clips.
  - **Set S** (secondary): 6 further categories, split-1 test, 177 clips.
  - **Leakage control:** perceptual hashing against Phase 1 clips.
- **No HMDB51 data were used** for training, calibration or threshold
  selection.

## 4. Statistics (Phase 2B, pre-registered)

- **Frozen protocol:** `research_analysis/phase2a/PREREGISTRATION.md`
  v1.0, frozen before any confirmatory inference.
- **Primary metric:** *within-class* AUROC, i.e. the probability that a
  monitor scores a correct clip above an incorrect clip of the **same
  class**.
- **Hypotheses:**
  - H1: MSP vs 0.5;
  - H2: Guardian-Full − MSP;
  - H3: TG0 − MSP.
- **Decision rule:** reject only if **both**:
  1. the 95% film-group cluster-bootstrap CI (B = 10,000) excludes the
     null, **and**
  2. the Holm-adjusted cluster-jackknife p < 0.05 (Holm over H1–H3).
- **Equivalence:** TOST with a ±0.03 margin.
- **Always reported alongside:**
  - hierarchical (class + group) intervals;
  - pooled AURC with a class-composition decomposition;
  - fixed-threshold coverage and risk;
  - negative controls.
- **Validity checks (6):** smoke equivalence, decode failures, leakage,
  minimum errors and classes, frozen hashes, protected artifacts.

## 5. Phase 1 audit

An independent re-implementation of the Phase 1 statistics:
- class-composition decomposition of AURC;
- random-ranking baselines;
- hierarchical bootstrap;
- zero-parameter heuristics.

Report: `research_analysis/cross_dataset_hmdb51/phase1_audit/AUDIT_REPORT.md`.

## 6. Phase 3 feasibility (in progress; no robustness experiment run)

Phase 3 plans to test whether monitor signals survive inputs chosen to
defeat them. That requires a **differentiable surrogate** of the frozen
pipeline, whose *forward* pass reproduces the frozen preprocessing exactly
and whose *backward* pass uses smooth approximations (straight-through /
BPDA).

**Completed validation steps:**

1. **Parity (P1, P2, P6), frozen criteria.** On 50 development clips × 19
   views, the surrogate output was compared with the frozen pipeline.
2. **Original gradient check P3**, frozen before any code was run: central
   finite differences vs autograd, relative error ≤ 5% in every one of 25
   directions. **This FAILED** (1/25 directions at 8.7%), and the failure
   stands.
3. **P3 diagnostic.** The failing direction had a near-zero derivative.
   Forward-mode JVP agreed with autograd, and a step-size sweep showed
   finite-difference truncation/round-off behaviour. Assessment: a
   numerical artefact of the check. This does **not** change the P3 FAIL.
4. **P3-v2**, a new protocol adopted prospectively as a documented
   deviation:
   - frozen by hash before its run;
   - 5 fresh development clips;
   - JVP vs autograd;
   - a scale-aware finite-difference criterion;
   - gradient-free checks that one surrogate gradient step moves the
     *frozen* pipeline's objective more than 20 random steps.

   **It PASSED. It is reported separately and does not overturn the
   original P3 FAIL.**
5. **CPU feasibility timing:** Apple MPS vs CPU backward cost.

The surrogate and validation code are not part of this release (see
`REPRODUCIBILITY.md` §3).
