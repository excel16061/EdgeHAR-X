# P3 Diagnostic Report

- **Date:** 2026-11-01.
- **Scope:** diagnosis only, as approved by the owner.
- **Not changed:** `CRITERIA_FREEZE.json`, `parity_results.json`,
  `parity_run.log`, `PHASE3_PARITY_VALIDATION.md` and the production
  surrogate/harness (SHA256 verified before and after:
  `preserved_files_before.sha256`, all OK).
- **Status of P3:** **remains FAILED** under the frozen criterion. Nothing here
  changes that.

**Run:** `p3_diagnose.py` (log `p3_diagnose.log`, Python exit 0, 169 s);
raw data in `P3_STEP_SIZE_SWEEP.csv` and `p3_diagnostic_results.json`.

## 1. Reproduction of the failing measurement — **confirmed**

- **Inputs:** the same 5 DEV clips (`parity_validation.dev_clips()[:5]`), the
  same RNG (seed 20271101, same draw order), the same objective (top-1 minus
  top-2 logit of the smooth `clean_adaptive` view), float64 CPU, and the same
  step h₀ = 0.01·√numel = 32.92 along a unit-norm direction.
- **HorseRace [clip id redacted], direction 3, reproduced:**
  - FD = 2.73268e-7 (|Δ| from the frozen record 1.1e-16);
  - autograd = 2.49503e-7 (|Δ| 3.7e-20);
  - relative error 0.0870, with denominator max(|FD|, |autograd|) = |FD|.
- **The other 7 reproduced measurements** match the frozen record within
  4.8e-11 (FD) and 2e-18 (autograd).

## 2. Independent derivative: forward-mode JVP — **confirmed agreement**

`torch.func.jvp` (forward mode) computes the directional derivative without
reverse-mode autograd and without finite differences.

| Clip / direction | Autograd (reverse) | JVP (forward) | Rel. diff |
|---|---|---|---|
| HorseRace d0 | +5.200347e-05 | +5.200347e-05 | <1e-6 |
| HorseRace d1 | +4.236793e-05 | +4.236793e-05 | <1e-6 |
| HorseRace d2 | +3.716260e-05 | +3.716259e-05 | ~3e-7 |
| **HorseRace d3 (failing)** | **+2.495027e-07** | **+2.495138e-07** | **4.4e-5** |
| HorseRace d4 | −9.060253e-05 | −9.060252e-05 | <1e-6 |
| ShavingBeard d0 | +2.291900e-06 | +2.291899e-06 | <1e-6 |
| PlayingSitar d2 | +2.752848e-06 | +2.752848e-06 | <1e-6 |
| SumoWrestling d2 | −6.579874e-05 | −6.579875e-05 | <1e-6 |

**Observation (confirmed).** Two independent automatic-differentiation modes
agree, including on the failing direction. Its slightly larger relative
difference (4.4e-5) is consistent with float64 accumulation error on a
derivative about 100× smaller than the others.

## 3. Finite-difference step sweep — **confirmed V-shaped error**

Central FD relative error vs autograd, for step multiples of h₀ (full table:
`P3_STEP_SIZE_SWEEP.csv`).

| Direction | 0.001 h₀ | 0.01 h₀ | 0.1 h₀ | 0.3 h₀ | **1 h₀ (frozen)** | 3 h₀ | 10 h₀ |
|---|---|---|---|---|---|---|---|
| HorseRace d0 | 0.0003 | 0.0058 | 0.0004 | 0.0001 | 0.0002 | 0.0011 | 0.0110 |
| HorseRace d1 | 0.0726 | 0.0031 | 0.0007 | 0.0001 | 0.0006 | 0.0064 | 0.0749 |
| HorseRace d2 | 0.1962 | 0.0192 | 0.0008 | 0.0005 | 0.0002 | 0.0029 | 0.0308 |
| **HorseRace d3** | 1.0757 | 0.7074 | 0.2226 | **0.0167** | **0.0870** | 0.3824 | 0.8695 |
| HorseRace d4 | 0.0382 | 0.0041 | 0.0002 | 0.0002 | 0.0001 | 0.0014 | 0.0160 |
| ShavingBeard d0 | 0.4318 | 0.1034 | 0.0044 | 0.0012 | 0.0034 | 0.0310 | 0.2211 |
| PlayingSitar d2 | 0.1415 | 0.0090 | 0.0002 | 0.0015 | 0.0001 | 0.0003 | 0.0031 |
| SumoWrestling d2 | 0.0776 | 0.0025 | 0.0011 | 0.0001 | 0.0001 | 0.0000 | 0.0003 |

**Observations (confirmed).**
- **Every direction shows the classic FD error curve.** At small steps, error
  is large and erratic: round-off and cancellation in f(x+h) − f(x−h),
  f ≈ 2.13. At large steps, error grows smoothly: truncation.
- **For d3, the FD estimate converges toward autograd/JVP at an intermediate
  step:** 0.3 h₀ gives 2.537e-7, a relative error of 1.7%. The frozen step
  h₀ sits on the truncation side.
- **Absolute truncation error at h₀** (|FD − autograd|) for the five HorseRace
  directions: 1.3e-8, 2.7e-8, 7e-9, **2.4e-8 (d3)**, 1.1e-8. The failing
  direction's absolute error is **typical**. Its *relative* error is large
  only because its derivative (2.5e-7) is 150–360× smaller than the other
  directions on this clip.
- **Truncation scaling (d3):** the error grows from 2.4e-8 (h₀) to 1.5e-7
  (3 h₀; ×6.5) to 1.7e-6 (10 h₀; ×70). That is close to the O(h²)
  expectation (×9, ×100), given that the second-order term is not exactly
  isolated.

## 4. Non-smoothness checks — **no evidence found**

- **No pair change.** At every evaluation point (8 directions × 7 steps ×
  ±h), the top-1/top-2 class pair equals that at x₀ (`top_pair_changed` =
  False for all 56 rows). The top-2/top-3 logit gap at x₀ is 0.619
  (HorseRace), so no class switch is near.
- **Fixed frame indices.** Sampled frame indices are fixed within the check
  (computed once from x₀), so adaptive-sampling discontinuities cannot enter.
- **Smooth path only.** The smooth path contains no rounding, clamping or
  argmax inside f: linear resize, crop, normalization, then X-CLIP
  (QuickGELU, softmax, LayerNorm). The margin is smooth while the top-2 pair
  is constant.
- **Forward/backward FD** differ symmetrically around the central estimate,
  which is the curvature signature of smooth truncation. No one-sided jump
  indicative of a kink appears.

## 5. Other checks

- **Direction energy.** Each random direction places about 8.3% of its energy
  (8/96) on the 8 frames that are actually sampled. d3 is not special
  (0.0834). Its small derivative is explained by near-orthogonality of the
  random direction to the gradient: a random projection, plausible among 25
  draws. *Likely, not proven.*
- **Precision (float32 CPU vs float64 CPU autograd, same direction):**
  relative difference 5e-4 for d3, ≤ 5e-5 for d0–d2 and d4. FD in float32 at
  h₀ for d3 = 2.607e-7.
  - Float32 is adequate for the attacker's gradient direction.
  - This is a **precision** effect, separate from preprocessing parity,
    which P1/P2 established on the frozen float32 path.
- **Framework.** The earlier dtype crash (`transformers` causal-mask
  construction with float64) was handled by `torch.set_default_dtype` in
  this diagnostic, as in the second parity run.

## 6. Limits of this diagnosis (unresolved / not covered)

- Only the **smooth `clean_adaptive` margin** was checked, as P3 was
  designed. Gradients through the other differentiable views (blur conv,
  dark scaling, occlusion masks, segment/index views) and through the
  **monitor terms** used by adaptive attacks (TG-min softmax min,
  Guardian-Full sigmoid with BPDA surrogates) were **not** validated by P3 or
  by this diagnosis.
- The **STE/BPDA gradient** (smooth backward through an exact forward) is an
  approximation by design. FD on the exact forward is ill-defined (piecewise
  constant), so its quality can only be assessed indirectly, through attack
  success decided by the frozen path (pilot P4).
- 8 directions on 4 clips were examined. That is broad enough to show the
  failure mechanism, but it is not a population-level validation.
