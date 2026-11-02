# Phase 3 Parity Validation — Result: **FAILED (criterion P3)**

Date: 2026-11-01.
- **Criteria:** `feasibility/CRITERIA_FREEZE.json`, SHA256 `73e4c496389a830c…`,
  unchanged before and after the run. The criteria were frozen before any
  surrogate code existed.
- **Raw results:** `feasibility/parity_results.json`.
- **Log:** `feasibility/parity_run.log` (Python exit 0; 50/50 clips; 1,155 s).
- **Code:** `feasibility/xclip_surrogate.py`, `feasibility/parity_validation.py`,
  `feasibility/test_surrogate.py` (30 unit tests pass).
- **Data:** 50 DEV clips (25 UCF101 Guardian-development, 25 HMDB51 Phase 2B
  set P), × 19 view types = 950 view comparisons. **No FINAL data used.**

## 1. Results against the frozen criteria

| Criterion | Frozen threshold | Measured (attacker surrogate, STE mode) | Result |
|---|---|---|---|
| P1 pixel max (pre-normalization) | ≤ 1/255 = 3.9e-3 | 1.19e-7 | **PASS** |
| P1 pixel mean | ≤ 0.25/255 = 9.8e-4 | 1.40e-8 | **PASS** |
| P2 top-1 agreement | 100% | 100% (950/950) | **PASS** |
| P2 max \|Δp\| | ≤ 2e-3 | 2.29e-6 | **PASS** |
| P2 median \|Δp\| | ≤ 2e-4 | 3.95e-7 | **PASS** |
| P2 monitor score max diff | ≤ 5e-3 | MSP 1.6e-6; G-F 5.3e-7; TG-min 2.1e-6; TG0 1.2e-6; energy 2.6e-5 | **PASS** |
| P2 accept/reject agreement | identical | 100% for all 5 monitors | **PASS** |
| **P3 gradient (FD vs autograd), 25 directions** | **rel. error ≤ 5% for every direction** | **24/25 ≤ 0.35% (median 0.02%); 1/25 = 8.70%** | **FAIL** |
| P6 determinism (frozen path ×2, 10 clips) | identical | identical; also bit-identical to Phase 2B exports | **PASS** |

P4 (attack decision agreement) and P5 (budget after quantization) apply to
the pilot. They were **not run**, because the pilot is gated on P1–P3 + P6.

## 2. The P3 failure (exact)

| Clip | Direction | Finite difference | Autograd | Abs. error | Rel. error |
|---|---|---|---|---|---|
| `HorseRace/[clip id redacted]` | 3 | +2.7327e-07 | +2.4950e-07 | 2.38e-08 | **0.0870** |

- **Context (measured):** in the other 24 directions, |directional
  derivative| ranged from 2.3e-6 to 9.1e-5. The failing direction's derivative
  is 9–360× smaller.
- **Setup:** central differences, h = 0.01 grey levels per element on average,
  float64 on CPU.
- **Scalar checked:** top-1 minus top-2 logit of `clean_adaptive`, in the
  **smooth** surrogate. Its gradient defines the attacker's backward pass.

**Possible causes (hypotheses, untested):**
- (a) **FD truncation or round-off** dominates when the true derivative is
  near zero. A pure relative-error criterion is unstable there; an absolute
  error of 2.4e-8 would be negligible for optimization.
- (b) **Local non-smoothness.** The top-2 order could change within ±h, the
  margin is non-differentiable at ties, and attention softmax saturation
  could also play a role.
- (c) **An implementation defect** affecting only some inputs. Considered
  unlikely given 24/25 agreement at the 1e-4 level, but **not excluded**.

## 3. Engineering findings (measured; informational, not criteria)

- **The STE surrogate's forward pass reproduces the frozen pipeline to float
  precision.** Bit-exact PIL resampling, exact OpenCV blur, uint8 truncation,
  occlusion masks and index provenance are all unit-tested, and logits match
  the frozen exports to about 1e-6.
- **The smooth path differs from the frozen pipeline.** This is what the
  attacker's gradients come from:
  - max pixel difference 0.294 (pre-normalization), mean 0.0011;
  - top-1 agreement 98.5%; max |Δp| 0.114;
  - TG0 accept/reject agreement 98%.

  The large maximum pixel difference most likely comes from views involving
  `dark` uint8 truncation and blur rounding, but that is not isolated. **The
  gradient is therefore a BPDA approximation of the exact pipeline.** That is
  acceptable by design (all outcomes are decided by the frozen path, P4), but
  it is a source of attack weakness to monitor.
- **Non-differentiable operations,** treated as fixed per step (BPDA):
  - adaptive frame selection (motion argmax);
  - index provenance for frame drop, delay, reverse and shuffle;
  - seeded occlusion box positions;
  - argmax ŷ for the Guardian/TG features;
  - uint8 rounding and truncation;
  - PIL fixed-point arithmetic.

## 4. Consequence

Per the owner's instruction and the frozen gate, **the pilot was not
executed.** Options for the owner (none taken):

1. **Diagnose without changing the criterion.**
   - Repeat P3 on the failing clip and direction with several step sizes
     (h/10, h×10) and Richardson extrapolation.
   - Check whether the top-2 classes change within ±h.
   - Re-test the margin computed on a fixed class pair.

   This is diagnosis only; the frozen P3 result remains a FAIL.
2. **Revise the criterion prospectively** with documented justification,
   e.g. a combined tolerance |fd − ad| ≤ 5%·|ad| + 1e-7, or a minimum
   derivative magnitude for the relative test. Then run a **new** P3 on
   **new** directions/clips under the revised, newly frozen criterion. This
   must be recorded as a deviation; the original failure stays reported.
3. **Stop Phase 3 attack work.**
