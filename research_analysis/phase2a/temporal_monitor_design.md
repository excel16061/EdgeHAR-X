# Temporal-Consistency Monitor: Design (not trained, not evaluated)

Status: design only. Every definition below is frozen in `baseline_metrics.py`
and `config/phase2b_config.json`. Nothing was tuned on any label.

## 1. Idea and hypothesis

A zero-shot video classifier sees 8 frames. If its prediction ŷ is right, the
evidence for ŷ should be **spread across the clip**: other temporal samples
should also support ŷ. If ŷ is wrong, it is more often driven by a short,
atypical moment (one punch that looks like a punching bag, one frame of a
horse), so support for ŷ should be **unstable across time**.

**This is a hypothesis.**
- **Against it:** in Phase 1 (exploratory), multi-view *corruption*
  agreement gave no within-class gain on HMDB51.
- **Also against it:** Schiappa et al. (CVPR 2023) report Kinetics-type
  models are fairly insensitive to temporal perturbation. For appearance-
  dominated classes, temporal views may just repeat the same evidence.

## 2. Signals (all label-free, from exported logits)

Let ŷ = argmax of `clean_adaptive` and pᵥ = softmax of view v. The temporal
views are T = {`clean_uniform`, `seg4_0`, `seg4_1`, `seg4_2`, `seg4_3`}.

| Signal | Formula | Captures | Extra passes |
|---|---|---|---|
| **TG0 (primary)** | mean_{v∈T} pᵥ(ŷ) | average temporal support for the prediction | 5 |
| `tg_min` | min_{v∈T} pᵥ(ŷ) | weakest segment's support | 5 |
| `tg_agreement_then_msp` | 2·#{v∈T: argmax pᵥ = ŷ} + MSP | segment vote stability, ties broken by confidence | 5 |
| `tg_neg_js` | −mean_{v∈T} JS(pᵥ ‖ p_clean) | change of the whole distribution | 5 |
| `tg_sampling_pred_prob` | p_uniform(ŷ) | adaptive vs uniform sampling disagreement | 1 |
| `tg_seg2` | mean over 2 halves of pᵥ(ŷ) | cheap variant | 2 |
| disruption sensitivity (RQ3, S7) | p_clean(ŷ) − p_disrupted(ŷ) for delay / drop / shuffle / reverse | sensitivity to temporal disruption | 1 each |

**Why these:**
- **TG0** is the simplest test-time-augmentation-style statistic restricted
  to *temporal* resampling. It has no parameters, so it cannot be tuned and
  cannot overfit.
- **The other variants** probe *which aspect* of temporal consistency
  matters (average vs worst case vs vote vs full distribution).

**Optional learned variant (secondary, Phase 2B only if approved):**
- logistic regression on the 6 TG signals;
- trained on UCF101 *development* exports, thresholds from UCF101
  *calibration* exports, using the frozen 70% rule;
- **never** fit on HMDB51.

## 3. Ablation that isolates "temporal"

`corruption_tta_pred_prob` = mean over the 4 Guardian corruption probes of
pᵥ(ŷ). It has the same functional form as TG0 but uses appearance
perturbations instead of temporal resampling.

| Outcome | Interpretation |
|---|---|
| TG0 ≈ corruption-TTA | any gain is generic multi-view averaging |
| TG0 > corruption-TTA within class | evidence for temporal structure specifically |

## 4. Cost vs baselines

| Monitor | Forward passes | Expected model time on MPS* | Notes |
|---|---|---|---|
| MSP / entropy / margin / energy | 1 | ~0.13 s | free once logits are saved |
| `tg_sampling_pred_prob` | 2 | ~0.25 s | cheapest temporal signal |
| `tg_seg2` | 3 | ~0.38 s | |
| Guardian-O | 2 | ~0.25 s | |
| Guardian-Full / corruption-TTA | 5 | ~0.63 s | |
| **TG0** and variants | 6 | ~0.75 s | uniform + 4 segments + clean |

\*Model time only, from Phase 1 recordings (about 125 ms per pass);
decode is shared.

**Cost reductions (design only, not evaluated):**
- **Cascade:** run temporal views only when MSP lies in an uncertain band
  set on UCF101 calibration.
- **Batching:** the 5 temporal views can be batched in one forward call.
  Exported model times are per view, unbatched, so they are an upper bound.

## 5. Expected failure modes

1. **Consistent confident errors.** If the whole clip looks like the wrong
   class (HMDB *walk* vs UCF *WalkingWithDog*; *punch* shot as boxing), all
   segments agree on ŷ. TG0 ≈ MSP, with no gain. This is exactly the Phase 1
   punch pattern (within-class AUROC 0.05–0.11).
2. **Short clips.** For clips under 32 frames, segments repeat frames, so
   views are less independent and TG0 is biased toward agreement. Analysis:
   report TG0 within-AUROC split by clip length (<64 vs ≥64 decoded frames)
   as exploratory.
3. **Segments that legitimately differ.** Some actions are multi-phase
   (golf address → swing). Disagreement there is natural, not an error,
   which produces false alarms.
4. **Class-composition confound.** TG0 can rise for "easy" classes overall.
   It must be evaluated within class, which is why it is the primary metric.
5. **Adversarial evasion (not tested).** An attacker controlling all
   segments defeats any consistency check. TG0 is not a security mechanism
   (Carlini & Wagner 2017).

## 6. What would make TG0 a contribution

TG0, or a preregistered variant, would need to beat MSP **and** the best of
entropy/margin/energy in within-class AUROC by ≥0.03, with CIs excluding 0
under both inference methods, on set P. It would need to be replicated on set
S and with strict correctness, and to beat corruption-TTA. Anything less is
reported as a negative or inconclusive result.
