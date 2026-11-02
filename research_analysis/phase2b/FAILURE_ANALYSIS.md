# Failure Analysis — Phase 2B (descriptive, post hoc)

Everything here is descriptive and post hoc. Values come from
`results/per_clip_table.csv` and `results/STATISTICAL_RESULTS.json` (set P,
mapped correctness, n = 797, 189 errors). None of it is a confirmatory claim.

## 1. Where the errors are

| Class | Accuracy | Errors | Main confusion | Within-class AUROC (MSP / G-F / TG0) |
|---|---|---|---|---|
| walk | 7.5% | 62 | TaiChi 10, BoxingSpeedBag 5, MoppingFloor 4 (only 5 predicted WalkingWithDog) | 0.50 / 0.41 / 0.47 |
| kick_ball | 50.0% | 35 | SoccerJuggling 12, Basketball 4 | 0.63 / 0.72 / 0.67 |
| fencing | 53.8% | 30 | TaiChi 21 | **0.37 / 0.26 / 0.34** (inverted) |
| punch | 66.1% (strict 12.9%) | 21 | TaiChi 8, Fencing 4 | 0.71 / 0.67 / 0.72 |
| shoot_ball | 72.9% | 19 | BasketballDunk 15 | 0.48 / 0.62 / 0.58 |
| climb | 87.0% | 9 | PullUps 3 | 0.96 / 0.96 / 0.98 |
| golf | 92.9% | 5 | HammerThrow 5 | 0.78 / 0.97 / 0.97 |
| pushup | 94.0% | 4 | HandstandPushups 2 | 0.93 / 0.93 / 0.95 |
| ride_horse | 94.7% | 3 | HorseRace 2 | 0.88 / 1.00 / 0.97 |
| ride_bike | 98.6% | 1 | — | 1.00 / 1.00 / 1.00 |
| pullup, shoot_bow | 100% | 0 | — | undefined (no errors) |

Five classes account for 167 of 189 errors (88%). Most errors are
**semantic-neighbour confusions** within the 101-label space:
- fencing/punch/walk → TaiChi;
- shoot_ball → BasketballDunk;
- kick_ball → SoccerJuggling.

They are not random failures.

## 2. Silent failures (errors the deployed gate accepts)

At the frozen UCF101-calibrated thresholds:

| Gate | Errors accepted (of 189) |
|---|---|
| MSP | 61 |
| TG0 | 48 |
| both | 43 |

- **MSP-accepted errors by class:** fencing 19, shoot_ball 16, kick_ball 11,
  punch 6, walk 4, others 5.
- **22 errors had MSP ≥ 0.5,** 12 of them in fencing.
- **Mean MSP:** 0.513 for correct predictions, 0.262 for errors.

**Mechanism (hypothesis).** The persistent silent errors are *temporally
consistent* confusions with a semantically close label. Every segment shows
the same "wrong but plausible" content (two people with swords → TaiChi; a
basketball shot near the hoop → BasketballDunk). Consistency-based monitors
cannot flag an error that is consistent over time. This matches the
expected failure mode #1 in `temporal_monitor_design.md`.

## 3. Clip length (exploratory; confounded)

- **Short clips** (fewer than 64 decoded frames, n = 169) had an error rate
  of 50.9%, vs 16.4% for longer clips. Within-class AUROC was ≈0.51–0.52 for
  every monitor on short clips, vs 0.73–0.80 on long clips.
- **Confound:** 132 of the 169 short clips belong to the three hardest
  classes (kick_ball 49, fencing 42, punch 41). Clip length and class
  difficulty are entangled, and this analysis cannot separate them.

## 4. Temporal disruptions did not create failures

- Frame drop (hold-last, up to 66% per-frame probability), delay (up to 50%
  of the clip stale), full reversal and segment shuffle changed accuracy by
  ≤1 pp, and every 95% CI included 0.
- The few flips (11–38 per view) were concentrated in low-confidence clips:
  the flip AUROC of clean-view MSP was 0.85–0.95.
- **Interpretation:** at 8 sampled frames, X-CLIP B/32 behaves mostly like an
  appearance classifier on these classes. Benign temporal disruption is not a
  meaningful failure source here. RQ3 is negative, consistent with Schiappa
  et al. (CVPR 2023).

## 5. Monitor-specific failure modes observed

- **Guardian-Full** is worst inside fencing (0.26). Its corruption probes
  agree with confident TaiChi errors.
- **Negative entropy** is worse than MSP: within-class Δ −0.051 [−0.092,
  −0.010]. The 101-way tail mass apparently adds noise.
- **TG-JS** (distribution change) has high pooled AUROC (0.80), but its
  pooled AURC CI vs MSP includes 0, and its macro within-class AUROC (0.711)
  ranks mid-pack.
- **TG-min / TG-agreement** have the highest within-class AUROC (0.70), but
  their advantage is uncorrected and fails the hierarchical criterion.

## 6. Implications

1. Failures are dominated by **label-space semantic neighbours**, plus a
   mapping artefact for *walk*. These are both evaluation-design issues and
   genuine model weaknesses.
2. The safety-relevant residual risk is **confident, temporally consistent
   misclassification**. Benign temporal noise is not the issue.
3. Any security study should target exactly that regime: inducing
   consistent, confident wrong predictions. That needs an adversary, not
   natural corruption.
