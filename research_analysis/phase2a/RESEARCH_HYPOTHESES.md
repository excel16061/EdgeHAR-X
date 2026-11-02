# Research Hypotheses — TemporalGuard

*Evaluating Failure Detection in Video Perception Under Distribution Shift and
Temporal Disruption.*

The formal tests, decision rules and data are fixed in `PREREGISTRATION.md`.
This file states the scientific reasoning, expected observations and what
would count against each idea. Prior evidence is from the **exploratory**
Phase 1 (6 classes, 180 clips, now burned) and is used only to form
expectations.

---

## RQ1 — Do monitors detect individual errors under cross-dataset shift, or only rank class difficulty?

- **Prior (exploratory) evidence.** On HMDB51-6:
  - within-class AUROC was MSP 0.60, Guardian-Full 0.61;
  - no monitor's within-class information was detectable;
  - about 76% of Guardian-Full's pooled ΔAURC was reproduced by class
    composition alone;
  - pooled AURC was not distinguishable from random ranking.
- **H1 (confirmatory).** MSP has within-class AUROC ≠ 0.5 on the new
  12-class set.
  - *Null:* within-class AUROC(MSP) = 0.5.
  - *Expected:* modestly above 0.5 (≈0.55–0.65) if X-CLIP's confidence keeps
    some within-class meaning under shift. Possibly ≈0.5, as in Phase 1's
    punch/fencing.
- **H2 (confirmatory).** Guardian-Full differs from MSP in within-class AUROC.
  - *Null:* Δ = 0.
  - *Expected:* Δ ≈ 0 (Phase 1). A preregistered equivalence test (±0.03)
    lets a null result be stated positively.
- **What would contradict "only class difficulty":** within-class AUROC
  clearly above 0.5 (H1 rejected upward) *and* a small class-composition
  share of pooled gains (secondary S1).

## RQ2 — Do temporal-consistency signals improve within-class error detection over confidence baselines?

- **Monitor (fixed now).** TG0 = the mean probability, over five temporal
  views, of the class predicted on the clean adaptive view. The five views
  are a uniform resample plus 4 contiguous segments. TG0 is label-free and
  has zero parameters (see `temporal_monitor_design.md`).
- **H3 (confirmatory).** ΔAUROC_within(TG0 − MSP) ≠ 0.
  - *Null:* Δ = 0.
  - *Expected:* uncertain.
    - **For:** a wrong prediction based on one salient segment should be
      less stable across segments.
    - **Against:** in Phase 1, multi-view corruption agreement gave no
      within-class gain on HMDB51. Schiappa et al. (CVPR 2023) found
      Kinetics-style models largely insensitive to temporal perturbations,
      so temporal views may carry little extra information.
- **Key ablation (S2).** TG0 vs the same statistic over *corruption* views
  (`corruption_tta_pred_prob`). If they tie, any gain comes from multi-view
  averaging, not from temporal structure.
- **What would reject the idea:** H3 not rejected, or rejected only in the
  pooled metric. Also if TG0 does not beat entropy/margin/energy within
  class (S3).

## RQ3 — How do frame loss, frame delay and sampling disruption affect classification reliability and monitors?

- **Disruptions (fixed):**
  - frame drop (hold-last), severities 1–3;
  - delay (stale stream: first frame held, 10/25/50%);
  - full reversal;
  - 4-segment shuffle;
  - adaptive vs uniform sampling.
- **Secondary hypotheses (descriptive, CIs, no confirmatory claim):**
  - S7a: accuracy drops monotonically with drop and delay severity.
  - S7b: MSP within-class AUROC on disrupted views is ≤ its clean value.
  - S7c: reversal and shuffle change accuracy little on UCF/HMDB-style
    classes (Schiappa et al. report temporal robustness on Kinetics but
    not on SSv2).
- **What would be interesting:** disruptions that cause *confident* errors,
  i.e. accuracy falls while MSP does not. That is the safety-relevant
  "silent failure" pattern.

## RQ4 — Can lightweight monitoring deliver meaningful reliability gains at acceptable overhead?

- **Cost model:** forward passes per decision — MSP 1, Guardian-O 2,
  Guardian-Full 5, TG0 6 (clean + 5), tg_seg2 3. Also measured median and
  p95 latency per clip.
- **Secondary (S8):** cost-normalised gain = ΔAUROC_within per extra pass,
  with CI.
- **What counts as "meaningful":** a ΔAUROC_within ≥ 0.03 with a CI
  excluding 0 under both primary inference methods. Below that, overhead is
  not justified.

---

## Threats that could make every RQ uninformative (watch-list)

1. **Too few errors**, i.e. very high accuracy in many classes. Stopping
   rule: fewer than 100 total errors, or fewer than 6 classes with ≥5
   errors and ≥5 correct, means "underpowered; no confirmatory claim".
2. **Very low accuracy from mapping semantics** (e.g. walk ↔
   WalkingWithDog). These classes contribute few correct predictions and
   therefore few within-class pairs.
3. **Group structure.** About 326 film groups over 797 clips. Under the
   synthetic power simulation, the minimum detectable ΔAUROC_within is
   roughly 0.04–0.05 at 80% power. Smaller true effects will likely be
   missed; the equivalence test makes that explicit.
4. **The bootstrap may be anti-conservative.** The simulation rejected a
   true null 9% of the time at nominal 5%. This is why the decision rule
   requires agreement with the cluster jackknife-t.

## Security relevance (no claim in Phase 2B)

RQ1–RQ3 measure *natural* silent failures: errors that a runtime monitor
accepts. They are a prerequisite for, not a substitute for, a security study.
A security claim needs a threat model and adaptive attacks against both
classifier and monitor (Carlini & Wagner 2017; Li et al. NDSS 2019). That is
deferred and requires separate approval.
