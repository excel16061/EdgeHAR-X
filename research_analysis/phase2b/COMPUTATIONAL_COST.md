# Computational Cost — Phase 2B

Device: Apple-silicon Mac, PyTorch MPS. Python 3.12.16, torch 2.15.0,
transformers 4.57.6. Model: X-CLIP B/32, revision `a2e27a78…`.

## 1. Measured during the confirmatory exports (set P, first 3 clips excluded)

Times are per clip and unbatched. Each view is a separate forward pass with a
device-synchronised timer. **Model ms** is the forward pass only.
**End-to-end ms** is the shared decode plus Σ over the monitor's views of
(transform + frame sampling + preprocessing + model).

| Monitor | Forward passes | Model ms, median (p95) | End-to-end ms, median (p95) | Within-class AUROC |
|---|---|---|---|---|
| MSP, entropy, margin, energy, max-logit | 1 | **120** (142) | **799** (2,158) | 0.58–0.64 |
| Guardian-O | 2 | 249 (299) | 934 (2,313) | 0.62 |
| TG-sampling | 2 | 261 (304) | 944 (2,312) | 0.65 |
| TG-seg2 | 3 | 406 (475) | 1,105 (2,485) | 0.65 |
| Guardian-Full / corruption-TTA | 5 | 626 (774) | 1,350 (2,881) | 0.64–0.65 |
| TG0, TG-min, TG-agreement, TG-JS | 6 | 835 (975) | 1,583 (2,965) | 0.65–0.70 |

- **Decoding dominates end-to-end latency:** median 661 ms per clip to decode
  up to 96 frames with OpenCV and resize. So the relative overhead of
  monitors is smaller end-to-end (TG0 ≈ 2.0× MSP) than in model time (≈ 7×).
- **Throughput of the whole export pipeline (19 views per clip):**
  - UCF101: 404 clips in 1,429 s (3.54 s per clip);
  - HMDB51: 974 clips in 3,677 s (3.78 s per clip).
- **Within-class gain per extra forward pass:** TG0 +0.007 [−0.001, +0.016];
  Guardian-Full +0.007 [−0.009, +0.024]. Neither excludes 0.

## 2. Measured separately (exploratory; 2 Phase 1 smoke clips; `memory_profile.json`)

| Quantity | Value |
|---|---|
| MPS memory allocated after model load + inference | 780 MB (driver 1,107 MB) |
| Process peak RSS | 592 MB |
| Single forward pass (incl. preprocessing), median | 146 ms (n = 10) |
| TG0's 6 views, sequential (incl. preprocessing), median | 886 ms |
| TG0's 6 views, **batched** in one forward call (forward only), median | **404 ms** (p95 1,137 ms; first-call warm-up included) |
| Decode, median | 296 ms (n = 2) |

**Batching (estimate, not validated at scale).** Batching the 6 temporal views
roughly halves model time. The batched figure excludes preprocessing and comes
from only 2 clips, so it is indicative only.

## 3. Not measured

- Energy or power: needs `powermetrics` (sudo), which was not used.
- GPU utilisation.
- Latency on embedded or edge hardware.
- Any real-time streaming pipeline. Exports process stored clips; no camera
  or stream buffering was simulated.

## 4. Summary

In this setup, a temporal-consistency monitor costs about 0.7 s extra model
time per decision (about 0.3 s if batched; estimate). Its within-class gain
over MSP was not statistically confirmed. **On current evidence the overhead
is not justified for within-class error detection.** For thresholded
deployment, TG-min / TG-agreement gave the lowest selective risk at reduced
coverage (secondary, uncorrected).
