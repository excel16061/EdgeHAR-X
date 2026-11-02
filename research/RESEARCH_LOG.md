# EdgeHAR-X Research Log

## Research Objective
Investigate motion-adaptive temporal frame sampling for
efficient zero-shot human activity recognition on edge devices.

## Model
microsoft/xclip-base-patch32

## Dataset
UCF101 official test split.
Balanced subset: 505 videos, 5 per class, 101 classes.

## Baseline Results

| Metric | Uniform | Adaptive |
|---|---:|---:|
| Top-1 accuracy | 65.35% | 66.53% |
| Macro-F1 | 0.5925 | 0.6052 |
| Mean inference latency | 129.89 ms | 131.74 ms |
| Median inference latency | 129.31 ms | 130.18 ms |
| P95 inference latency | 140.31 ms | 143.74 ms |

## Paired Comparison
- Adaptive fixed: 12 videos
- Adaptive regressed: 6 videos
- Both incorrect: 163 videos
- Net improvement: 6 videos

## Research Limitations
- Only 505 videos evaluated, not the full UCF101 test split.
- Zero-shot X-CLIP without UCF101 fine-tuning.
- Reported inference latency excludes some processing.
- Top-5 accuracy not recorded.
- Accuracy improvement not yet statistically established.

## Future Experiments
1. Full temporal coverage.
2. Frame-budget ablation.
3. Natural-language prompt evaluation.
4. End-to-end latency measurement.
5. Per-class error analysis.
6. Full official UCF101 test evaluation.
