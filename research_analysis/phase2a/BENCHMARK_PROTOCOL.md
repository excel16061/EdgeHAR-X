# Benchmark Protocol (operational)

This document covers how the benchmark is built and run. The scientific
decisions are in `PREREGISTRATION.md`.

## 1. Pipeline overview

```
official split files ──► benchmark_manifest.py ──► manifests/*.csv + manifest_summary.json
   (HMDB51 split 1,          (mapping verification, leakage + duplicate controls,
    UCF101 classInd,          deterministic clip IDs, SHA256 of videos)
    Guardian UCF splits)
                                   │
                                   ▼
local videos ─────────────► prediction_export.py ──► <clip_id>.npz (logits [19,101] float32)
 (decode ≤96 frames,          (frozen config, 19 views,      <clip_id>.jsonl (per-view provenance + timing)
  width 224)                   pinned X-CLIP revision)       run_metadata.json (config, env, checkpoint hashes)
                                   │
                                   ▼
                            baseline_metrics.py ──► monitors, within-class / pooled metrics,
                            (+ phase2b_analysis.py,      cluster bootstrap / jackknife / hierarchical,
                               written in 2B step 0)     controls, decomposition
```

## 2. Manifest construction (`benchmark_manifest.py`)

1. **Discover splits.** Parse all `<class>_test_split{1,2,3}.txt`. Check
   that there are 51 classes and that each has 3 splits.
2. **Verify the mapping** (`verify_mapping`):
   - TA3N labels exist locally (UCF101 `classInd.txt` with 101 classes;
     HMDB51 split files);
   - no UCF101 label is used twice.

   The run aborts on failure.
3. **Assign roles** from split-1 labels: 1 → primary; 2 → secondary for
   the non-Phase-1 classes only; everything else is unused.
4. **Exclusions:** exact Phase 1 clips, film-group overlap with Phase 1, and
   perceptual near-duplicates of Phase 1 clips (`--phash`).
5. **Clip ID** = `sha256("hmdb51/<class>/<filename>")[:16]`. The ID is
   deterministic and independent of ordering. Rows are sorted by (role,
   class, filename).
6. **Hashes:** SHA256 of every locally available video (`--hash-videos`) and
   of all input split files.
7. **Assertions:**
   - unique clip IDs;
   - no Phase 1 clip;
   - no Phase 1 film group;
   - UCF development/calibration disjoint from each other and from the
     UCF101 evaluation groups.
8. **No overwriting:** outputs are opened with mode `"x"`.

Generated in Phase 2A (preliminary; regenerated once after extraction):
`manifests/hmdb51_confirmatory_manifest.csv`, `manifests/hmdb51_excluded.csv`,
`manifests/ucf101_reference_manifest.csv`, `manifests/manifest_summary.json`.

## 3. Prediction export (`prediction_export.py`)

- **Model:** X-CLIP B/32, revision `a2e27a78…0963`. Loaded with
  `local_files_only=True` and `HF_HUB_OFFLINE=1`. Never downloads.
- **Inference:** the same call sequence as `edgehar.predict`:
  - tokenizer over 101 prompts;
  - video processor on 8 RGB frames;
  - `logits_per_video[0]`.

  The **full logit vector** is stored, not just top-1.
- **Decoding:** identical to `guardian_benchmark.read_frames` (verified
  frame-by-frame), plus source-frame provenance.
- **Views (19):**

  | Group | Views | Purpose |
  |---|---|---|
  | Guardian | `clean_adaptive`, `blur3`, `dark3`, `occlusion3`, `frame_drop3` | identical to Phase 1; Guardian features |
  | Temporal consistency | `clean_uniform`, `seg2_{0,1}`, `seg4_{0..3}` | RQ2 |
  | Disruptions | `frame_drop{1,2}`, `delay{1,2,3}`, `reverse`, `segment_shuffle` | RQ3 |

- **Seeds:**
  - Guardian views use `seed_for(42, relative, transform, severity)`
    (Phase 1 rule; verified against the seeds in the Phase 1 CSV).
  - New views use `seed_for(2027, relative, view_id, 0)`.
- **Segment sampling:** exactly 8 frames, `linspace` within the segment.
  Frames repeat if the segment is shorter than 8 (flag `frames_repeated`).
  Real clips have as few as 23 frames.
- **Per clip:** `<clip_id>.npz` (`logits` [V,101] float32, `view_ids`) and
  `<clip_id>.jsonl`, one record per view:
  - view spec and seed;
  - sampled indices (after transform, into decoded frames, and into source
    frames);
  - decoded count, container frame count, fps;
  - `decode_ms_shared`, `transform_ms`, `sampling_ms`, `preprocess_ms`,
    `model_ms` (device-synchronised);
  - top-1 index.

  No correctness or label fields are written.
- **Run level:** `run_metadata.json` holds:
  - the config and its SHA256;
  - the 101-class order and prompts;
  - the environment (Python/torch/transformers/OpenCV/device);
  - SHA256 of every checkpoint file.
- **Safety:**
  - resume skips completed clips;
  - a partial clip raises an error;
  - a different config in the same directory raises an error;
  - the npz write is atomic;
  - an edited frozen config is rejected (hash check).

## 4. Metrics (`baseline_metrics.py`)

- **Scores from logits:**
  - MSP;
  - negative entropy;
  - top-2 margin;
  - energy (logsumexp, T=1);
  - max logit.
- **Monitors:** Guardian-O/Full (frozen; reproduce Phase 1 to 1e-9 in
  tests), TG family, and corruption-TTA ablations.
- **Metrics:**
  - pair-weighted within-class AUROC (primary), macro within-class AUROC,
    pooled AUROC;
  - AURC (stable and tie-aware), E-AURC, AUGRC, risk@coverage;
  - fixed-threshold operating point (coverage, risk, false-alarm rate, miss
    rate);
  - the frozen 70% threshold rule.
- **Inference:**
  - unstratified film-group cluster bootstrap;
  - hierarchical class→group bootstrap;
  - delete-one-group jackknife-t;
  - Holm;
  - TOST equivalence (90% CI);
  - within-class label-permutation null.
- **Controls:** constant, random, within-class-shuffled MSP, class oracle
  (diagnostic).
- **Performance:** a weighted AUROC takes about 0.4 ms (vectorised
  `searchsorted`). 10,000-replicate bootstraps take seconds.

## 5. Latency and resource measurement

- **Device:** Apple-silicon MPS (record the exact model with
  `sysctl -n machdep.cpu.brand_string` and macOS version in Phase 2B).
- **Model time:** wall-clock around the forward pass with
  `torch.mps.synchronize()`. Preprocessing, sampling, transform and decode
  are timed separately.
- **Summaries:** first 3 clips excluded as warm-up; median and p95 per
  monitor.
- **Not measured:** energy and memory. Energy would need `powermetrics`
  (sudo), so it is excluded unless the owner approves. Peak memory can be
  added via `torch.mps.current_allocated_memory()` if wanted.
- **Smoke measurement (2 clips × 7 views, MPS):** 3.5 s total, about
  0.25 s per view including amortised decode.

## 6. Isolation of smoke and confirmatory data

- Smoke uses only Phase 1 clips (IDs prefixed `smoke_`).
- Smoke outputs live in `research_analysis/phase2a/smoke/` and are flagged
  `smoke_test_excluded_from_confirmatory_analysis: true`.
- The analysis script must refuse any `smoke_` clip ID.
