# Data and Model Access

**This repository contains no video data, no neural-network weights and no
per-clip records.**

## Datasets (obtain from the providers, under their terms)

| Dataset | Used for | Source |
|---|---|---|
| UCF101 | 101 class names (prompts); Guardian development, calibration and evaluation splits (official split files) | UCF Center for Research in Computer Vision (CRCV): https://www.crcv.ucf.edu/data/UCF101.php |
| HMDB51 | Phase 1: 6 mapped classes, split-1 test. Phase 2B: 12 TA3N categories (split-1 train) plus 6 more (split-1 test) | Serre Lab, Brown University (the HMDB51 page has moved; the old URL redirects to https://serre.lab.brown.edu/) |

The terms of use for both datasets must be checked with the providers.
They were not verified during release preparation.

## Expected local layout (not included; git-ignored)

```
datasets/UCF-101/<Class>/v_<Class>_gXX_cYY.avi
datasets/splits/ucfTrainTestlist/{classInd,trainlist01,testlist01}.txt
datasets/HMDB51/extracted/<class>/*.avi
datasets/HMDB51/mapped_subset/videos/<class>/<class>/*.avi
research_analysis/cross_dataset_hmdb51/hmdb51_splits/testTrainMulti_7030_splits/*.txt
```

## Model

- **X-CLIP B/32:** `microsoft/xclip-base-patch32`, revision
  `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.
- **Licence:** MIT, per the model card.
- **How it is obtained:** downloaded by `transformers` at run time.
