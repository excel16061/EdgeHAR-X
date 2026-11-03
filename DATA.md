# Data and Model Access

**This repository contains no video data, no neural-network weights and no
per-clip records.** You must obtain them from their providers, under the
providers' terms. The MIT licence of this repository does **not** apply to
any dataset or model.

**Labels used below:**
- *(verified)*: checked against an official source on the stated date;
- *(maintainer attestation)*: a statement by the maintainer, not
  independently verified;
- *(reported)*: stated by secondary sources only.

## Datasets

| Dataset | Used for | Source | Terms |
|---|---|---|---|
| **UCF101** (Soomro, Zamir & Shah, CRCV-TR-12-01, 2012) | 101 class names (prompts); Guardian development, calibration and evaluation splits | Official CRCV page: https://www.crcv.ucf.edu/data/UCF101.php. The videos and split files used here were downloaded from `www.crcv.ucf.edu` *(verified from local download metadata)*. | The official page requests citation of the technical report and states **no licence and no redistribution permission** *(verified 2026-11-02)*. **Redistribution of the videos or split files is therefore not permitted on the evidence available, and is not done here.** This repository publishes only aggregate statistics and class names. |
| **HMDB51** (Kuehne, Jhuang, Garrote, Poggio & Serre, ICCV 2011) | Phase 1: 6 mapped classes, split-1 test. Phase 2B: 12 TA3N categories (split-1 train) plus 6 more (split-1 test). | Downloaded from the official HMDB51 / Serre Lab (Brown University) source *(maintainer attestation; the local archive carries no download metadata)*. The original page `serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/` has moved and could not be retrieved on 2026-11-02. | **Creative Commons Attribution 4.0** *(reported by secondary sources that cite the official page; not confirmed from the official page)*. Other secondary sources report different terms. A CC BY 4.0 grant by the dataset authors would **not** resolve rights in the underlying film and web clips. **No HMDB51 media are redistributed here.** |

## Expected local layout (not included; git-ignored)

```
datasets/UCF-101/<Class>/v_<Class>_gXX_cYY.avi
datasets/splits/ucfTrainTestlist/{classInd,trainlist01,testlist01}.txt
datasets/HMDB51/extracted/<class>/*.avi
datasets/HMDB51/mapped_subset/videos/<class>/<class>/*.avi
research_analysis/cross_dataset_hmdb51/hmdb51_splits/testTrainMulti_7030_splits/*.txt
```

The UCF101 class index (`classInd.txt`) can be regenerated from your own
UCF101 download with `tools/make_ucf101_classind.py`. Nothing is copied
from the dataset.

## Model

- **X-CLIP B/32:** `microsoft/xclip-base-patch32`, revision
  `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.
- **Licence:** MIT, per the Hugging Face model card *(verified 2026-11-02)*.
- **Citation:** Ni et al., "Expanding Language-Image Pretrained Models for
  General Video Recognition", ECCV 2022 (arXiv:2208.02816).
- **How it is obtained:** the weights are downloaded by `transformers` at
  run time and are **not** included in or relicensed by this repository.
