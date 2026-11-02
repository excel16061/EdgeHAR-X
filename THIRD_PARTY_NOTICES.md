# Third-Party Notices

- **What this repository contains:** research code, documentation and
  aggregate results. It does **not** knowingly redistribute third-party
  source code, model weights or dataset media. See §6 for open provenance
  questions.
- **What it depends on, and cites:** the components below.

> **Licence status of this repository:** no licence has been granted yet;
> all rights are reserved until a `LICENSE` file is added. Nothing in this
> file grants rights to this repository's own code.

## 1. Runtime dependencies (installed by the user; not redistributed)

| Package | Version used | Licence (from package metadata) | Notes |
|---|---|---|---|
| PyTorch (`torch`) | 2.15.0 | metadata: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib (project plus bundled components) | — |
| Hugging Face `transformers` | 4.57.6 | Apache-2.0 | Loads X-CLIP and its processor |
| `tokenizers`, `safetensors` | 0.22.2, 0.8.0 | not stated in the installed metadata; the upstream projects publish Apache-2.0 (not re-verified here) | Installed with `transformers` |
| NumPy | 1.26.4 | BSD-3-Clause (project), with bundled components under MIT, Zlib, Apache-2.0 and others (metadata) | — |
| SciPy | 1.17.1 | BSD-3-Clause-style SciPy licence (licence text in the metadata) | Statistics |
| Matplotlib | 3.11.2 | Matplotlib licence (PSF-based) | Figures |
| OpenCV (`opencv-python`) | 4.10.0.84 | Apache-2.0 (OpenCV ≥ 4.5) | Decoding, blur |
| Pillow | 11.3.0 | MIT-CMU (HPND) | Resizing (via the transformers image processor) |
| pytest | 9.2.0 | MIT | Tests |

The licences were read from the installed packages' metadata on
2026-11-02. Users obtain these packages under their own licences.

## 2. Pretrained model (downloaded at run time; not redistributed)

- **Model:** X-CLIP B/32, `microsoft/xclip-base-patch32`, revision
  `a2e27a78a2b5d802e894b8a1ef14f3a8ce490963`.
- **Licence:** the Hugging Face model card states **MIT** (checked
  2026-11-02).
- **Citation:** B. Ni, H. Peng, M. Chen, S. Zhang, G. Meng, J. Fu,
  S. Xiang, H. Ling. "Expanding Language-Image Pretrained Models for
  General Video Recognition." ECCV 2022. arXiv:2208.02816.

## 3. Datasets (not redistributed)

| Dataset | Citation | Use here | Terms |
|---|---|---|---|
| UCF101 | K. Soomro, A. R. Zamir, M. Shah. "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild." CRCV-TR-12-01, 2012. arXiv:1212.0402 | Class names (prompts), splits, evaluation | Videos originate from YouTube. **The terms of use were not verified** by the preparer; users must obtain the data from the provider and follow its terms. |
| HMDB51 | H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, T. Serre. "HMDB: A Large Video Database for Human Motion Recognition." ICCV 2011 | Cross-dataset evaluation (mapped categories) | Clips come from films and web video. **The terms of use were not verified** by the preparer. |

**What this repository publishes about the datasets:**
- class names and aggregate statistics;
- the category mapping, below.

**What it does not publish:**
- video frames;
- per-clip file names or per-clip outputs.

## 4. Category mapping

- **Source:** the 12 UCF101↔HMDB51 categories used in Phase 2B are
  transcribed from TA3N (the documents cite "Table 13"; that table number
  is not yet verified, see `CITATIONS.md`): M.-H. Chen, Z. Kira, G. AlRegib,
  J. Yoo, R. Chen, J. Zheng. "Temporal Attentive Alignment for Large-Scale
  Video Domain Adaptation." ICCV 2019. arXiv:1907.12743.
- **Where it is documented:**
  `research_analysis/phase2a/DATASET_MAPPING.md`.
- **Status:** the mapping is restated as factual correspondences. No code
  from TA3N is included.

## 5. Other works cited in the reports

These are cited for context only. No code or data from them is included.
The bibliographic verification status of every cited work is in
`CITATIONS.md`.

## 6. Provenance of this repository's code

- **Creator and maintainer:** Syed Omer Ali.
- **Origin of the code:** `AI_ASSISTANCE.md`.
- **Files omitted from this reduced release:** files produced in the
  maintainer's ChatGPT conversations, and files whose redistribution
  rights are not yet verified.
- **No licence** is granted until redistribution rights are verified and
  the maintainer adds one.
