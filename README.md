# EdgeHAR-X / TemporalGuard

**Temporal reliability monitoring for zero-shot video action recognition**

- **Creator and maintainer:** Syed Omer Ali.
- **Status:** v0.1, research in progress.
- **Release scope (reduced release).**
  - **Included:** the research analysis code (monitor definitions, Phase 2B
    statistics and figure code, Phase 1 audit), all aggregate results,
    figures, protocols and reports.
  - **Excluded:** the **complete video inference pipeline** (frame decoding
    and sampling, corruption and temporal views, the X-CLIP prediction
    exporter) and its tests. Inference therefore cannot be re-run from video
    with this repository.
- **Not reviewed:** this work has not been peer-reviewed or independently
  replicated.
- **How it was built:** development used AI coding assistants. See
  [Provenance and AI assistance](#provenance-and-ai-assistance).

## Research motivation

Open-vocabulary video models can classify actions they were never trained
on, by comparing a clip against text prompts. When such a model is wrong,
it gives no reliable warning: its softmax scores are relative to the chosen
label set, not calibrated probabilities.

This project asks whether a **label-free reliability monitor** can flag the
model's own errors at deployment time, by re-querying the same frozen model
on altered views of a clip. It also asks whether that signal **survives a
shift to a different dataset**.

## Setting: zero-shot video action recognition

- **Model:** a frozen, pretrained X-CLIP model
  (`microsoft/xclip-base-patch32`).
- **Labels:** clips are classified against the 101 UCF101 action names
  ("a video of a person {action}").
- **Frame sampling:** 8 frames per clip, chosen with a motion-adaptive
  heuristic.
- **Datasets:**
  - **UCF101:** calibration, and a group-disjoint baseline evaluation;
  - **HMDB51:** shifted-data evaluation, on mapped categories only.
    **This is not a 51-class HMDB51 evaluation.**

## Temporal reliability monitoring

| Monitor | Forward passes | Signal |
|---|---|---|
| MSP | 1 | Confidence on the clean clip |
| Guardian-O / Guardian-Full | 2 / 5 | Agreement under corruption probes (blur, dark, occlusion, frame drop), combined by a frozen logistic model |
| **TemporalGuard** (TG0, TG-min) | 6 | Consistency of the predicted class across a uniform view and four temporal segments |

All accept/abstain thresholds are fixed on UCF101 calibration data (70%
target coverage). No HMDB51 data were used for training or calibration.

## Experimental methodology

- **Pre-registration.** The confirmatory study (Phase 2B) was frozen by
  hash before any confirmatory inference: hypotheses, decision rule, data
  and analysis.
- **Within-class evaluation.** The primary metric is *within-class* AUROC:
  can a monitor separate correct from incorrect predictions *of the same
  class*? Pooled metrics are reported separately, with a class-composition
  decomposition, so that "easy class" effects are not mistaken for
  reliability.
- **Source-group-aware statistics.** Clips cut from the same source video
  are resampled together:
  - film-group cluster bootstrap and jackknife;
  - Holm correction;
  - hierarchical intervals;
  - equivalence tests;
  - negative controls.

Details: [METHODOLOGY.md](METHODOLOGY.md).

## Results

Full tables with all qualifications are in [RESULTS.md](RESULTS.md).

### Completed experiments (Phases 1–2)

| Study | Data | Outcome |
|---|---|---|
| Guardian v3 baseline | UCF101, 505 group-disjoint clips | No significant risk reduction vs MSP |
| Phase 1: cross-dataset check (exploratory) | HMDB51, 6 mapped classes, 180 clips | Pooled gain mostly explained by class composition; audit verdict **REVISE** |
| Phase 2B: pre-registered confirmatory study | HMDB51, 797 + 177 clips in mapped categories | All 6 validity checks passed. **H1–H3 not rejected.** H2/H3 inconclusive (neither an effect nor equivalence shown). |

**Descriptive findings:**
- **Pooled AURC:** every monitor ranked errors far better than random on
  the 12 categories, but about half of the multi-view gain is explained by
  class composition.
- **TG-min:** the largest within-class gain (+0.075), but it does not pass
  the multiplicity or hierarchical criteria.
- **Disruptions:** benign temporal disruptions did not measurably change
  accuracy (all within ±1 pp).

### Engineering validation (Phase 3 preparation); not a robustness result

These checks test whether a differentiable copy of the frozen pipeline is
faithful enough to support a later robustness study:

- surrogate forward parity: **PASS**;
- **original gradient check P3: FAILED** (the failure stands);
- a separate, prospectively frozen check, **P3-v2: PASS**. It does not
  overturn the P3 failure.

## Limitations

- **Narrow scope.** One backbone, zero-shot, 8-frame sampling. HMDB51
  results cover mapped categories only (6, or 12 + 6), never all 51.
- **Mapping.** One-to-one label mapping turns some plausible predictions
  into "errors".
- **Generalisation and power.** Class-level generalisation is weak:
  hierarchical intervals often include the null. The study was powered for
  within-class differences of about 0.04–0.05.
- **Split reuse.** The UCF101 official test list was inspected in early
  experiments, so the Guardian v3 evaluation uses group-disjoint
  training-partition clips.
- **Verification.** It is internal only, with no external review.

## Future research

- **Robustness under adaptive attack.** Test whether monitor signals
  survive inputs optimized against them. **Not run, not approved, and no
  such code is included.** It requires a pilot, a new preregistration and
  the maintainer's approval.
- **Data and mapping.** Multi-label mapping sensitivity; short-clip failure
  analysis.
- **Models.** Further backbones.
- **Re-runs and cost.** Independent re-runs; on-device latency and energy.

See [ROADMAP.md](ROADMAP.md).

## Repository contents

| Path | Content |
|---|---|
| `RESULTS.md`, `METHODOLOGY.md`, `REPRODUCIBILITY.md`, `TEST_STATUS.md`, `ROADMAP.md`, `DATA.md`, `CITATIONS.md` | Results, methods, reproduction guide, test status, roadmap, data access, citation verification |
| `research_analysis/phase2a/` | Benchmark protocol, preregistration, label mapping, monitor definitions |
| `research_analysis/phase2b/` | Confirmatory analysis code, aggregate results, figures, reports |
| `research_analysis/cross_dataset_hmdb51/` | Phase 1 audit code, aggregate outputs, reports |
| `research_analysis/phase3/` | Phase 3 parity and gradient-validation reports and an aggregate summary (no code) |
| `results/guardian_v3_official/` | Frozen Guardian v3 parameters and the UCF101 evaluation summary |
| `tools/make_ucf101_classind.py` | Regenerates the UCF101 class index from a local UCF101 download (nothing copied) |

**This is the reduced release.**
- **Included:** research analysis and results.
- **Excluded:** the complete video inference pipeline. That means the
  pipeline modules whose redistribution rights are still being verified
  (`edgehar.py`, `guardian_benchmark.py`, two Guardian v3 scripts and their
  tests), and the code that depends on them: the prediction exporter, the
  manifest builder, the Phase 1 validation script, and their tests.
- **Results are unaffected.** The reported results come from the frozen
  records; the analysis code, results, figures and reports are complete.
  See `REPRODUCIBILITY.md`.

## Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider research_analysis/cross_dataset_hmdb51/phase1_audit/test_statistical_audit.py
```

- **Tests:** some pass only partly without the datasets or the unpublished
  per-clip records. Every blocked test and its missing input is listed in
  [TEST_STATUS.md](TEST_STATUS.md).
- **What this repository can and cannot reproduce:**
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Provenance and AI assistance

EdgeHAR-X was created and is maintained by Syed Omer Ali. Development used
AI coding assistants:
- **ChatGPT (OpenAI):** the initial EdgeHAR MVP and Guardian v1 files
  were produced in the maintainer's own ChatGPT conversations, and were
  later modified locally.
- **Claude Code (Anthropic):** for the analysis code, protocols and
  documentation under `research_analysis/`, and for this release's
  documents.

Whether third-party code entered the earlier ChatGPT conversations has not
been determined. Details and verification status:
[AI_ASSISTANCE.md](AI_ASSISTANCE.md),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Licence

**No licence has been granted.** All rights are reserved until a `LICENSE`
file is added.
