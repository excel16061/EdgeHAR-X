# Provenance and AI-Assisted Development Disclosure

- **Creator and maintainer:** Syed Omer Ali.
- **How to read this document:**
  - *(record)*: verifiable from file metadata, download metadata, the
    project's decision log, or the assistant's session records.
  - *(maintainer attestation)*: a statement by the maintainer. It is
    **not** independent proof.
  - **[OPEN]**: not yet confirmed.

## 1. Development history

| Tool | Where it was used | Basis |
|---|---|---|
| **ChatGPT (OpenAI)** | The initial EdgeHAR MVP and the Guardian v1 benchmark files were produced in the maintainer's own ChatGPT conversations and downloaded on 2026-10-30. They were later edited locally. Further demo bundles (SecureSense, Guardian Live v2) came from the maintainer's own AI conversations. **None of these files is included in this reduced release.** | Download metadata `https://chatgpt.com/` *(record)*; own conversations *(maintainer attestation)* |
| **AI assistance plus the maintainer's own code** | The Guardian v3 training and evaluation scripts, created 2026-10-31. **Not included in this release.** The frozen Guardian v3 parameters and results they produced are included as data. | *(maintainer attestation)*; no download trace *(record)* |
| **Claude Code (Anthropic)** | All analysis code, protocols, reports and figures under `research_analysis/`; the release documents; `tools/`. This includes **all six source files in this release.** | Session records, 2026-11-01 to 2026-11-02, under per-phase approvals by the maintainer *(record)* |

## 2. Files in this release and their origin

| Files | Origin |
|---|---|
| `research_analysis/**` (code, protocols, reports, aggregate results, figures), top-level documents, `tools/` | Written and executed by Claude Code in the recorded sessions *(record)* |
| `results/guardian_v3_official/*`, `results/figures/guardian_risk_coverage.png`, `research/RESEARCH_LOG.md` | Produced in the maintainer's development workflow before the Claude Code sessions *(record: timestamps)* |
| `LICENSE` | Standard MIT text, Copyright (c) 2026 Syed Omer Ali |

## 3. Third-party material

- **No notices found.** No copyright notices, licence headers or source
  attributions were found in any source file, by local inspection
  *(record)*.
- **Targeted public-code comparison** (2026-11-02; short fragments only;
  validated search index) *(record)*:
  - **Released statistics:** the DeLong and Obuchowski code follows the
    cited papers' formulas and is structurally different from the common
    public implementation. It is a standard algorithm, implemented
    independently.
  - **ChatGPT-derived and Guardian v3 files:** no public code matching
    their distinctive fragments was found.
  - **Limits:** the comparison was **not exhaustive**.
- **Third-party input to the ChatGPT conversations:** the maintainer
  attests that **no third-party source code was knowingly supplied**. The
  inputs were the maintainer's instructions and project material. The
  local edits were the maintainer's own or AI-generated, and were not
  knowingly copied from third-party repositories *(maintainer
  attestation)*. AI output can in principle resemble existing code without
  the user's knowledge; this is not ruled out by the checks above.

## 4. The maintainer's role

| Statement | Status |
|---|---|
| Created and maintains EdgeHAR-X; set the research questions, constraints and integrity rules | *(maintainer attestation; decision log)* |
| Approved, rejected or stopped every research phase | *(record: decision log)* |
| Reviewed or tested the project | **Some parts, not everything** *(maintainer attestation)* |
| Edited assistant-drafted protocols or reports | **[OPEN]** |

**Not claimed:**
- exclusive authorship of every file;
- copyright protection for AI-generated material;
- comprehensive personal verification.

**The MIT licence** applies only to the extent that the maintainer holds
rights (see `README.md`).

## 5. Verification status

- **Internal checks only.** All research checks were run within the same
  AI-assisted workflow:
  - hash checks of frozen records;
  - unit tests;
  - parity tests;
  - an automated re-check of every reported number against the frozen
    result files.
- **No external reviewer** has examined this work.
- **Negative results are reported as they are:**
  - H1–H3 not rejected;
  - the original P3 gradient check FAILED and remains failed;
  - the Phase 1 audit verdict was REVISE.
