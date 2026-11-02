# Provenance and AI-Assisted Development Disclosure

- **Creator and maintainer:** Syed Omer Ali.
- **How to read this document:**
  - Statements marked *(record)* come from file metadata, the project's
    decision log, or the assistant's session records.
  - Statements marked *(maintainer)* are the maintainer's own
    confirmations.
  - Open items are marked **[OPEN]**.

## 1. Tools used

| Tool | Where it was used | Basis |
|---|---|---|
| **ChatGPT (OpenAI)** | The initial EdgeHAR MVP and Guardian v1 files were produced in the maintainer's own ChatGPT conversations and downloaded on 2026-10-30. | Download metadata records `https://chatgpt.com/` as the source *(record)*; the conversations were the maintainer's own *(maintainer)* |
| **Claude Code (Anthropic)** | Analysis code, protocols, reports and figures under `research_analysis/`; the release documents; `tools/` | Session records, 2026-11-01 to 2026-11-02, under per-phase approvals by the maintainer *(record)* |

## 2. Files and their origin

| Files (in this release) | Origin |
|---|---|
| *(pipeline modules not included in this reduced release)* | See the full provenance record (maintained privately). They were omitted while redistribution rights are verified. |
| `research_analysis/**` (code, protocols, reports, aggregate results, figures), top-level documents, `tools/` | Written and executed by Claude Code in the recorded sessions *(record)* |
| `results/guardian_v3_official/*`, `research/RESEARCH_LOG.md` | Produced in the maintainer's development workflow before the Claude Code sessions *(record: timestamps)* |

## 3. Third-party material

- **No notices found.** No copyright notices, licence headers or source
  attributions were found in any of these files, by local inspection
  *(record)*.
- **Not compared against public code** *(record)*.
- **[OPEN] Third-party input to the ChatGPT conversations.** The
  maintainer is **unsure** whether third-party source code was pasted or
  uploaded into the conversations that produced the MVP and Guardian v1
  files *(maintainer)*. Third-party content in those files is therefore
  **not ruled out**.

## 4. The maintainer's role

| Statement | Status |
|---|---|
| Created and maintains EdgeHAR-X; set the research questions, constraints and integrity rules | *(maintainer; decision log)* |
| Approved, rejected or stopped every research phase | *(record: decision log)* |
| Reviewed or tested the project | **Some parts, not everything** *(maintainer)* |
| Edited assistant-drafted protocols or reports | **[OPEN]** |

**Not claimed:** exclusive authorship of every file, verified
redistribution rights for every file, or comprehensive personal
verification.

## 5. Verification status

- **Internal checks only.** All checks were run within the same
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
