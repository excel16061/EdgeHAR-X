# Citations: verification record

- **Scope:** every research work cited in this repository was checked on
  2026-11-02 against a bibliographic record (arXiv abstract page, Crossref
  DOI record, PubMed, or the official proceedings page).
- **What "verified" covers:** the bibliographic details (title, authors,
  venue, year). It does **not** mean the claim attributed to the work was
  re-checked against its full text, unless stated.

| # | Cited as (location) | Bibliographic record found | Status |
|---|---|---|---|
| 1 | X-CLIP, "Ni et al., ECCV 2022, arXiv:2208.02816" (`THIRD_PARTY_NOTICES.md`, `DATA.md`) | B. Ni, H. Peng, M. Chen, S. Zhang, G. Meng, J. Fu, S. Xiang, H. Ling. *Expanding Language-Image Pretrained Models for General Video Recognition.* arXiv:2208.02816; arXiv comment: "Accepted by ECCV2022, Oral" | ✅ verified (arXiv record) |
| 2 | UCF101, "Soomro, Zamir & Shah, CRCV-TR-12-01, 2012, arXiv:1212.0402" | K. Soomro, A. R. Zamir, M. Shah. *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild.* Report CRCV-TR-12-01. arXiv:1212.0402 | ✅ verified (arXiv record) |
| 3 | HMDB51, "Kuehne et al., ICCV 2011" | H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, T. Serre. *HMDB: A large video database for human motion recognition.* ICCV 2011, pp. 2556–2563. DOI 10.1109/ICCV.2011.6126543 | ✅ verified (Crossref) |
| 4 | TA3N, "Chen et al., ICCV 2019, arXiv:1907.12743", **"Table 13"** as the source of the 12-category mapping (`phase2a/DATASET_MAPPING.md`, `THIRD_PARTY_NOTICES.md`) | M.-H. Chen, Z. Kira, G. AlRegib, J. Yoo, R. Chen, J. Zheng. *Temporal Attentive Alignment for Large-Scale Video Domain Adaptation.* arXiv:1907.12743; arXiv comment: "ICCV 2019 (Oral) camera-ready + supplementary" | ✅ bibliographic details verified. ✅ **"Table 13" verified** (2026-11-02) against the ar5iv rendering of arXiv:1907.12743 (main paper plus supplementary): Table 13, "The lists of all collected categories in UCF and HMDB" (§7.4.1). The project mapping matches **12/12 categories and 15/15 UCF101 classes**. The workshop version (arXiv:1905.10861) corroborates the 12-class count. |
| 5 | Schiappa et al., CVPR 2023 (`phase2a/RESEARCH_HYPOTHESES.md`, `phase2a/temporal_monitor_design.md`, `phase2b/CONFIRMATORY_RESULTS.md`, `phase2b/FAILURE_ANALYSIS.md`) | M. C. Schiappa, N. Biyani, P. Kamtam, S. Vyas, H. Palangi, V. Vineet, Y. Rawat. *Large-scale Robustness Analysis of Video Action Recognition Models.* arXiv:2207.01398; arXiv comment: "Accepted in 2023 Conference on Computer Vision and Pattern Recognition (CVPR)" | ✅ bibliographic details verified. ⚠️ The attributed finding (temporal robustness of Kinetics-type models) was **not re-checked** against the full text. |
| 6 | "Carlini & Wagner 2017" (`phase2a/RESEARCH_HYPOTHESES.md`, `phase2a/temporal_monitor_design.md`) | **Most likely:** N. Carlini, D. Wagner. *Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods.* Proc. 10th ACM Workshop on Artificial Intelligence and Security (AISec '17), pp. 3–14. DOI 10.1145/3128572.3140444; arXiv:1705.07263 | ⚠️ **Ambiguous in the source.** Carlini & Wagner have more than one 2017 paper. This referent fits the context ("detectors/monitors can be bypassed by adaptive attacks"), but the frozen documents do not say which paper is meant. |
| 7 | "Li et al. NDSS 2019" (`phase2a/RESEARCH_HYPOTHESES.md`) | **Most likely:** S. Li, A. Neupane, S. Paul, C. Song, S. V. Krishnamurthy, A. K. Roy-Chowdhury, A. Swami. *Stealthy Adversarial Perturbations Against Real-Time Video Classification Systems.* NDSS 2019 (official NDSS paper page) | ⚠️ **Ambiguous in the source.** It is the only "Li et al." NDSS 2019 paper found on video attacks. It concerns attacks on video *classifiers*, not specifically on monitors, so it supports the cited sentence only partly. |
| 8 | "Liu et al. 2020" energy score (`phase2a/baseline_metrics.py`) | W. Liu, X. Wang, J. D. Owens, Y. Li. *Energy-based Out-of-distribution Detection.* NeurIPS 2020 (Advances in Neural Information Processing Systems 33). arXiv:2010.03759 | ✅ verified (NeurIPS proceedings page) |
| 9 | "Traub et al., NeurIPS 2024" AUGRC (`phase2a/baseline_metrics.py`) | J. Traub, T. J. Bungert, C. T. Lüth, M. Baumgartner, K. H. Maier-Hein, L. Maier-Hein, P. F. Jäger. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems.* NeurIPS 2024. arXiv:2407.01032. The abstract proposes AUGRC. | ✅ bibliographic details verified. ⚠️ The code comment itself says the implementation is "our reading" and should be checked against the reference implementation; that has **not** been done. |
| 10 | "Obuchowski 1997, Biometrics 53:567–578" (`phase1_audit/statistical_audit.py`, `AUDIT_REPORT.md`) | N. A. Obuchowski. *Nonparametric analysis of clustered ROC curve data.* Biometrics 53(2):567–578, 1997. PMID 9192452 | ✅ verified (PubMed) |
| 11 | "Sun & Xu 2014" DeLong formulation (`phase1_audit/test_statistical_audit.py`) | X. Sun, W. Xu. *Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves.* IEEE Signal Processing Letters 21(11):1389–1393, 2014. DOI 10.1109/LSP.2014.2337313 | ✅ verified (Crossref) |

**Methods named without a citation:**
- DeLong's test;
- Holm's correction;
- TOST equivalence testing;
- the cluster bootstrap and jackknife.

These are standard methods named in the reports and code; no specific
bibliographic claim is made for them here.

**Sources consulted:**
- arXiv abstract pages: 2208.02816, 1212.0402, 1907.12743, 2207.01398,
  2010.03759, 2407.01032, 1705.07263;
- ar5iv renderings of arXiv:1907.12743 (Table 13) and 1905.10861
  (accessed 2026-11-02);
- Crossref records: 10.1109/ICCV.2011.6126543, 10.1109/LSP.2014.2337313,
  10.1145/3128572.3140444;
- PubMed 9192452;
- NeurIPS proceedings pages (2020; 2024);
- the NDSS 2019 paper page.
