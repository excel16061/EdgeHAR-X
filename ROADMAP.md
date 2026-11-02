# Roadmap (v0.1, 2026-11-02)

**Legend:**
- ✅ completed;
- ⚠️ completed with a negative or failed outcome (reported as-is);
- 🔒 designed but not approved to run;
- ⬜ not started.

## Completed

| Item | Outcome |
|---|---|
| EdgeHAR baseline: uniform vs adaptive sampling (UCF101, 505 clips) | ✅ Adaptive 66.5% vs uniform 65.4% top-1; not significance-tested |
| Guardian v1–v3; frozen v3; UCF101 group-disjoint evaluation | ⚠️ No significant risk reduction vs MSP |
| Phase 1: HMDB51 6-class validation (180 clips) | ✅ / ⚠️ Pooled gain mostly class composition |
| Phase 1 independent audit | ⚠️ Verdict REVISE |
| Phase 2A: benchmark protocol, label mapping, preregistration, exporter | ✅ |
| Phase 2B: pre-registered confirmatory study | ✅ Validity 6/6; ⚠️ H1–H3 not rejected |
| Phase 3 surrogate parity (P1, P2, P6) | ✅ PASS |
| Phase 3 original gradient check P3 | ⚠️ **FAILED** (stands) |
| P3 diagnostic | ✅ Numerical artefact of the check; P3 still FAILED |
| P3-v2 prospective gradient validation (documented deviation) | ✅ PASS |
| CPU feasibility timing | ✅ CPU backward about 210× faster than Apple MPS |

## Designed, not approved

- 🔒 **Phase 3 development pilot**, on development data only.
- 🔒 **Phase 3 confirmatory robustness evaluation** on a held-out UCF101
  set. This requires a pilot, a new preregistration and the author's
  approval.

## Future work

| Item | Why |
|---|---|
| ⬜ Independent re-run of the Phase 2B analysis by a second person | All verification so far is internal |
| ⬜ Multi-label mapping sensitivity | Part of the measured "error" is label mapping |
| ⬜ Short-clip failure analysis (< 64 frames; within-class AUROC ≈ 0.5) | Exploratory Phase 2B observation |
| ⬜ Further backbones and datasets | Single-backbone, mapped-category scope |
| ⬜ Batched multi-view inference; on-device latency and energy | Only laptop measurements exist |

## Not planned

- Physical-world experiments.
- Deployment against real systems.
- Surveillance or biometric use.
