"""Phase 2B confirmatory analysis (implements PREREGISTRATION.md v1.0 §4-§8).

Run ONCE on the confirmatory exports after all exports finish:

    PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python research_analysis/phase2b/phase2b_analysis.py \
        --freeze research_analysis/phase2b/FREEZE.json \
        --output-dir research_analysis/phase2b/results

Operationalizations fixed before any confirmatory inference (also logged in
DEVIATIONS.md as clarifications, not deviations):
  * Decision rule (H1-H3, and S3 family): reject iff 95% film-group cluster
    bootstrap percentile CI excludes the null AND Holm-adjusted cluster
    jackknife p < 0.05 (Holm over the family).
  * Class-composition share = mean(ΔAURC under within-class label
    permutation) / observed ΔAURC, reported only if the 95% cluster
    bootstrap CI of observed ΔAURC excludes 0.
  * S7 flip AUROC: among clips correct on clean_adaptive, positive = still
    correct on the disrupted view, negative = flipped to wrong; score = the
    clean-view monitor (higher = more confident).
  * S8 warm-up: the first 3 clips in manifest order are excluded from timing.
    End-to-end per clip = decode_ms_shared + Σ over the monitor's views of
    (transform + sampling + preprocess + model) ms.
  * S9: a monitor whose UCF101-calibration scores tie at the threshold
    boundary gets threshold N/A.
  * Probabilities are softmax(float64) of the exported float32 logits and are
    written to a consolidated file.

--test-mode (synthetic fixtures only) skips the freeze-hash checks and allows
small B; it is refused when the export directory contains real-run metadata.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P2A = ROOT / "research_analysis/phase2a"
sys.path.insert(0, str(P2A))
import baseline_metrics as M  # noqa: E402

PREREG = {"B": 10000, "seed_cluster": 20270, "seed_hier": 20271, "seed_perm": 20272,
          "P": 10000, "margin": 0.03, "alpha": 0.05}
LOGIT_BASELINES = ["neg_entropy", "margin", "energy", "max_logit"]
TG_VARIANTS = ["tg_min", "tg_agreement_then_msp", "tg_neg_js", "tg_sampling_pred_prob", "tg_seg2"]
ALL_MONITORS = (["msp"] + LOGIT_BASELINES + ["guardian_o", "guardian_full", "tg0"] + TG_VARIANTS
                + ["corruption_tta_pred_prob", "corruption_agreement_then_msp"])
CONTROLS = ["constant", "random", "within_class_shuffled_msp", "class_oracle"]
MONITOR_VIEWS = {
    "msp": ["clean_adaptive"], "neg_entropy": ["clean_adaptive"], "margin": ["clean_adaptive"],
    "energy": ["clean_adaptive"], "max_logit": ["clean_adaptive"],
    "guardian_o": ["clean_adaptive", "occlusion3"],
    "guardian_full": ["clean_adaptive", "blur3", "dark3", "occlusion3", "frame_drop3"],
    "corruption_tta_pred_prob": ["clean_adaptive", "blur3", "dark3", "occlusion3", "frame_drop3"],
    "corruption_agreement_then_msp": ["clean_adaptive", "blur3", "dark3", "occlusion3", "frame_drop3"],
    "tg0": ["clean_adaptive", "clean_uniform", "seg4_0", "seg4_1", "seg4_2", "seg4_3"],
    "tg_min": ["clean_adaptive", "clean_uniform", "seg4_0", "seg4_1", "seg4_2", "seg4_3"],
    "tg_agreement_then_msp": ["clean_adaptive", "clean_uniform", "seg4_0", "seg4_1", "seg4_2", "seg4_3"],
    "tg_neg_js": ["clean_adaptive", "clean_uniform", "seg4_0", "seg4_1", "seg4_2", "seg4_3"],
    "tg_sampling_pred_prob": ["clean_adaptive", "clean_uniform"],
    "tg_seg2": ["clean_adaptive", "seg2_0", "seg2_1"],
}
DISRUPTIONS = ["frame_drop1", "frame_drop2", "frame_drop3", "delay1", "delay2", "delay3",
               "reverse", "segment_shuffle", "clean_uniform"]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================================================================ loading

def load_manifest_rows(path):
    return list(csv.DictReader(Path(path).open(newline="")))


def load_exports(export_dir, clip_ids, expected_views):
    """Returns dict clip_id -> {'logits': {view: z}, 'records': [...]}, plus
    list of missing clip ids. Refuses smoke clips."""
    out, missing = {}, []
    for cid in clip_ids:
        if cid.startswith("smoke_"):
            raise ValueError("smoke clip id in confirmatory analysis")
        npz, jsl = Path(export_dir) / f"{cid}.npz", Path(export_dir) / f"{cid}.jsonl"
        if not (npz.exists() and jsl.exists()):
            missing.append(cid)
            continue
        d = np.load(npz)
        views = [str(v) for v in d["view_ids"]]
        if views != expected_views:
            raise ValueError(f"{cid}: view list differs from frozen config")
        recs = [json.loads(l) for l in jsl.open()]
        out[cid] = {"logits": {v: d["logits"][i] for i, v in enumerate(views)}, "records": recs}
    extra = [p.stem for p in Path(export_dir).glob("*.npz") if p.stem.startswith("smoke_")]
    if extra:
        raise ValueError(f"smoke clips present in confirmatory export dir: {extra}")
    return out, missing


def build_table(rows, exports, labels, frozen):
    """Per-clip table for clips with exports."""
    T = []
    for r in rows:
        e = exports.get(r["clip_id"])
        if e is None:
            continue
        m = M.monitor_scores(e["logits"], frozen)
        pred = labels[m["_yhat"]]
        mapped = r["ucf101_mapped_labels"].split("|")
        rec = {"clip_id": r["clip_id"], "cls": r["hmdb_class"], "group": r["film_group"],
               "role": r["role"], "pred": pred,
               "correct_mapped": pred in mapped, "correct_strict": pred == r["ucf101_strict_label"],
               **{k: m[k] for k in ALL_MONITORS}}
        for v in DISRUPTIONS:
            z = e["logits"][v]
            p = M.softmax(z)
            rec[f"pred_{v}"] = labels[int(np.argmax(p))]
            rec[f"correct_{v}"] = rec[f"pred_{v}"] in mapped
            rec[f"msp_{v}"] = float(p.max())
        recs = {x["view_id"]: x for x in e["records"]}
        rec["decode_ms"] = recs["clean_adaptive"]["decode_ms_shared"]
        for mon, views in MONITOR_VIEWS.items():
            rec[f"model_ms_{mon}"] = sum(recs[v]["model_ms"] for v in views)
            rec[f"e2e_ms_{mon}"] = rec["decode_ms"] + sum(
                recs[v]["transform_ms"] + recs[v]["sampling_ms"] + recs[v]["preprocess_ms"]
                + recs[v]["model_ms"] for v in views)
        rec["frames_repeated_any"] = any(x.get("frames_repeated", False) for x in e["records"])
        rec["num_decoded_frames"] = recs["clean_adaptive"]["num_decoded_frames"]
        T.append(rec)
    return T


# ============================================================ statistics

def arr(T, k, dtype=float):
    return np.array([t[k] for t in T], dtype=dtype)


def safe_ci(v):
    v = np.asarray(v, float)
    return M.percentile_ci(v) if np.isfinite(v).any() else None


def not_estimable(reason, null):
    return {"estimate": None, "null": null, "not_estimable": True, "reason": reason,
            "ci95_cluster": None, "ci95_hierarchical": None,
            "jackknife": {"se": None, "p": None}}


def one_sample(y, cls, grp, s, cfg):
    theta = M.auroc(y, s, cls)
    if not np.isfinite(theta):
        return not_estimable("no within-class (correct, incorrect) pairs", 0.5)
    W = M.cluster_bootstrap_weights(grp, cfg["B"], cfg["seed_cluster"])
    v = np.array([M.auroc(y, s, cls, w) for w in W])
    J = M.jackknife_weights(grp)
    vj = np.array([M.auroc(y, s, cls, w) for w in J])
    Wh = M.hierarchical_bootstrap_weights(cls, grp, cfg["B"], cfg["seed_hier"])
    vh = np.array([M.auroc(y, s, cls, w) for w in Wh])
    jk = M.jackknife_t(theta, vj, null=0.5)
    return {"estimate": theta, "null": 0.5, "ci95_cluster": M.percentile_ci(v),
            "p_cluster_bootstrap": M.bootstrap_p(v, 0.5), "jackknife": jk,
            "ci95_hierarchical": M.percentile_ci(vh), "p_hierarchical": M.bootstrap_p(vh, 0.5),
            "n_groups": int(len(set(grp)))}


def two_sample(y, cls, grp, sa, sb, cfg):
    if not np.isfinite(M.auroc(y, sa, cls)):
        return not_estimable("no within-class (correct, incorrect) pairs", 0.0)
    r = M.compare(y, cls, grp, sa, sb, B=cfg["B"], seed=cfg["seed_cluster"],
                  margin=cfg["margin"], hierarchical=True)
    r["null"] = 0.0
    r["seed_hierarchical"] = cfg["seed_cluster"] + 1
    return r


def fix_degenerate_jackknife(r):
    """baseline_metrics.jackknife_t returns p=0 when se=0 (degenerate); a
    zero-variance estimate equal to the null must give p=1."""
    jk = r["jackknife"]
    if jk["se"] == 0:
        jk["p"] = 1.0 if r["estimate"] == r["null"] else 0.0
        jk["degenerate_se_zero"] = True
    return r


def decide(results, cfg, key_est="estimate"):
    """Holm over jackknife p; reject iff cluster CI excludes null and Holm p < alpha."""
    est = {k: v for k, v in results.items() if not v.get("not_estimable")}
    for k, v in results.items():
        if v.get("not_estimable"):
            v.update({"holm_jackknife_p": None, "reject_null": False,
                      "generalizes_across_classes": False})
    for v in est.values():
        fix_degenerate_jackknife(v)
    p = {k: v["jackknife"]["p"] for k, v in est.items()}
    adj = M.holm(p) if p else {}
    for k, v in est.items():
        lo, hi = v["ci95_cluster"]
        null = v["null"]
        v["holm_jackknife_p"] = adj[k]
        v["reject_null"] = bool((lo > null or hi < null) and adj[k] < cfg["alpha"])
        lo_h, hi_h = v["ci95_hierarchical"]
        v["generalizes_across_classes"] = bool(v["reject_null"] and (lo_h > null or hi_h < null))
    return results


def primary_family(T, correct_key, cfg):
    y = arr(T, correct_key, bool)
    cls = arr(T, "cls", object).astype(str)
    grp = arr(T, "group", object).astype(str)
    res = {"H1_msp_vs_chance": one_sample(y, cls, grp, arr(T, "msp"), cfg),
           "H2_guardian_full_minus_msp": two_sample(y, cls, grp, arr(T, "guardian_full"), arr(T, "msp"), cfg),
           "H3_tg0_minus_msp": two_sample(y, cls, grp, arr(T, "tg0"), arr(T, "msp"), cfg)}
    for k in ("H2_guardian_full_minus_msp", "H3_tg0_minus_msp"):
        if not res[k].get("not_estimable"):
            res[k]["estimate"] = res[k]["delta_within_auroc"]
    return decide(res, cfg)


def pooled_metrics(y, s, grp, W):
    out = {"aurc": M.aurc(y, s), "eaurc": M.eaurc(y, s), "augrc": M.augrc(y, s),
           "aurc_minus_random": M.aurc(y, s) - float(np.mean(~y)),
           "pooled_auroc": M.auroc(y, s)}
    for c in (0.5, 0.7, 0.8):
        out[f"risk_at_{int(c*100)}"] = M.risk_at_coverage(y, s, c)
    bs = np.array([[M.aurc(y, s, w), M.auroc(y, s, w=w), M.aurc(y, s, w) - (w * ~y).sum() / w.sum()]
                   for w in W])
    out["aurc_ci95"] = M.percentile_ci(bs[:, 0])
    out["pooled_auroc_ci95"] = M.percentile_ci(bs[:, 1])
    out["aurc_minus_random_ci95"] = M.percentile_ci(bs[:, 2])
    return out, bs[:, 0]


def validity_checks(T, rows, missing, smoke_report, freeze_ok, protected_ok, summary):
    P = [t for t in T if t["role"] == "confirmatory_primary"]
    y = np.array([t["correct_mapped"] for t in P], bool)
    cls = np.array([t["cls"] for t in P])
    ok_classes = sum(1 for c in np.unique(cls)
                     if np.sum(y[cls == c]) >= 5 and np.sum(~y[cls == c]) >= 5)
    smoke_rows = [r for r in smoke_report.get("rows", []) if "edgehar_top1_equal" in r]
    checks = {
        "1_smoke_equivalence": bool(smoke_rows) and all(
            r["edgehar_top1_equal"] and r["edgehar_indices_equal"]
            and r["edgehar_score_absdiff"] <= 1e-4 for r in smoke_rows),
        "2_decode_failures_le_2pct": len(missing) <= 0.02 * len(rows),
        "3_no_leakage": bool(summary.get("leakage_ok", False)),
        "4_errors_ge_100_and_6_classes_with_5_5": bool((~y).sum() >= 100 and ok_classes >= 6),
        "5_frozen_hashes": bool(freeze_ok),
        "6_protected_artifacts_unchanged": bool(protected_ok),
    }
    return checks, {"missing_clips": len(missing), "manifest_clips": len(rows),
                    "primary_errors_mapped": int((~y).sum()), "primary_clips": int(len(y)),
                    "classes_with_5_correct_5_incorrect": int(ok_classes)}


# ================================================================== main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=Path)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--exports", type=Path)
    ap.add_argument("--ucf-exports", type=Path)
    ap.add_argument("--ucf-manifest", type=Path)
    ap.add_argument("--smoke-report", type=Path)
    ap.add_argument("--manifest-summary", type=Path)
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--bootstraps", type=int)
    ap.add_argument("--permutations", type=int)
    args = ap.parse_args(argv)
    t_start = time.time()
    cfg = dict(PREREG)

    freeze_ok, protected_ok, freeze = True, True, {}
    if args.test_mode:
        if args.bootstraps:
            cfg["B"] = args.bootstraps
        if args.permutations:
            cfg["P"] = args.permutations
        meta = Path(args.exports) / "run_metadata.json"
        if meta.exists() and not json.loads(meta.read_text())["config"].get("fake_model"):
            raise SystemExit("--test-mode refused on real exports")
    else:
        if args.bootstraps or args.permutations:
            raise SystemExit("B/P are fixed by the preregistration outside --test-mode")
        freeze = json.loads(args.freeze.read_text())
        mism = {k: v for k, v in freeze["sha256"].items() if sha256_file(ROOT / k) != v}
        freeze_ok = not mism
        if mism:
            print("FROZEN HASH MISMATCH:", mism)
        p2a = json.loads((P2A / "phase2a_manifest.json").read_text())
        protected_ok = all(sha256_file(ROOT / k) == v
                           for k, v in p2a["protected_artifacts_sha256"].items())
        for k in ("manifest", "exports", "ucf_exports", "ucf_manifest", "smoke_report",
                  "manifest_summary"):
            setattr(args, k, ROOT / freeze["paths"][k])

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output dir not empty: {out}")

    run_meta = json.loads((Path(args.exports) / "run_metadata.json").read_text())
    exp_cfg = run_meta["config"]
    if not args.test_mode and run_meta["config_sha256"] != freeze["config_sha256"]:
        freeze_ok = False
    views = [v["view_id"] for v in exp_cfg["views"]]
    labels = run_meta["class_order"]
    frozen = json.loads(M.FROZEN.read_text())["models"]

    rows = load_manifest_rows(args.manifest)
    exports, missing = load_exports(args.exports, [r["clip_id"] for r in rows], views)
    T = build_table(rows, exports, labels, frozen)
    smoke_report = json.loads(Path(args.smoke_report).read_text())
    summary = json.loads(Path(args.manifest_summary).read_text())
    nd = summary.get("near_duplicate_check") or {}
    flagged = {(a if a.startswith("new") else b).split(":")[1]
               for a, b, _ in nd.get("pairs_new_vs_phase1_excluded", [])}
    summary["leakage_ok"] = bool(
        summary["leakage_checks"]["no_phase1_clips"]
        and summary["leakage_checks"]["no_phase1_film_groups"]
        # every kept clip + every near-dup-excluded clip + all 180 Phase 1 clips were hashed
        and nd.get("videos_hashed") == len(rows) + len(flagged) + 180
        and not flagged & {r["clip_id"] for r in rows}        # no flagged near-duplicate remains
        and summary.get("ambiguous_local_matches", 1) == 0
        and all(r.get("available_locally") in ("True", True) for r in rows))
    checks, counts = validity_checks(T, rows, missing, smoke_report, freeze_ok, protected_ok, summary)
    confirmatory_valid = all(checks.values())
    print("VALIDITY:", json.dumps(checks), json.dumps(counts))

    # consolidated predictions (logits + probabilities)
    cids = [t["clip_id"] for t in T]
    Z = np.stack([np.stack([exports[c]["logits"][v] for v in views]) for c in cids])
    np.savez_compressed(out / "predictions_consolidated.npz", clip_ids=np.array(cids),
                        view_ids=np.array(views), logits=Z, probabilities=M.softmax(Z),
                        class_order=np.array(labels))

    res = {"preregistration": "PREREGISTRATION.md v1.0", "config": cfg,
           "validity_checks": checks, "counts": counts,
           "confirmatory_claims_permitted": confirmatory_valid, "test_mode": args.test_mode}

    sets = {"P": [t for t in T if t["role"] == "confirmatory_primary"],
            "S": [t for t in T if t["role"] == "confirmatory_secondary"]}
    P = sets["P"]
    y = arr(P, "correct_mapped", bool)
    cls = arr(P, "cls", object).astype(str)
    grp = arr(P, "group", object).astype(str)
    S = {k: arr(P, k) for k in ALL_MONITORS}
    S.update(M.control_scores(y, cls, S["msp"]))

    # ---------------- primary
    res["primary"] = primary_family(P, "correct_mapped", cfg)
    res["S5_secondary_set"] = primary_family(sets["S"], "correct_mapped", cfg) if sets["S"] else None
    res["S6_strict_correctness"] = primary_family(P, "correct_strict", cfg)

    # ---------------- S2-S4 comparisons
    def fam(pairs):
        out_ = {}
        for a, b in pairs:
            r = two_sample(y, cls, grp, S[a], S[b], cfg)
            if not r.get("not_estimable"):
                r["estimate"] = r["delta_within_auroc"]
            out_[f"{a}_minus_{b}"] = r
        return out_
    res["S2_tg0_vs_corruption_tta"] = decide(fam([("tg0", "corruption_tta_pred_prob")]), cfg)
    res["S3_logit_baselines_vs_msp"] = decide(fam([(m, "msp") for m in LOGIT_BASELINES]), cfg)
    res["S4_tg_variants_vs_msp"] = decide(fam([(m, "msp") for m in TG_VARIANTS]), cfg)
    other = fam([("guardian_o", "msp"), ("corruption_agreement_then_msp", "msp"),
                 ("tg0", "guardian_full")])
    res["exploratory_other_comparisons"] = decide(other, cfg)

    # ---------------- S1 pooled + decomposition
    W = M.cluster_bootstrap_weights(grp, cfg["B"], cfg["seed_cluster"])
    pooled, boot_aurc = {}, {}
    for m in ALL_MONITORS + CONTROLS:
        s = S[m]
        if m in ("constant", "class_oracle"):
            pooled[m] = {"aurc_tie_aware": M.aurc_tie_aware(y, s), "pooled_auroc": M.auroc(y, s)}
            continue
        pooled[m], boot_aurc[m] = pooled_metrics(y, s, grp, W)
    perm = {m: M.within_class_permutation_null(y, S[m], cls, cfg["P"], cfg["seed_perm"])
            for m in ALL_MONITORS}
    for m in ALL_MONITORS:
        pooled[m]["aurc_class_composition_only"] = float(perm[m].mean())
        pooled[m]["p_within_class_information"] = float(
            (1 + np.sum(perm[m] <= pooled[m]["aurc"])) / (1 + len(perm[m])))
        if m != "msp":
            d_obs = pooled[m]["aurc"] - pooled["msp"]["aurc"]
            d_boot = boot_aurc[m] - boot_aurc["msp"]
            ci = M.percentile_ci(d_boot)
            d_null = float((perm[m] - perm["msp"]).mean())
            pooled[m]["delta_aurc_vs_msp"] = d_obs
            pooled[m]["delta_aurc_vs_msp_ci95"] = ci
            pooled[m]["delta_aurc_expected_from_class_composition"] = d_null
            pooled[m]["class_composition_share"] = (
                d_null / d_obs if (ci[0] > 0 or ci[1] < 0) and d_obs != 0 else None)
    res["S1_pooled"] = pooled

    # ---------------- within-class per monitor (descriptive) + S10 + controls
    within = {}
    for m in ALL_MONITORS + CONTROLS:
        macro, k = M.macro_within_class_auroc(y, S[m], cls)
        within[m] = {"within_auroc": M.auroc(y, S[m], cls), "macro_within_auroc": macro,
                     "macro_classes": k}
    for m in ALL_MONITORS:
        v = np.array([M.auroc(y, S[m], cls, w) for w in W])
        within[m]["within_auroc_ci95_cluster"] = safe_ci(v)
    res["within_class"] = within

    per_class = []
    for c in sorted(set(cls)):
        k = cls == c
        row = {"hmdb_class": c, "clips": int(k.sum()), "film_groups": int(len(set(grp[k]))),
               "correct_mapped": int(y[k].sum()), "accuracy_mapped": float(y[k].mean()),
               "accuracy_strict": float(arr(P, "correct_strict", bool)[k].mean()),
               "within_pairs": int(y[k].sum() * (~y[k]).sum()),
               "top_predictions": json.dumps(Counter(arr(P, "pred", object)[k]).most_common(3))}
        for m in ["msp", "neg_entropy", "margin", "energy", "guardian_o", "guardian_full", "tg0",
                  "tg_agreement_then_msp", "corruption_tta_pred_prob"]:
            a = M.auroc(y[k], S[m][k])
            row[f"auroc_{m}"] = None if np.isnan(a) else a
            row[f"mean_{m}"] = float(S[m][k].mean())
        per_class.append(row)
    res["per_class"] = per_class

    # ---------------- S7 disruptions (RQ3)
    yc = y
    s7 = {}
    for v in DISRUPTIONS:
        yv = arr(P, f"correct_{v}", bool)
        accs = np.array([(w * yv).sum() / w.sum() for w in W])
        diff = np.array([((w * yv).sum() - (w * yc).sum()) / w.sum() for w in W])
        flipped = yc & ~yv
        rec = {"accuracy": float(yv.mean()), "accuracy_ci95": M.percentile_ci(accs),
               "accuracy_minus_clean": float(yv.mean() - yc.mean()),
               "accuracy_minus_clean_ci95": M.percentile_ci(diff),
               "msp_within_auroc_on_view": M.auroc(yv, arr(P, f"msp_{v}"), cls),
               "flip_rate_among_clean_correct": float(flipped.sum() / max(1, yc.sum())),
               "n_flips": int(flipped.sum()),
               "mean_msp_view": float(arr(P, f"msp_{v}").mean())}
        keep = yc
        for m in ("msp", "guardian_full", "tg0"):
            rec[f"flip_auroc_{m}"] = (M.auroc(~flipped[keep], S[m][keep])
                                      if flipped[keep].any() else None)
        s7[v] = rec
    res["S7_disruptions"] = s7
    res["clean_accuracy_mapped"] = float(yc.mean())

    # ---------------- S8 cost
    Tt = P[3:] if len(P) > 3 else P
    s8 = {}
    for m, vs in MONITOR_VIEWS.items():
        mm = np.array([t[f"model_ms_{m}"] for t in Tt])
        ee = np.array([t[f"e2e_ms_{m}"] for t in Tt])
        s8[m] = {"forward_passes": len(vs), "extra_passes": len(vs) - 1,
                 "model_ms_median": float(np.median(mm)), "model_ms_p95": float(np.percentile(mm, 95)),
                 "e2e_ms_median": float(np.median(ee)), "e2e_ms_p95": float(np.percentile(ee, 95))}
    for key, m in (("guardian_full", "guardian_full"), ("tg0", "tg0")):
        name = "H2_guardian_full_minus_msp" if m == "guardian_full" else "H3_tg0_minus_msp"
        r = res["primary"][name]
        ex = s8[m]["extra_passes"]
        if not r.get("not_estimable"):
            s8[m]["delta_within_auroc_per_extra_pass"] = r["estimate"] / ex
            s8[m]["delta_within_auroc_per_extra_pass_ci95"] = [x / ex for x in r["ci95_cluster"]]
    res["S8_cost"] = s8
    res["S8_note"] = "Measured per-view, unbatched, MPS; first 3 manifest clips excluded."

    # ---------------- S9 calibration transfer
    ucf_rows = load_manifest_rows(args.ucf_manifest)
    for r in ucf_rows:
        r["clip_id"] = hashlib.sha256(f"ucf101/{r['video']}".encode()).hexdigest()[:16]
    ucf_exp, ucf_missing = load_exports(args.ucf_exports, [r["clip_id"] for r in ucf_rows], views)
    cal = [r for r in ucf_rows if r["role"] == "ucf_calibration" and r["clip_id"] in ucf_exp]
    cal_mon = [M.monitor_scores(ucf_exp[r["clip_id"]]["logits"], frozen) for r in cal]
    cal_scores = {m: np.array([c[m] for c in cal_mon]) for m in ALL_MONITORS}
    frozen_thr = {"msp": frozen["baseline"]["threshold"],
                  "guardian_o": frozen["guardian_occlusion"]["threshold"],
                  "guardian_full": frozen["guardian_full"]["threshold"]}
    s9 = {"ucf_calibration_clips": len(cal), "ucf_missing": len(ucf_missing)}
    for m in ALL_MONITORS:
        try:
            rule_t = M.threshold_rule(cal_scores[m])
        except ValueError:
            rule_t = None
        t = frozen_thr.get(m, rule_t)
        rec = {"threshold": t, "threshold_source": "frozen_models_v2.json" if m in frozen_thr
               else "UCF101 calibration exports, frozen 70% rule",
               "rule_threshold_on_reexported_ucf_cal": rule_t}
        if m in frozen_thr and rule_t is not None:
            rec["reexport_reproduces_frozen_threshold_absdiff"] = abs(rule_t - frozen_thr[m])
        if t is not None:
            rec["ucf_cal_coverage"] = float(np.mean(cal_scores[m] >= t))
            rec["hmdb_P"] = M.operating_point(y, S[m], t)
            rec["hmdb_P"]["risk_accept_all"] = float(np.mean(~y))
        s9[m] = rec
    res["S9_calibration_transfer"] = s9

    # ---------------- exploratory: clip length
    nd_ = arr(P, "num_decoded_frames")
    short = nd_ < 64
    res["exploratory_clip_length"] = {
        m: {"within_auroc_short_lt64": M.auroc(y[short], S[m][short], cls[short]) if short.any() else None,
            "within_auroc_long_ge64": M.auroc(y[~short], S[m][~short], cls[~short]),
            "n_short": int(short.sum())} for m in ("msp", "tg0", "guardian_full")}
    res["runtime_seconds"] = time.time() - t_start

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    res = clean(res)
    (out / "STATISTICAL_RESULTS.json").open("x").write(json.dumps(res, indent=2) + "\n")

    # CSVs
    def wcsv(name, rows_):
        fields = []
        for r in rows_:
            fields += [k for k in r if k not in fields]
        with (out / name).open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows_)
    comp = {}
    for famname in ("S3_logit_baselines_vs_msp", "S4_tg_variants_vs_msp",
                    "exploratory_other_comparisons", "S2_tg0_vs_corruption_tta"):
        comp.update(res[famname])
    comp["guardian_full_minus_msp"] = res["primary"]["H2_guardian_full_minus_msp"]
    comp["tg0_minus_msp"] = res["primary"]["H3_tg0_minus_msp"]
    base_rows = []
    for m in ALL_MONITORS + CONTROLS:
        c = comp.get(f"{m}_minus_msp", {})
        pl = pooled[m]
        base_rows.append({
            "monitor": m, "forward_passes": len(MONITOR_VIEWS.get(m, [])) or None,
            "within_auroc": within[m]["within_auroc"],
            "within_auroc_ci_low": (within[m].get("within_auroc_ci95_cluster") or [None, None])[0],
            "within_auroc_ci_high": (within[m].get("within_auroc_ci95_cluster") or [None, None])[1],
            "macro_within_auroc": within[m]["macro_within_auroc"],
            "delta_within_vs_msp": c.get("estimate"),
            "delta_ci_low": (c.get("ci95_cluster") or [None, None])[0],
            "delta_ci_high": (c.get("ci95_cluster") or [None, None])[1],
            "delta_holm_jackknife_p": c.get("holm_jackknife_p"),
            "delta_reject_null": c.get("reject_null"),
            "delta_equivalent_pm0.03": (c.get("tost") or {}).get("equivalent"),
            "pooled_auroc": pl.get("pooled_auroc"), "aurc": pl.get("aurc", pl.get("aurc_tie_aware")),
            "aurc_ci_low": (pl.get("aurc_ci95") or [None, None])[0],
            "aurc_ci_high": (pl.get("aurc_ci95") or [None, None])[1],
            "aurc_minus_random": pl.get("aurc_minus_random"),
            "eaurc": pl.get("eaurc"), "augrc": pl.get("augrc"),
            "delta_aurc_vs_msp": pl.get("delta_aurc_vs_msp"),
            "class_composition_share": pl.get("class_composition_share"),
            "model_ms_median": s8.get(m, {}).get("model_ms_median"),
            "e2e_ms_median": s8.get(m, {}).get("e2e_ms_median"),
        })
    wcsv("BASELINE_COMPARISON.csv", clean(base_rows))
    wcsv("PER_CLASS_RESULTS.csv", clean(per_class))
    tg_rows = []
    for m in ["tg0"] + TG_VARIANTS + ["corruption_tta_pred_prob"]:
        c = comp.get(f"{m}_minus_msp", {})
        tg_rows.append({"monitor": m, "within_auroc": within[m]["within_auroc"],
                        "delta_within_vs_msp": c.get("estimate"),
                        "ci_low": (c.get("ci95_cluster") or [None, None])[0],
                        "ci_high": (c.get("ci95_cluster") or [None, None])[1],
                        "holm_p": c.get("holm_jackknife_p"), "reject": c.get("reject_null"),
                        "equivalent_pm0.03": (c.get("tost") or {}).get("equivalent"),
                        "forward_passes": len(MONITOR_VIEWS[m])})
    for v, r in s7.items():
        tg_rows.append({"monitor": f"disruption:{v}", **{k: r[k] for k in r if not isinstance(r[k], list)}})
    wcsv("TEMPORAL_MONITOR_RESULTS.csv", clean(tg_rows))
    wcsv("per_clip_table.csv", clean([{k: v for k, v in t.items()} for t in T]))

    manifest = {"analysis_sha256": sha256_file(Path(__file__)), "freeze": freeze,
                "outputs_sha256": {p.name: sha256_file(p) for p in sorted(out.iterdir())},
                "test_mode": args.test_mode, "runtime_seconds": res["runtime_seconds"]}
    (out / "analysis_run_manifest.json").open("x").write(json.dumps(manifest, indent=2) + "\n")
    print("confirmatory_claims_permitted:", confirmatory_valid)
    return res


if __name__ == "__main__":
    main()
