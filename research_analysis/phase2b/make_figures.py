"""Figures for Phase 2B from the analysis outputs (no new statistics).

    PYTHONDONTWRITEBYTECODE=1 .venv312/bin/python research_analysis/phase2b/make_figures.py \
        --results research_analysis/phase2b/results --output-dir research_analysis/phase2b/figures

Palette: reference categorical slots 1-3 (documented as passing all-pairs
CVD checks in light mode; the local validator could not be run because node is
not installed). At most 3 categorical series per figure; controls in neutral gray.
Every series is direct-labelled or has a legend (identity never color-alone).
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2 = "#9a9993", "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"
NICE = {"msp": "MSP", "neg_entropy": "Neg. entropy", "margin": "Top-2 margin",
        "energy": "Energy", "max_logit": "Max logit", "guardian_o": "Guardian-O",
        "guardian_full": "Guardian-Full", "tg0": "TG0 (temporal)",
        "tg_min": "TG-min", "tg_agreement_then_msp": "TG-agreement",
        "tg_neg_js": "TG-JS", "tg_sampling_pred_prob": "TG-sampling",
        "tg_seg2": "TG-seg2", "corruption_tta_pred_prob": "Corruption-TTA",
        "corruption_agreement_then_msp": "Corruption-agreement",
        "constant": "Constant (ctrl)", "random": "Random (ctrl)",
        "within_class_shuffled_msp": "Shuffled MSP (ctrl)", "class_oracle": "Class oracle (diag.)"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e8e7e2", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_within(res, out):
    w = res["within_class"]
    mons = [m for m in NICE if m in w and w[m]["within_auroc"] is not None]
    mons = sorted(mons, key=lambda m: w[m]["within_auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(mons) + 1.4))
    style(ax)
    for i, m in enumerate(mons):
        ctrl = m in ("constant", "random", "within_class_shuffled_msp", "class_oracle")
        c = GRAY if ctrl else (ORANGE if m in ("msp", "guardian_full", "tg0") else BLUE)
        ci = w[m].get("within_auroc_ci95_cluster")
        if ci:
            ax.plot(ci, [i, i], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(w[m]["within_auroc"], i, "o", color=c, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.5, len(mons) - 0.3, " chance", color=INK2, fontsize=8, va="bottom")
    ax.set_yticks(range(len(mons)))
    ax.set_yticklabels([NICE[m] for m in mons], color=INK, fontsize=9)
    ax.set_xlabel("Within-class AUROC (primary metric), 95% film-group cluster bootstrap CI",
                  color=INK2, fontsize=9)
    ax.set_title("Within-class error detection on HMDB51 (set P)\n"
                 "orange = preregistered primary monitors, gray = controls",
                 loc="left", color=INK, fontsize=10)
    save(fig, out, "fig1_within_class_auroc")


def fig_pooled_vs_within(res, out):
    w, p = res["within_class"], res["S1_pooled"]
    mons = [m for m in NICE if m in w and m in p and p[m].get("pooled_auroc") is not None
            and m not in ("constant",)]
    mons = sorted(mons, key=lambda m: p[m]["pooled_auroc"])
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(mons) + 1.4))
    style(ax)
    for i, m in enumerate(mons):
        a, b = p[m]["pooled_auroc"], w[m]["within_auroc"]
        ax.plot([a, b], [i, i], color="#d6d5cf", linewidth=2)
        ax.plot(a, i, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label="Pooled AUROC" if i == 0 else None)
        ax.plot(b, i, "s", color=ORANGE, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label="Within-class AUROC" if i == 0 else None)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(mons)))
    ax.set_yticklabels([NICE[m] for m in mons], color=INK, fontsize=9)
    ax.set_xlabel("AUROC (correct vs incorrect)", color=INK2, fontsize=9)
    ax.set_title("Pooled vs within-class AUROC (HMDB51 set P)\n"
                 "pooled AUROC also scores between-class pairs; within-class AUROC excludes them",
                 loc="left", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, out, "fig2_pooled_vs_within")


def fig_primary(res, out):
    fam = [("S6_strict_correctness", "strict"), ("S5_secondary_set", "set S"), ("primary", "set P")]
    keys = [("H2_guardian_full_minus_msp", "H2  Guardian-Full − MSP"),
            ("H3_tg0_minus_msp", "H3  TG0 − MSP")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    style(ax)
    ax.axvspan(-0.03, 0.03, color="#eeede8", zorder=0)
    ax.text(0.031, -0.75, "shaded: ±0.03 MID", ha="left", color=INK2, fontsize=8)
    ax.axvline(0, color=INK2, linewidth=1)
    y, labels = 0, []
    for k, name in keys:
        for f, fname in fam:
            r = (res.get(f) or {}).get(k)
            if not r or r.get("not_estimable"):
                labels.append(f"{name} ({fname}): n/a")
                y += 1
                continue
            ci, hci = r["ci95_cluster"], r["ci95_hierarchical"]
            ax.plot(hci, [y - 0.12] * 2, color=AQUA, linewidth=2, solid_capstyle="round",
                    label="Hierarchical (class→group) 95% CI" if y == 0 else None)
            ax.plot(ci, [y + 0.12] * 2, color=BLUE, linewidth=2, solid_capstyle="round",
                    label="Film-group cluster 95% CI" if y == 0 else None)
            ax.plot(r["estimate"], y + 0.12, "o", color=BLUE if f == "primary" else GRAY,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
            tag = "reject H0" if r.get("reject_null") else "not rejected"
            labels.append(f"{name} ({fname}) — {tag}")
            y += 1
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("Δ within-class AUROC vs MSP", color=INK2, fontsize=9)
    ax.set_ylim(-1, len(labels) - 0.4)
    ax.set_title("Preregistered H2/H3: Δ within-class AUROC vs MSP\n"
                 "set P = primary; set S and strict correctness = secondary", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.45), ncol=2)
    save(fig, out, "fig3_primary_comparisons")


def fig_risk_coverage(results, out):
    rows = list(csv.DictReader((results / "per_clip_table.csv").open()))
    rows = [r for r in rows if r["role"] == "confirmatory_primary"]
    y = np.array([r["correct_mapped"] == "True" for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.8)
    for m, c in (("msp", BLUE), ("guardian_full", ORANGE), ("tg0", AQUA)):
        s = np.array([float(r[m]) for r in rows])
        e = ~y[np.argsort(-s, kind="stable")]
        k = np.arange(1, len(e) + 1)
        ax.plot(k / len(e), np.cumsum(e) / k, color=c, linewidth=2, label=NICE[m])
    ax.axhline(np.mean(~y), color=GRAY, linestyle="--", linewidth=1.5, label="Random ranking (error rate)")
    ax.set_xlabel("Coverage (fraction of clips accepted)", color=INK2, fontsize=9)
    ax.set_ylabel("Selective risk (error rate among accepted)", color=INK2, fontsize=9)
    ax.set_title("Risk–coverage on HMDB51 set P (pooled; point estimates)", loc="left",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, 1)
    save(fig, out, "fig4_risk_coverage")


def fig_disruptions(res, out):
    s7 = res["S7_disruptions"]
    vs = list(s7)
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(vs) + 1.4))
    style(ax)
    for i, v in enumerate(vs):
        r = s7[v]
        ax.plot(r["accuracy_minus_clean_ci95"], [i, i], color=BLUE, linewidth=2,
                solid_capstyle="round")
        ax.plot(r["accuracy_minus_clean"], i, "o", color=BLUE, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(max(r["accuracy_minus_clean_ci95"][1], 0) + 0.004, i,
                f"flips {r['n_flips']}", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(vs, fontsize=9, color=INK)
    ax.set_xlabel("Top-1 accuracy change vs clean adaptive view (mapped), 95% cluster CI",
                  color=INK2, fontsize=9)
    ax.set_title("Temporal disruptions: effect on accuracy (HMDB51 set P)", loc="left",
                 color=INK, fontsize=10)
    save(fig, out, "fig5_disruptions")


def fig_per_class(res, out):
    pc = res["per_class"]
    pc = sorted(pc, key=lambda r: r["accuracy_mapped"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 0.34 * len(pc) + 1.6), sharey=True)
    for ax in (a1, a2):
        style(ax)
    ys = range(len(pc))
    a1.barh(ys, [r["accuracy_mapped"] for r in pc], color=BLUE, height=0.6)
    a1.set_xlabel("Clean top-1 accuracy (mapped)", color=INK2, fontsize=9)
    a1.set_xlim(0, 1)
    a1.set_yticks(list(ys))
    a1.set_yticklabels([f"{r['hmdb_class']} (pairs {r['within_pairs']})" for r in pc], fontsize=8)
    for j, (m, c, mk) in enumerate((("msp", BLUE, "o"), ("guardian_full", ORANGE, "s"),
                                    ("tg0", AQUA, "D"))):
        xs = [r.get(f"auroc_{m}") for r in pc]
        a2.plot([x for x in xs if x is not None], [i + (j - 1) * 0.18 for i, x in enumerate(xs) if x is not None],
                mk, color=c, markersize=7, markeredgecolor=SURFACE, label=NICE[m])
    a2.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
    a2.set_xlim(0, 1)
    a2.set_xlabel("Per-class AUROC (missing = no errors or no correct)", color=INK2, fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.suptitle("Per-class accuracy and error detection (HMDB51 set P)", x=0.01, ha="left",
                 color=INK, fontsize=10)
    save(fig, out, "fig6_per_class")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if any(a.output_dir.iterdir()):
        raise FileExistsError("figure dir not empty")
    res = json.loads((a.results / "STATISTICAL_RESULTS.json").read_text())
    fig_within(res, a.output_dir)
    fig_pooled_vs_within(res, a.output_dir)
    fig_primary(res, a.output_dir)
    fig_risk_coverage(a.results, a.output_dir)
    fig_disruptions(res, a.output_dir)
    fig_per_class(res, a.output_dir)
    print("figures ->", a.output_dir)


if __name__ == "__main__":
    main()
