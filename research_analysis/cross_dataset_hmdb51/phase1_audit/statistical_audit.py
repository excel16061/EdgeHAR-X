"""Independent statistical audit of Phase 1 (HMDB51 validation V2).

Re-derives every per-clip score from the raw prediction CSVs and the frozen
model JSON using an implementation independent of the Phase 1 code, then:

  A. verifies AURC / risk-coverage / tie handling and reproduces Phase 1
     point estimates, CIs, bootstrap p-values and Holm adjustment;
  B. re-runs inference under alternative, defensible methods (cluster
     bootstrap variants, BCa, cluster jackknife, clustered DeLong /
     Obuchowski, randomization) plus seed / replicate sensitivity;
  C. audits the HMDB51 subset, source-group structure and decomposes the
     Guardian-Full effect into between-class and within-class parts;
  D. evaluates calibration transfer UCF101 -> HMDB51;
  E. compares against lightweight baselines computable from saved artifacts.

Nothing is trained. No threshold is chosen on HMDB51 (baseline thresholds use
the same rule as the frozen models, applied to UCF101 calibration data only).

Usage (repository root):
    .venv312/bin/python research_analysis/cross_dataset_hmdb51/phase1_audit/statistical_audit.py \
        [--output-dir DIR] [--bootstraps 10000] [--seed 2027]
Refuses to overwrite existing output files.
"""

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "results/guardian_v3_official"
FROZEN = OFFICIAL / "frozen_models_v2.json"
UCF_EVAL = OFFICIAL / "evaluation_balanced.csv"
UCF_CAL = OFFICIAL / "calibration_balanced.csv"
UCF_FINAL = OFFICIAL / "final_evaluation.json"
UCF_RC = OFFICIAL / "risk_coverage/risk_coverage_report.json"
HMDB = ROOT / "research_analysis/cross_dataset_hmdb51"
HMDB_PRED = HMDB / "hmdb51_full_predictions.csv"
HMDB_MANIFEST = HMDB / "hmdb51_mapped_test_manifest.csv"
SPLITS = HMDB / "hmdb51_splits/testTrainMulti_7030_splits"
V2_RESULTS = HMDB / "validation_v2/outputs/validation_v2_results.json"
V2_CLIPS = HMDB / "validation_v2/outputs/per_clip_scores.csv"
V2_LOCO = HMDB / "validation_v2/outputs/leave_one_class_out.csv"
OUT_DEFAULT = HMDB / "phase1_audit"

INPUTS = [FROZEN, UCF_EVAL, UCF_CAL, UCF_FINAL, UCF_RC, HMDB_PRED,
          HMDB_MANIFEST, V2_RESULTS, V2_CLIPS, V2_LOCO]
EXPECTED_SHA256 = (
    "e56fececf1f980a2eafbc9fc882ac4157b3829802a8c492b9a31994db1022cd1")
PROBES = ("blur", "dark", "occlusion", "frame_drop")
MAPPING = {"fencing": "Fencing", "punch": "Punch", "pushup": "PushUps",
           "pullup": "PullUps", "ride_bike": "Biking",
           "ride_horse": "HorseRiding"}
TARGET_COVERAGE = 0.70
LOTR = {"Two_Towers", "Return_of_the_King"}

# Monitors: name -> (kind, forward passes). All defined from saved
# artifacts only; none uses HMDB51 labels.
MONITORS = {
    "msp": ("confidence (frozen baseline)", 1),
    "guardian_o": ("learned on UCF101 (frozen)", 2),
    "guardian_full": ("learned on UCF101 (frozen)", 5),
    "mean_view_conf": ("zero-parameter heuristic", 5),
    "min_view_conf": ("zero-parameter heuristic", 5),
    "agreement_then_msp": ("zero-parameter heuristic", 5),
    "occ_agreement_then_msp": ("zero-parameter heuristic", 2),
}
CONTROLS = {
    "constant": "negative control (no information)",
    "class_oracle": "DIAGNOSTIC ONLY: uses test labels (class accuracy)",
}
PRIMARY = [("guardian_o", "msp"), ("guardian_full", "msp")]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ===================================================================== data

def load_grouped(path, rows_per_video):
    grouped = defaultdict(dict)
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("error", "").strip():
                raise ValueError(f"inference error row: {r['video']}")
            key = (r["condition"], int(r["severity"]))
            if key in grouped[r["video"]]:
                raise ValueError(f"duplicate row {r['video']} {key}")
            grouped[r["video"]][key] = r
    for v, g in grouped.items():
        if len(g) != rows_per_video:
            raise ValueError(f"{v}: {len(g)} rows, expected {rows_per_video}")
    return grouped


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def frozen_score(model, X):
    w = np.asarray(model["weights"], float)
    mu = np.asarray(model["feature_mean"], float)
    sd = np.asarray(model["feature_std"], float)
    X = np.atleast_2d(np.asarray(X, float))
    assert X.shape[1] == len(mu) == len(w) - 1 and np.all(sd > 0)
    return sigmoid(w[0] + ((X - mu) / sd) @ w[1:])


def hmdb_prefix(video):
    cls, fn = video.split("/", 1)
    marker = f"_{cls}_"
    if marker not in fn:
        raise ValueError(video)
    return fn.rsplit(marker, 1)[0]


def film_group(prefix, franchise=True):
    """Class-agnostic source grouping: drop a trailing part number
    ('Return_of_the_King_4' -> 'Return_of_the_King') and optionally merge the
    Lord of the Rings films. Rule fixed before computing any statistic."""
    title = re.sub(r"_\d+$", "", prefix)
    if franchise and title in LOTR:
        title = "Lord_of_the_Rings"
    return title


def build(grouped, models, dataset):
    rec = []
    for video in sorted(grouped):
        g = grouped[video]
        clean = g[("clean", 0)]
        probes = [g[(p, 3)] for p in PROBES]
        s = float(clean["score"])
        ps = np.array([float(r["score"]) for r in probes])
        drops = s - ps
        agree = np.array([r["predicted_label"] == clean["predicted_label"]
                          for r in probes])
        occ_change = float(g[("occlusion", 3)]["predicted_label"]
                           != clean["predicted_label"])
        full = [s, drops.mean(), drops.max(), 1.0 - agree.mean(), ps.std()]
        cls = video.split("/")[0]
        if dataset == "hmdb51":
            prefix = hmdb_prefix(video)
            group = f"{cls}/{prefix}"
            film = film_group(prefix, franchise=False)
            franchise = film_group(prefix, franchise=True)
        else:
            m = re.search(r"_g(\d+)_c\d+\.avi$", video)
            group = film = franchise = f"{cls}/g{m.group(1)}"
        ms_clean = float(clean["inference_ms"])
        ms_occ = float(g[("occlusion", 3)]["inference_ms"])
        ms_all = ms_clean + sum(float(r["inference_ms"]) for r in probes)
        rec.append({
            "video": video, "cls": cls, "group": group, "film": film,
            "franchise": franchise,
            "correct": bool(int(clean["correct"])),
            "pred": clean["predicted_label"], "true": clean["true_label"],
            "msp": s,
            "guardian_o": float(frozen_score(models["guardian_occlusion"],
                                             [[s, occ_change]])[0]),
            "guardian_full": float(frozen_score(models["guardian_full"],
                                                [full])[0]),
            "mean_view_conf": float(np.mean(np.r_[s, ps])),
            "min_view_conf": float(np.min(np.r_[s, ps])),
            "agreement_then_msp": 2.0 * agree.sum() + s,
            "occ_agreement_then_msp": 2.0 * (1.0 - occ_change) + s,
            "ms": {"msp": ms_clean, "guardian_o": ms_clean + ms_occ,
                   "guardian_full": ms_all, "mean_view_conf": ms_all,
                   "min_view_conf": ms_all, "agreement_then_msp": ms_all,
                   "occ_agreement_then_msp": ms_clean + ms_occ},
        })
    return rec


class Data:
    def __init__(self, recs, name):
        self.name = name
        self.recs = recs
        self.n = len(recs)
        self.y = np.array([r["correct"] for r in recs], bool)
        self.cls = np.array([r["cls"] for r in recs])
        self.group = np.array([r["group"] for r in recs])
        self.film = np.array([r["film"] for r in recs])
        self.franchise = np.array([r["franchise"] for r in recs])
        self.s = {m: np.array([r[m] for r in recs]) for m in MONITORS}
        self.s["constant"] = np.full(self.n, 0.5)
        acc = {c: self.y[self.cls == c].mean() for c in np.unique(self.cls)}
        self.s["class_oracle"] = np.array([acc[c] for c in self.cls])
        self.ms = {m: np.array([r["ms"][m] for r in recs]) for m in MONITORS}


def verify_hmdb_manifest():
    with HMDB_MANIFEST.open(newline="") as f:
        man = list(csv.DictReader(f))
    out = {}
    for cls, ucf in MAPPING.items():
        rows = [r for r in man if r["hmdb_class"] == cls]
        official = dict(l.split() for l in
                        (SPLITS / f"{cls}_test_split1.txt").read_text().splitlines()
                        if l.strip())
        test = {k for k, v in official.items() if v == "2"}
        assert {r["filename"] for r in rows} == test
        assert all(r["ucf_class"] == ucf for r in rows)
        out[cls] = {"test_clips_split1": len(test),
                    "train_clips_split1": sum(v == "1" for v in official.values()),
                    "unused_split1": sum(v == "0" for v in official.values())}
    assert len(man) == 180
    return out


def cross_class_sources():
    """Source prefixes of the 180 clips that occur in other HMDB51 classes
    (any partition)."""
    man = list(csv.DictReader(HMDB_MANIFEST.open(newline="")))
    prefixes = defaultdict(set)
    for r in man:
        prefixes[r["filename"].rsplit(f"_{r['hmdb_class']}_", 1)[0]].add(r["hmdb_class"])
    other = defaultdict(set)
    for f in SPLITS.glob("*_test_split1.txt"):
        c = f.name.replace("_test_split1.txt", "")
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            name = line.split()[0]
            pref = name.rsplit(f"_{c}_", 1)[0] if f"_{c}_" in name else None
            if pref in prefixes and c not in prefixes[pref]:
                other[pref].add(c)
    within_subset = {p: sorted(c) for p, c in prefixes.items() if len(c) > 1}
    return within_subset, {p: sorted(c) for p, c in other.items()}


# ================================================================== metrics

def aurc(y, s, counts=None, order=None):
    """Discrete AURC = mean selective risk over accepted-set sizes 1..N,
    descending score, stable tie order (same definition as Phase 1 and
    guardian_v3_risk_coverage.py). `counts` = bootstrap multiplicities."""
    if order is None:
        order = np.argsort(-s, kind="stable")
    e = ~y[order]
    if counts is not None:
        e = np.repeat(e, counts[order])
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def aurc_expected(y, s):
    """Tie-aware AURC: expectation over uniformly random ordering of tied
    scores (errors accrue linearly inside a tied block)."""
    order = np.argsort(-s, kind="stable")
    ss, e = s[order], (~y[order]).astype(float)
    n = len(s)
    cum = np.empty(n)
    i, E = 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        rate = e[i:j + 1].mean()
        cum[i:j + 1] = E + rate * np.arange(1, j - i + 2)
        E += e[i:j + 1].sum()
        i = j + 1
    return float(np.mean(cum / np.arange(1, n + 1)))


def aurc_optimal(y):
    return aurc(y, y.astype(float))


def n_ties(s):
    _, c = np.unique(s, return_counts=True)
    return int(np.sum(c[c > 1]))


def pairs(y, cls=None):
    pos, neg = np.where(y)[0], np.where(~y)[0]
    P, Q = np.meshgrid(pos, neg, indexing="ij")
    P, Q = P.ravel(), Q.ravel()
    if cls is not None:
        keep = cls[P] == cls[Q]
        P, Q = P[keep], Q[keep]
    return P, Q


def psi(s, P, Q):
    return (s[P] > s[Q]) + 0.5 * (s[P] == s[Q])


def weighted_auc(ps, w):
    den = w.sum()
    return float((ps * w).sum() / den) if den > 0 else np.nan


def auroc(y, s, cls=None):
    P, Q = pairs(y, cls)
    return weighted_auc(psi(s, P, Q), np.ones(len(P))) if len(P) else np.nan


def risk_at(y, s, frac):
    k = int(round(len(y) * frac))
    return float(np.mean(~y[np.argsort(-s, kind="stable")[:k]]))


def threshold_rule(scores):
    """Exact rule of guardian_v3_freeze_corrected.get_threshold."""
    o = np.sort(np.asarray(scores))[::-1]
    k = round(len(o) * TARGET_COVERAGE)
    if o[k - 1] == o[k]:
        raise ValueError("tied threshold boundary")
    return float((o[k - 1] + o[k]) / 2)


def operating_point(y, s, t, counts=None):
    w = np.ones(len(y)) if counts is None else counts.astype(float)
    acc = s >= t
    n_acc, n = (w * acc).sum(), w.sum()
    n_c, n_w = (w * y).sum(), (w * ~y).sum()
    return {
        "coverage": n_acc / n,
        "selective_risk": (w * (acc & ~y)).sum() / n_acc if n_acc else np.nan,
        "false_alarm_rate": (w * (~acc & y)).sum() / n_c if n_c else np.nan,
        "miss_rate": (w * (acc & ~y)).sum() / n_w if n_w else np.nan,
        "error_rate_accept_all": n_w / n,
    }


# ================================================================ resampling

def group_index(labels):
    keys = sorted(set(labels))
    pos = {k: i for i, k in enumerate(keys)}
    return keys, np.array([pos[x] for x in labels])


def stratified_cluster_counts(d, B, seed, groups=None):
    """Resample groups with replacement within class. RNG call order matches
    validation_v2.cluster_bootstrap exactly (classes sorted, groups sorted)."""
    groups = d.group if groups is None else groups
    keys, gidx = group_index(groups)
    by_class = defaultdict(list)
    for k in keys:
        by_class[k.split("/", 1)[0]].append(keys.index(k))
    rng = np.random.default_rng(seed)
    C = np.empty((B, d.n), dtype=np.int32)
    G = len(keys)
    for b in range(B):
        gc = np.zeros(G, dtype=np.int32)
        for c in sorted(by_class):
            ks = by_class[c]
            for j in rng.integers(0, len(ks), len(ks)):
                gc[ks[j]] += 1
        C[b] = gc[gidx]
    return C


def unstratified_cluster_counts(groups, B, seed):
    keys, gidx = group_index(groups)
    rng = np.random.default_rng(seed)
    G = len(keys)
    draws = rng.integers(0, G, (B, G))
    gc = np.zeros((B, G), dtype=np.int32)
    for b in range(B):
        gc[b] = np.bincount(draws[b], minlength=G)
    return gc[:, gidx]


def hierarchical_counts(d, B, seed):
    """Two-stage: classes with replacement, then groups within each drawn
    class. Treats the 6 classes as a sample of activities (crude with 6)."""
    keys, gidx = group_index(d.group)
    classes = sorted(set(d.cls))
    by_class = {c: [i for i, k in enumerate(keys) if k.split("/", 1)[0] == c]
                for c in classes}
    rng = np.random.default_rng(seed)
    C = np.empty((B, d.n), dtype=np.int32)
    for b in range(B):
        gc = np.zeros(len(keys), dtype=np.int32)
        for ci in rng.integers(0, len(classes), len(classes)):
            ks = by_class[classes[ci]]
            for j in rng.integers(0, len(ks), len(ks)):
                gc[ks[j]] += 1
        C[b] = gc[gidx]
    return C


def jackknife_counts(groups):
    keys, gidx = group_index(groups)
    return np.array([(gidx != g).astype(np.int32) for g in range(len(keys))])


def percentile_ci(v):
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def boot_p(v):
    v = v[~np.isnan(v)]
    return float(min(1.0, 2 * min(np.mean(v <= 0), np.mean(v >= 0))))


def bca_ci(theta, boot, jack, alpha=0.05):
    boot = boot[~np.isnan(boot)]
    prop = np.mean(boot < theta) + 0.5 * np.mean(boot == theta)
    prop = min(max(prop, 1e-10), 1 - 1e-10)
    z0 = stats.norm.ppf(prop)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6 * np.sum((jm - jack) ** 2) ** 1.5
    a = num / den if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = stats.norm.ppf(q)
        out.append(float(np.percentile(
            boot, 100 * stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))
    return out, float(z0), float(a)


def jackknife_t(theta, jack):
    G = len(jack)
    se = float(np.sqrt((G - 1) / G * np.sum((jack - jack.mean()) ** 2)))
    t = theta / se if se > 0 else np.inf
    q = stats.t.ppf(0.975, G - 1)
    return {"se": se, "ci95": [theta - q * se, theta + q * se],
            "p_two_sided": float(2 * stats.t.sf(abs(t), G - 1)), "df": G - 1}


def holm(p):
    names = sorted(p, key=p.get)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * p[k]))
        out[k] = run
    return out


# ------------------------------------------- clustered DeLong (Obuchowski)

def obuchowski(y, score_list, groups):
    """Clustered nonparametric AUC variance/covariance (Obuchowski 1997,
    Biometrics 53:567-578). Reduces to DeLong when every cluster has one
    unit. Returns AUC vector and covariance matrix."""
    keys, gidx = group_index(groups)
    I = len(keys)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    M, N = len(pos), len(neg)
    K = len(score_list)
    A = np.zeros(K)
    V10 = np.zeros((K, I))
    V01 = np.zeros((K, I))
    m_i = np.bincount(gidx[pos], minlength=I)
    n_i = np.bincount(gidx[neg], minlength=I)
    for k, s in enumerate(score_list):
        P = psi(s, *np.meshgrid(pos, neg, indexing="ij"))  # M x N
        A[k] = P.mean()
        np.add.at(V10[k], gidx[pos], P.mean(axis=1))
        np.add.at(V01[k], gidx[neg], P.mean(axis=0))
    I10, I01 = np.sum(m_i > 0), np.sum(n_i > 0)
    D10 = V10 - m_i[None] * A[:, None]
    D01 = V01 - n_i[None] * A[:, None]
    S10 = I10 / ((I10 - 1) * M) * D10 @ D10.T
    S01 = I01 / ((I01 - 1) * N) * D01 @ D01.T
    S11 = I / (I - 1) * D10 @ D01.T
    cov = S10 / M + S01 / N + (S11 + S11.T) / (M * N)
    return A, cov


def delong_unclustered(y, a, b):
    """Plain DeLong (independent units) for test cross-check."""
    return obuchowski(y, [a, b], np.arange(len(y)).astype(str))


# ======================================================== replicate engine

class Engine:
    """Computes all metrics for a dataset given multiplicity vectors."""

    def __init__(self, d, thresholds):
        self.d = d
        self.thr = thresholds
        self.order = {m: np.argsort(-s, kind="stable") for m, s in d.s.items()}
        self.Pp, self.Qp = pairs(d.y)
        self.Pw, self.Qw = pairs(d.y, d.cls)
        self.psi_p = {m: psi(s, self.Pp, self.Qp) for m, s in d.s.items()}
        self.psi_w = {m: psi(s, self.Pw, self.Qw) for m, s in d.s.items()}

    def run(self, counts, monitors):
        d = self.d
        wp = counts[self.Pp] * counts[self.Qp]
        ww = counts[self.Pw] * counts[self.Qw]
        out = {"accuracy": float((counts * d.y).sum() / counts.sum())}
        for m in monitors:
            out[f"aurc/{m}"] = aurc(d.y, d.s[m], counts, self.order[m])
            # random ranking has expected AURC = error rate
            out[f"aurc_minus_random/{m}"] = out[f"aurc/{m}"] - (1 - out["accuracy"])
            out[f"auroc/{m}"] = weighted_auc(self.psi_p[m], wp)
            out[f"strat_auroc/{m}"] = weighted_auc(self.psi_w[m], ww)
            if m in self.thr:
                op = operating_point(d.y, d.s[m], self.thr[m], counts)
                out[f"fixed_risk/{m}"] = op["selective_risk"]
                out[f"fixed_cov/{m}"] = op["coverage"]
                out[f"fixed_far/{m}"] = op["false_alarm_rate"]
                out[f"fixed_miss/{m}"] = op["miss_rate"]
                out[f"risk_reduction/{m}"] = (
                    op["error_rate_accept_all"] - op["selective_risk"])
        return out

    def replicate(self, C, monitors):
        rows = [self.run(c, monitors) for c in C]
        return {k: np.array([r[k] for r in rows], float) for k in rows[0]}


# ================================================================== audit

def within_class_label_permutation(d, monitors, P, seed):
    """Permute correctness labels within class (same permutation for every
    monitor). Null: monitors carry no within-class information; class
    accuracies and score distributions are preserved. Its mean AURC is the
    AURC attributable to class composition alone. Clip-level exchangeability
    ignores source groups (p-values approximate)."""
    rng = np.random.default_rng(seed)
    idx_by_class = [np.where(d.cls == c)[0] for c in np.unique(d.cls)]
    orders = {m: np.argsort(-d.s[m], kind="stable") for m in monitors}
    tied = {m for m in monitors if n_ties(d.s[m]) > 0}
    res = {m: np.empty(P) for m in monitors}
    for p in range(P):
        y = d.y.copy()
        for ix in idx_by_class:
            y[ix] = d.y[rng.permutation(ix)]
        for m in monitors:
            res[m][p] = (aurc_expected(y, d.s[m]) if m in tied
                         else aurc(y, d.s[m], order=orders[m]))
    return res


def swap_randomization(d, a, b, groups, P, seed):
    """Paired randomization: within each source group, swap the two
    monitors' normalized-rank scores with probability 1/2. Assumes the two
    monitors' rank scores are exchangeable within group under H0; this is a
    heuristic assumption (see AUDIT_REPORT.md)."""
    ra = stats.rankdata(d.s[a]) / d.n
    rb = stats.rankdata(d.s[b]) / d.n
    obs = aurc(d.y, ra) - aurc(d.y, rb)
    keys, gidx = group_index(groups)
    rng = np.random.default_rng(seed)
    null = np.empty(P)
    for p in range(P):
        flip = rng.random(len(keys)) < 0.5
        f = flip[gidx]
        null[p] = aurc(d.y, np.where(f, rb, ra)) - aurc(d.y, np.where(f, ra, rb))
    pval = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + P)
    return {"observed_rank_aurc_diff": float(obs), "p_two_sided": float(pval),
            "permutations": P}


def write_csv(path, rows):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--bootstraps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--sensitivity-grid", default="1000,5000,10000,50000")
    ap.add_argument("--sensitivity-seeds", default="1,42,2027,12345,99991")
    args = ap.parse_args(argv)
    B, SEED = args.bootstraps, args.seed

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    targets = ["audit_results.json", "baseline_comparison.csv",
               "inference_sensitivity.csv", "seed_sensitivity.csv",
               "decomposition.csv", "calibration_transfer.csv",
               "run_manifest.json"]
    conflicts = [t for t in targets if (out / t).exists()]
    if conflicts:
        raise FileExistsError(f"Refusing to overwrite: {conflicts}")

    in_hash = {p: sha256(p) for p in INPUTS}
    if in_hash[FROZEN] != EXPECTED_SHA256:
        raise RuntimeError("Frozen model hash mismatch")
    frozen = json.loads(FROZEN.read_text())
    assert frozen["evaluation_data_used"] is False
    fm = frozen["models"]

    # ---------------------------------------------------------- load data
    split_info = verify_hmdb_manifest()
    hmdb = Data(build(load_grouped(HMDB_PRED, 5), fm, "hmdb51"), "hmdb51")
    ucf = Data(build(load_grouped(UCF_EVAL, 13), fm, "ucf101"), "ucf101")
    cal = Data(build(load_grouped(UCF_CAL, 13), fm, "ucf101"), "ucf101_cal")
    assert hmdb.n == 180 and ucf.n == 505 and cal.n == 202
    assert len(set(hmdb.group)) == 50
    assert not set(ucf.group) & set(cal.group)
    assert all(MAPPING[r["cls"]] == r["true"] for r in hmdb.recs)

    checks = {}

    # independent reproduction of Phase 1 per-clip scores
    v2 = {r["video"]: r for r in csv.DictReader(V2_CLIPS.open(newline=""))}
    mx = 0.0
    for r in hmdb.recs:
        for mine, theirs in (("msp", "baseline"), ("guardian_o", "guardian_occlusion"),
                             ("guardian_full", "guardian_full")):
            mx = max(mx, abs(r[mine] - float(v2[r["video"]][theirs])))
        assert int(v2[r["video"]]["correct"]) == int(r["correct"])
    checks["max_abs_score_diff_vs_phase1"] = mx
    assert mx < 1e-12

    # thresholds: reproduce frozen thresholds from UCF101 calibration data
    thr = {}
    for m, key in (("msp", "baseline"), ("guardian_o", "guardian_occlusion"),
                   ("guardian_full", "guardian_full")):
        t = threshold_rule(cal.s[m])
        checks[f"threshold_reproduced/{m}"] = abs(t - fm[key]["threshold"])
        assert abs(t - fm[key]["threshold"]) < 1e-12
        thr[m] = fm[key]["threshold"]
    for m in MONITORS:
        if m not in thr:
            thr[m] = threshold_rule(cal.s[m])  # UCF101 calibration only

    # UCF101 reproduction of reported numbers
    ucf_final = json.loads(UCF_FINAL.read_text())
    ucf_rc = json.loads(UCF_RC.read_text())
    for m, key in (("msp", "baseline"), ("guardian_o", "guardian_occlusion"),
                   ("guardian_full", "guardian_full")):
        op = operating_point(ucf.y, ucf.s[m], thr[m])
        assert abs(op["coverage"] - ucf_final["models"][key]["coverage"]) < 1e-12
        assert abs(op["selective_risk"]
                   - ucf_final["models"][key]["selective_risk"]) < 1e-12
        assert abs(aurc(ucf.y, ucf.s[m]) - ucf_rc["models"][key]["aurc"]) < 1e-12
    checks["ucf101_final_eval_and_aurc_reproduced"] = True

    # ties
    checks["ties"] = {d.name: {m: n_ties(d.s[m]) for m in MONITORS}
                      for d in (hmdb, ucf)}
    for d in (hmdb, ucf):
        for m in MONITORS:
            assert abs(aurc(d.y, d.s[m]) - aurc_expected(d.y, d.s[m])) < 1e-12

    # ------------------------------------------------- A. Phase 1 reproduction
    eh = Engine(hmdb, thr)
    allm = list(MONITORS)
    C_strat = stratified_cluster_counts(hmdb, B, SEED)
    R = eh.replicate(C_strat, allm)
    v2r = json.loads(V2_RESULTS.read_text())
    repro = {}
    p_primary = {}
    for a, b in PRIMARY:
        v2name = {"guardian_o": "guardian_occlusion_minus_baseline",
                  "guardian_full": "guardian_full_minus_baseline"}[a]
        diff = R[f"aurc/{a}"] - R[f"aurc/{b}"]
        theta = aurc(hmdb.y, hmdb.s[a]) - aurc(hmdb.y, hmdb.s[b])
        ci = percentile_ci(diff)
        p = boot_p(diff)
        p_primary[a] = p
        ref = v2r["comparisons"][v2name]["aurc"]
        repro[a] = {
            "delta_aurc": theta, "ci95": ci, "p_boot": p,
            "replicates_ge_zero": int(np.sum(diff >= 0)),
            "replicates_le_zero": int(np.sum(diff <= 0)),
            "phase1_delta": ref["estimate"], "phase1_ci95": ref["ci95"],
            "phase1_p": ref["bootstrap_p_two_sided"],
            "phase1_holm": ref["holm_adjusted_p"],
            "matches_phase1": bool(
                abs(theta - ref["estimate"]) < 1e-12
                and np.allclose(ci, ref["ci95"], atol=1e-12)
                and abs(p - ref["bootstrap_p_two_sided"]) < 1e-12),
            "monte_carlo_se_of_p": float(np.sqrt(p * (1 - p) / B)),
        }
    hp = holm(p_primary)
    for a in hp:
        repro[a]["holm"] = hp[a]
        repro[a]["holm_matches_phase1"] = abs(hp[a] - repro[a]["phase1_holm"]) < 1e-12
    if B == 10000 and SEED == 2027:  # Phase 1 settings: must match exactly
        assert all(r["matches_phase1"] and r["holm_matches_phase1"]
                   for r in repro.values())
    checks["phase1_settings_reproduced"] = bool(
        B == 10000 and SEED == 2027
        and all(r["matches_phase1"] for r in repro.values()))

    # -------------------------------------------- B. alternative inference
    sens_rows = []
    J_group = jackknife_counts(hmdb.group)
    J_film = jackknife_counts(hmdb.franchise)
    C_unstrat = unstratified_cluster_counts(hmdb.group, B, SEED)
    C_film = unstratified_cluster_counts(hmdb.franchise, B, SEED)
    C_hier = hierarchical_counts(hmdb, B, SEED)
    # every clip its own "group" (class prefix kept for stratification)
    clip_groups = np.array([f"{r['cls']}/{r['video']}" for r in hmdb.recs])
    C_clip = stratified_cluster_counts(hmdb, B, SEED, groups=clip_groups)
    prim_m = ["msp", "guardian_o", "guardian_full"]
    R_un = eh.replicate(C_unstrat, prim_m)
    R_film = eh.replicate(C_film, prim_m)
    R_hier = eh.replicate(C_hier, prim_m)
    R_clip = eh.replicate(C_clip, prim_m)
    RJ_g = eh.replicate(J_group, prim_m)
    RJ_f = eh.replicate(J_film, prim_m)

    def add(method, a, metric, theta, ci, p, note):
        sens_rows.append({"method": method, "comparison": f"{a}_minus_msp",
                          "metric": metric, "estimate": theta,
                          "ci95_low": ci[0] if ci else None,
                          "ci95_high": ci[1] if ci else None,
                          "p_two_sided": p,
                          "ci_excludes_zero": (None if not ci else
                                               bool(ci[0] > 0 or ci[1] < 0)),
                          "note": note})

    swap = {}
    for a in ("guardian_o", "guardian_full"):
        theta = aurc(hmdb.y, hmdb.s[a]) - aurc(hmdb.y, hmdb.s["msp"])
        dd = lambda RR: RR[f"aurc/{a}"] - RR["aurc/msp"]
        add("clip_bootstrap_stratified_by_class (ignores groups)", a, "aurc",
            theta, percentile_ci(dd(R_clip)), boot_p(dd(R_clip)),
            "Treats correlated clips as independent; shown for contrast only.")
        add("cluster_bootstrap_stratified_percentile (Phase 1)", a, "aurc",
            theta, percentile_ci(dd(R)), boot_p(dd(R)),
            "50 class-specific groups, classes fixed.")
        bca, z0, acc = bca_ci(theta, dd(R), dd(RJ_g))
        add("cluster_bootstrap_stratified_BCa", a, "aurc", theta, bca, None,
            f"z0={z0:.3f}, acceleration={acc:.4f} (group jackknife).")
        add("cluster_bootstrap_unstratified_percentile", a, "aurc", theta,
            percentile_ci(dd(R_un)), boot_p(dd(R_un)),
            "50 groups resampled ignoring class; class mix varies.")
        add("film_level_cluster_bootstrap_unstratified", a, "aurc", theta,
            percentile_ci(dd(R_film)), boot_p(dd(R_film)),
            f"{len(set(hmdb.franchise))} class-agnostic film/source groups.")
        jt = jackknife_t(theta, dd(RJ_g))
        add("cluster_jackknife_t (50 groups)", a, "aurc", theta, jt["ci95"],
            jt["p_two_sided"], f"df={jt['df']}, se={jt['se']:.5f}")
        jf = jackknife_t(theta, dd(RJ_f))
        add("cluster_jackknife_t (film-level groups)", a, "aurc", theta,
            jf["ci95"], jf["p_two_sided"], f"df={jf['df']}, se={jf['se']:.5f}")
        add("hierarchical_bootstrap_class_then_group", a, "aurc", theta,
            percentile_ci(dd(R_hier)), boot_p(dd(R_hier)),
            "Treats the 6 classes as sampled activities; crude with 6 classes.")
        sw = swap_randomization(hmdb, a, "msp", hmdb.franchise,
                                args.permutations, SEED)
        swap[a] = sw
        add("paired_swap_randomization_rank_scores (film groups)", a,
            "rank_aurc", sw["observed_rank_aurc_diff"], None, sw["p_two_sided"],
            "Heuristic exchangeability assumption; statistic on rank scores.")
        # clustered DeLong for pooled AUROC difference
        for gname, gl in (("50 groups", hmdb.group), ("film groups", hmdb.franchise)):
            A, cov = obuchowski(hmdb.y, [hmdb.s[a], hmdb.s["msp"]], gl)
            dlt = A[0] - A[1]
            se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
            z = dlt / se
            add(f"clustered_DeLong_Obuchowski ({gname})", a, "pooled_auroc",
                float(dlt), [dlt - 1.96 * se, dlt + 1.96 * se],
                float(2 * stats.norm.sf(abs(z))), f"z={z:.3f}, se={se:.4f}")
        # stratified AUROC difference
        ds = R[f"strat_auroc/{a}"] - R["strat_auroc/msp"]
        th_s = auroc(hmdb.y, hmdb.s[a], hmdb.cls) - auroc(hmdb.y, hmdb.s["msp"], hmdb.cls)
        add("cluster_bootstrap_stratified_percentile", a, "within_class_auroc",
            th_s, percentile_ci(ds), boot_p(ds), "Same-class pairs only.")
        js = jackknife_t(th_s, RJ_g[f"strat_auroc/{a}"] - RJ_g["strat_auroc/msp"])
        add("cluster_jackknife_t (50 groups)", a, "within_class_auroc", th_s,
            js["ci95"], js["p_two_sided"], f"df={js['df']}")

    # seed / replicate sensitivity (Phase 1 method)
    seed_rows = []
    grid = [int(x) for x in args.sensitivity_grid.split(",")]
    seeds = [int(x) for x in args.sensitivity_seeds.split(",")]
    for sd in seeds:
        for bb in grid:
            Cs = stratified_cluster_counts(hmdb, bb, sd)
            Rs = eh.replicate(Cs, prim_m)
            for a in ("guardian_o", "guardian_full"):
                dd = Rs[f"aurc/{a}"] - Rs["aurc/msp"]
                ci = percentile_ci(dd)
                seed_rows.append({"seed": sd, "replicates": bb,
                                  "comparison": f"{a}_minus_msp",
                                  "ci95_low": ci[0], "ci95_high": ci[1],
                                  "p_boot": boot_p(dd),
                                  "excludes_zero": bool(ci[1] < 0 or ci[0] > 0)})

    # ----------------------------------- C. decomposition between/within class
    perm = {}
    decomp_rows = []
    for d, eng in ((hmdb, eh), (ucf, Engine(ucf, thr))):
        mons = allm + ["constant", "class_oracle"]
        null = within_class_label_permutation(d, mons, args.permutations, SEED)
        perm[d.name] = null
        classes = np.unique(d.cls)
        cacc = np.array([d.y[d.cls == c].mean() for c in classes])
        obs_msp = aurc_expected(d.y, d.s["msp"])
        null_msp = null["msp"]
        for m in mons:
            obs = aurc_expected(d.y, d.s[m])
            cmean = np.array([d.s[m][d.cls == c].mean() for c in classes])
            rho = (stats.spearmanr(cmean, cacc) if np.std(cmean) > 0
                   else (np.nan, np.nan))
            dnull = null[m] - null_msp
            dobs = obs - obs_msp
            decomp_rows.append({
                "dataset": d.name, "monitor": m,
                "kind": MONITORS.get(m, (CONTROLS.get(m),))[0],
                "aurc": obs,
                "aurc_class_composition_only": float(null[m].mean()),
                "aurc_within_class_contribution": float(obs - null[m].mean()),
                "p_within_class_info": float(
                    (1 + np.sum(null[m] <= obs)) / (1 + len(null[m]))),
                "delta_aurc_vs_msp": dobs,
                "delta_aurc_vs_msp_expected_from_class_composition":
                    float(dnull.mean()),
                "delta_within_class_part": float(dobs - dnull.mean()),
                "pooled_auroc": auroc(d.y, d.s[m]),
                "within_class_auroc": auroc(d.y, d.s[m], d.cls),
                "spearman_classmean_score_vs_class_accuracy": float(rho[0]),
                "spearman_p": float(rho[1]),
                "n_classes": len(classes),
            })

    # LOCO reproduction
    loco = []
    for c in sorted(set(hmdb.cls)):
        k = hmdb.cls != c
        row = {"held_out_class": c}
        for m in prim_m:
            row[m] = aurc(hmdb.y[k], hmdb.s[m][k])
        row["guardian_o_minus_msp"] = row["guardian_o"] - row["msp"]
        row["guardian_full_minus_msp"] = row["guardian_full"] - row["msp"]
        loco.append(row)
    v2loco = {r["held_out_class"]: r for r in csv.DictReader(V2_LOCO.open(newline=""))}
    for row in loco:
        ref = v2loco[row["held_out_class"]]
        assert abs(row["guardian_full_minus_msp"]
                   - float(ref["aurc_guardian_full_minus_baseline"])) < 1e-12
        assert abs(row["guardian_o_minus_msp"]
                   - float(ref["aurc_guardian_occlusion_minus_baseline"])) < 1e-12
    checks["loco_reproduced"] = True

    # ------------------------------------------- D. calibration transfer
    calib_rows = []
    eu = Engine(ucf, thr)
    C_ucf = stratified_cluster_counts(ucf, B, SEED)
    RU = eu.replicate(C_ucf, allm)
    for d, RR in ((cal, None), (ucf, RU), (hmdb, R)):
        for m in allm:
            op = operating_point(d.y, d.s[m], thr[m])
            row = {"dataset": d.name, "monitor": m, "threshold": thr[m],
                   "threshold_source": ("frozen_models_v2.json" if m in
                                        ("msp", "guardian_o", "guardian_full")
                                        else "UCF101 calibration, frozen rule"),
                   **op,
                   "risk_reduction_vs_accept_all":
                       op["error_rate_accept_all"] - op["selective_risk"]}
            if RR is not None:
                for key, name in (("fixed_cov", "coverage"),
                                  ("fixed_risk", "selective_risk"),
                                  ("fixed_far", "false_alarm_rate"),
                                  ("fixed_miss", "miss_rate"),
                                  ("risk_reduction", "risk_reduction_vs_accept_all")):
                    lo, hi = percentile_ci(RR[f"{key}/{m}"])
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = lo, hi
                row["risk_reduction_p_boot"] = boot_p(RR[f"risk_reduction/{m}"])
            calib_rows.append(row)

    # ---------------------------------------------- E. baseline comparison
    base_rows = []
    pfam = {}
    for d, RR in ((ucf, RU), (hmdb, R)):
        for m in list(MONITORS) + list(CONTROLS):
            row = {"dataset": d.name, "monitor": m,
                   "kind": MONITORS[m][0] if m in MONITORS else CONTROLS[m],
                   "forward_passes": MONITORS[m][1] if m in MONITORS else None,
                   "mean_recorded_inference_ms":
                       float(d.ms[m].mean()) if m in MONITORS else None,
                   "aurc": aurc_expected(d.y, d.s[m]),
                   "eaurc": aurc_expected(d.y, d.s[m]) - aurc_optimal(d.y),
                   "pooled_auroc": auroc(d.y, d.s[m]),
                   "within_class_auroc": auroc(d.y, d.s[m], d.cls)}
            for r_ in (0.5, 0.7, 0.8):
                row[f"risk_at_{int(r_*100)}pct_coverage"] = (
                    risk_at(d.y, d.s[m], r_) if m in MONITORS else None)
            if m in MONITORS:
                for met in ("aurc", "auroc", "strat_auroc"):
                    lo, hi = percentile_ci(RR[f"{met}/{m}"])
                    row[f"{met}_ci_low"], row[f"{met}_ci_high"] = lo, hi
                th_r = aurc(d.y, d.s[m]) - float(np.mean(~d.y))
                row["delta_aurc_vs_random"] = th_r
                lo, hi = percentile_ci(RR[f"aurc_minus_random/{m}"])
                row["delta_aurc_vs_random_ci_low"] = lo
                row["delta_aurc_vs_random_ci_high"] = hi
                row["delta_aurc_vs_random_p_boot"] = boot_p(RR[f"aurc_minus_random/{m}"])
                if m != "msp":
                    for met in ("aurc", "strat_auroc"):
                        dd = RR[f"{met}/{m}"] - RR[f"{met}/msp"]
                        th = (aurc(d.y, d.s[m]) - aurc(d.y, d.s["msp"])
                              if met == "aurc" else
                              auroc(d.y, d.s[m], d.cls) - auroc(d.y, d.s["msp"], d.cls))
                        row[f"delta_{met}_vs_msp"] = th
                        lo, hi = percentile_ci(dd)
                        row[f"delta_{met}_vs_msp_ci_low"] = lo
                        row[f"delta_{met}_vs_msp_ci_high"] = hi
                        row[f"delta_{met}_vs_msp_p_boot"] = boot_p(dd)
                        pfam[(d.name, m, met)] = boot_p(dd)
                op = operating_point(d.y, d.s[m], thr[m])
                row.update({"ucf101_calibrated_threshold": thr[m],
                            "fixed_coverage": op["coverage"],
                            "fixed_selective_risk": op["selective_risk"],
                            "fixed_false_alarm_rate": op["false_alarm_rate"],
                            "fixed_miss_rate": op["miss_rate"]})
            base_rows.append(row)
    for dname in ("ucf101", "hmdb51"):
        for met in ("aurc", "strat_auroc"):
            fam = {k: v for k, v in pfam.items() if k[0] == dname and k[2] == met}
            for k, v in holm(fam).items():
                for row in base_rows:
                    if row["dataset"] == k[0] and row["monitor"] == k[1]:
                        row[f"delta_{met}_vs_msp_holm_within_dataset"] = v

    # ------------------------------------------------------------ outputs
    within_subset, other_classes = cross_class_sources()
    group_audit = {
        "class_specific_groups": len(set(hmdb.group)),
        "class_agnostic_film_groups_no_franchise_merge": len(set(hmdb.film)),
        "class_agnostic_groups_franchise_merge": len(set(hmdb.franchise)),
        "prefixes_spanning_classes_within_subset": within_subset,
        "film_merges": {f: sorted(set(hmdb.group[hmdb.franchise == f]))
                        for f in sorted(set(hmdb.franchise))
                        if len(set(hmdb.group[hmdb.franchise == f])) > 1},
        "subset_prefixes_also_in_other_hmdb51_classes": other_classes,
        "group_sizes": sorted(np.unique(hmdb.group, return_counts=True)[1].tolist()),
        "official_split1_counts": split_info,
    }
    per_class = []
    for c in sorted(set(hmdb.cls)):
        k = hmdb.cls == c
        preds = defaultdict(int)
        for r in np.array(hmdb.recs)[k]:
            preds[r["pred"]] += 1
        per_class.append({
            "class": c, "accuracy": float(hmdb.y[k].mean()),
            "top_predictions": sorted(preds.items(), key=lambda x: -x[1])[:4],
            **{f"within_auroc/{m}": auroc(hmdb.y[k], hmdb.s[m][k]) for m in allm},
            **{f"mean_score/{m}": float(hmdb.s[m][k].mean()) for m in prim_m}})

    results = {
        "scope": ("6-class, 180-clip HMDB51 subset (split-1 test) and 505-video "
                  "UCF101 evaluation split. Not a 51-class HMDB51 evaluation."),
        "checks": checks,
        "phase1_reproduction": repro,
        "swap_randomization": swap,
        "group_audit": group_audit,
        "hmdb51_per_class": per_class,
        "leave_one_class_out": loco,
        "permutations": args.permutations,
        "bootstrap": {"replicates": B, "seed": SEED},
        "thresholds_used": thr,
        "hmdb51_labels_used_for_thresholds_or_training": False,
    }
    (out / "audit_results.json").open("x").write(
        json.dumps(clean(results), indent=2) + "\n")
    write_csv(out / "baseline_comparison.csv", clean(base_rows))
    write_csv(out / "inference_sensitivity.csv", clean(sens_rows))
    write_csv(out / "seed_sensitivity.csv", clean(seed_rows))
    write_csv(out / "decomposition.csv", clean(decomp_rows))
    write_csv(out / "calibration_transfer.csv", clean(calib_rows))

    after = {p: sha256(p) for p in INPUTS}
    assert after == in_hash, "inputs changed during run"
    docs = ["AUDIT_REPORT.md", "PHASE2_RESEARCH_PLAN.md", "RESEARCH_GAPS.md",
            "statistical_audit.py", "test_statistical_audit.py"]
    manifest = {
        "command": " ".join(["python"] + [str(a) for a in (argv if argv is not None else sys.argv)]),
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(),
        "inputs_sha256": {str(p.relative_to(ROOT)): h for p, h in in_hash.items()},
        "inputs_unchanged_after_run": True,
        "frozen_model_sha256_verified": True,
        "seeds": {"bootstrap": SEED, "permutation": SEED,
                  "sensitivity_seeds": seeds, "sensitivity_grid": grid},
        "bootstrap_replicates": B, "permutations": args.permutations,
        "outputs": targets[:-1],
        "outputs_sha256": {t: sha256(out / t) for t in targets[:-1]},
        "documents_sha256_at_run_time": {
            d_: sha256(Path(__file__).parent / d_) for d_ in docs
            if (Path(__file__).parent / d_).exists()},
        "models_retrained": False,
        "hmdb51_used_for_threshold_selection": False,
    }
    (out / "run_manifest.json").open("x").write(json.dumps(manifest, indent=2) + "\n")
    print("Audit complete ->", out)
    return results


if __name__ == "__main__":
    main()
