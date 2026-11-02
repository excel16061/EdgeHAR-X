"""Phase 2A metrics library: confidence scores from full logits, the
preregistered monitors, within-class / pooled reliability metrics and
group-aware statistical inference.

All monitors are label-free functions of exported logits (plus the frozen
Guardian V3 weights). Nothing here fits anything on evaluation labels.
Definitions are frozen by PREREGISTRATION.md; any change after Phase 2B
starts must be logged as a deviation.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/guardian_v3_official/frozen_models_v2.json"
FROZEN_SHA256 = "e56fececf1f980a2eafbc9fc882ac4157b3829802a8c492b9a31994db1022cd1"

GUARDIAN_PROBES = ("blur3", "dark3", "occlusion3", "frame_drop3")
TEMPORAL_VIEWS = ("clean_uniform", "seg4_0", "seg4_1", "seg4_2", "seg4_3")
SEG2_VIEWS = ("seg2_0", "seg2_1")


# ============================================================ scores

def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logit_scores(z):
    """Confidence scores (higher = more confident) from logits [..., C]."""
    z = np.asarray(z, dtype=np.float64)
    p = softmax(z)
    ps = np.sort(p, axis=-1)
    lse = np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1)) + z.max(-1)
    return {
        "msp": ps[..., -1],
        "neg_entropy": np.sum(p * np.log(np.clip(p, 1e-300, None)), axis=-1),
        "margin": ps[..., -1] - ps[..., -2],
        "energy": lse,  # negative free energy (Liu et al. 2020 form, T=1)
        "max_logit": z.max(axis=-1),
    }


def js_divergence(p, q):
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m), -1) + 0.5 * np.sum(q * np.log(q / m), -1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def guardian_scores(view_logits, frozen=None):
    """Frozen Guardian-O / Guardian-Full from the same five views used in
    Phase 1 (definitions identical to guardian_v3_official_train)."""
    if frozen is None:
        frozen = json.loads(FROZEN.read_text())["models"]
    pc = softmax(view_logits["clean_adaptive"])
    s, yhat = pc.max(), int(pc.argmax())
    probes = [softmax(view_logits[v]) for v in GUARDIAN_PROBES]
    ps = np.array([p.max() for p in probes])
    changes = [int(p.argmax()) != yhat for p in probes]
    drops = s - ps
    full = np.array([s, drops.mean(), drops.max(), np.mean(changes), ps.std()])
    occ = np.array([s, float(int(probes[2].argmax()) != yhat)])

    def score(m, x):
        w, mu, sd = (np.asarray(m[k], float) for k in ("weights", "feature_mean", "feature_std"))
        return float(_sigmoid(w[0] + ((x - mu) / sd) @ w[1:]))

    return {"guardian_o": score(frozen["guardian_occlusion"], occ),
            "guardian_full": score(frozen["guardian_full"], full)}


def monitor_scores(view_logits, frozen=None):
    """All preregistered monitors for one clip. view_logits: view_id -> [C].
    Primary temporal monitor TG0 = mean predicted-class probability over the
    five temporal views (uniform resample + 4 segments), where the predicted
    class is the clean adaptive top-1."""
    zc = view_logits["clean_adaptive"]
    pc = softmax(zc)
    yhat = int(pc.argmax())
    out = {k: float(v) for k, v in logit_scores(zc).items()}
    if all(v in view_logits for v in GUARDIAN_PROBES):
        out.update(guardian_scores(view_logits, frozen))
        pg = [softmax(view_logits[v]) for v in GUARDIAN_PROBES]
        out["corruption_tta_pred_prob"] = float(np.mean([p[yhat] for p in pg]))
        out["corruption_agreement_then_msp"] = float(
            2 * sum(int(p.argmax()) == yhat for p in pg) + pc.max())
    if all(v in view_logits for v in TEMPORAL_VIEWS):
        pt = [softmax(view_logits[v]) for v in TEMPORAL_VIEWS]
        out["tg0"] = float(np.mean([p[yhat] for p in pt]))
        out["tg_min"] = float(np.min([p[yhat] for p in pt]))
        out["tg_agreement_then_msp"] = float(
            2 * sum(int(p.argmax()) == yhat for p in pt) + pc.max())
        out["tg_neg_js"] = float(-np.mean([js_divergence(p, pc) for p in pt]))
        out["tg_sampling_pred_prob"] = float(pt[0][yhat])  # uniform vs adaptive
    if all(v in view_logits for v in SEG2_VIEWS):
        out["tg_seg2"] = float(np.mean([softmax(view_logits[v])[yhat] for v in SEG2_VIEWS]))
    out["_yhat"] = yhat
    return out


def control_scores(y, cls, msp, seed_random=20273, seed_shuffle=20274):
    """Preregistered negative controls. `class_oracle` uses labels and is a
    DIAGNOSTIC only (upper bound of pure between-class ranking)."""
    y, cls, msp = np.asarray(y, bool), np.asarray(cls), np.asarray(msp, float)
    shuffled = msp.copy()
    rng = np.random.default_rng(seed_shuffle)
    for c in np.unique(cls):
        ix = np.where(cls == c)[0]
        shuffled[ix] = msp[rng.permutation(ix)]
    acc = {c: y[cls == c].mean() for c in np.unique(cls)}
    return {"constant": np.zeros(len(y)),
            "random": np.random.default_rng(seed_random).random(len(y)),
            "within_class_shuffled_msp": shuffled,
            "class_oracle": np.array([acc[c] for c in cls])}


def correctness(pred_label, mapped_labels, strict_label):
    return {"correct_mapped": pred_label in mapped_labels,
            "correct_strict": pred_label == strict_label}


# =========================================================== metrics

def auroc(y, s, cls=None, w=None):
    """P(score_correct > score_incorrect), ties 1/2. With `cls`: only
    same-class pairs (pair-weighted within-class AUROC). `w`: per-sample
    multiplicities (bootstrap). Returns nan if no valid pairs."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    groups = [np.arange(len(y))] if cls is None else [np.where(cls == c)[0] for c in np.unique(cls)]
    num = den = 0.0
    for ix in groups:
        pos, neg = ix[y[ix]], ix[~y[ix]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        # vectorized weighted Mann-Whitney, O(n log n)
        order = np.argsort(s[neg], kind="mergesort")
        nv = s[neg][order]
        cw = np.r_[0.0, np.cumsum(w[neg][order])]
        lo = np.searchsorted(nv, s[pos], side="left")
        hi = np.searchsorted(nv, s[pos], side="right")
        num += float(np.sum(w[pos] * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))))
        den += float(w[pos].sum() * w[neg].sum())
    return num / den if den > 0 else float("nan")


def macro_within_class_auroc(y, s, cls, min_each=3):
    """Unweighted mean of per-class AUROC over classes with >= min_each
    correct and >= min_each incorrect predictions."""
    vals = []
    for c in np.unique(cls):
        m = cls == c
        if np.sum(y[m]) >= min_each and np.sum(~y[m]) >= min_each:
            vals.append(auroc(y[m], s[m]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def _ranked_errors(y, s, w=None):
    order = np.argsort(-np.asarray(s, float), kind="stable")
    e = ~np.asarray(y, bool)[order]
    if w is not None:
        e = np.repeat(e, np.asarray(w, int)[order])
    return e


def aurc(y, s, w=None):
    """Mean selective risk over accepted-set sizes 1..N (no ties expected;
    ties broken by stable order — use aurc_tie_aware for tied scores)."""
    e = _ranked_errors(y, s, w)
    return float(np.mean(np.cumsum(e) / np.arange(1, len(e) + 1)))


def aurc_tie_aware(y, s):
    order = np.argsort(-np.asarray(s, float), kind="stable")
    ss, e = np.asarray(s, float)[order], (~np.asarray(y, bool)[order]).astype(float)
    n, i, E = len(ss), 0, 0.0
    cum = np.empty(n)
    while i < n:
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        cum[i:j + 1] = E + e[i:j + 1].mean() * np.arange(1, j - i + 2)
        E += e[i:j + 1].sum()
        i = j + 1
    return float(np.mean(cum / np.arange(1, n + 1)))


def augrc(y, s, w=None):
    """Area under the generalized risk-coverage curve: mean over k of
    (#accepted errors in top-k)/N (our reading of Traub et al., NeurIPS
    2024; verify against the paper's reference implementation)."""
    e = _ranked_errors(y, s, w)
    return float(np.mean(np.cumsum(e) / len(e)))


def eaurc(y, s):
    y = np.asarray(y, bool)
    return aurc(y, s) - aurc(y, y.astype(float))


def risk_at_coverage(y, s, coverage):
    order = np.argsort(-np.asarray(s, float), kind="stable")
    k = max(1, int(round(coverage * len(y))))
    return float(np.mean(~np.asarray(y, bool)[order[:k]]))


def threshold_rule(scores, target=0.70):
    """Frozen Guardian V3 rule: midpoint between k-th and (k+1)-th score,
    k = round(target*N), applied to *calibration* scores only."""
    o = np.sort(np.asarray(scores, float))[::-1]
    k = round(len(o) * target)
    if o[k - 1] == o[k]:
        raise ValueError("tied threshold boundary")
    return float((o[k - 1] + o[k]) / 2)


def operating_point(y, s, t):
    y, acc = np.asarray(y, bool), np.asarray(s, float) >= t
    return {"coverage": float(acc.mean()),
            "selective_risk": float(np.mean(~y[acc])) if acc.any() else float("nan"),
            "false_alarm_rate": float(np.mean(~acc[y])) if y.any() else float("nan"),
            "miss_rate": float(np.mean(acc[~y])) if (~y).any() else float("nan")}


# ========================================================= inference

def group_index(groups):
    keys = sorted(set(groups))
    lut = {k: i for i, k in enumerate(keys)}
    return np.array([lut[g] for g in groups]), len(keys)


def cluster_bootstrap_weights(groups, B, seed):
    """Primary: resample class-agnostic film groups with replacement
    (unstratified, handles groups spanning classes). Returns [B, n] ints."""
    gidx, G = group_index(groups)
    rng = np.random.default_rng(seed)
    W = np.empty((B, len(gidx)), dtype=np.int32)
    for b in range(B):
        gc = np.bincount(rng.integers(0, G, G), minlength=G)
        W[b] = gc[gidx]
    return W


def hierarchical_bootstrap_weights(cls, groups, B, seed):
    """Generalization check: classes with replacement, then (class, group)
    units within each drawn class."""
    classes = sorted(set(cls))
    units = defaultdict(list)
    for i, (c, g) in enumerate(zip(cls, groups)):
        units[(c, g)].append(i)
    by_class = defaultdict(list)
    for (c, g) in sorted(units):
        by_class[c].append((c, g))
    rng = np.random.default_rng(seed)
    W = np.zeros((B, len(cls)), dtype=np.int32)
    for b in range(B):
        for ci in rng.integers(0, len(classes), len(classes)):
            keys = by_class[classes[ci]]
            for j in rng.integers(0, len(keys), len(keys)):
                W[b, units[keys[j]]] += 1
    return W


def jackknife_weights(groups):
    gidx, G = group_index(groups)
    return np.array([(gidx != g).astype(np.int32) for g in range(G)])


def percentile_ci(v, level=0.95):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    a = (1 - level) / 2
    return [float(np.percentile(v, 100 * a)), float(np.percentile(v, 100 * (1 - a)))]


def bootstrap_p(v, null=0.0):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return float(min(1.0, 2 * min(np.mean(v <= null), np.mean(v >= null))))


def jackknife_t(theta, jack, null=0.0):
    jack = np.asarray(jack, float)
    G = len(jack)
    se = float(np.sqrt((G - 1) / G * np.sum((jack - jack.mean()) ** 2)))
    t = (theta - null) / se if se > 0 else np.inf
    q = stats.t.ppf(0.975, G - 1)
    return {"se": se, "ci95": [theta - q * se, theta + q * se],
            "p": float(2 * stats.t.sf(abs(t), G - 1)), "df": G - 1}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (len(names) - i) * pvals[k]))
        out[k] = run
    return out


def tost_equivalent(boot_diffs, margin):
    """Equivalence (two one-sided tests at alpha=0.05) declared iff the
    90% percentile interval lies inside (-margin, +margin)."""
    lo, hi = percentile_ci(boot_diffs, 0.90)
    return {"ci90": [lo, hi], "equivalent": bool(-margin < lo and hi < margin)}


def within_class_permutation_null(y, s, cls, P, seed, metric=aurc):
    """Correctness permuted within class; scores fixed. Mean = metric value
    attributable to class composition alone."""
    rng = np.random.default_rng(seed)
    ix = [np.where(cls == c)[0] for c in np.unique(cls)]
    out = np.empty(P)
    for p in range(P):
        yp = y.copy()
        for i in ix:
            yp[i] = y[rng.permutation(i)]
        out[p] = metric(yp, s)
    return out


def compare(y, cls, groups, sa, sb, B=10000, seed=20270, margin=0.03, hierarchical=True):
    """Primary paired comparison of two monitors: within-class AUROC
    difference (a - b) with cluster bootstrap CI, bootstrap p, cluster
    jackknife, TOST equivalence and (optionally) hierarchical CI."""
    y = np.asarray(y, bool)
    theta = auroc(y, sa, cls) - auroc(y, sb, cls)
    W = cluster_bootstrap_weights(groups, B, seed)
    d = np.array([auroc(y, sa, cls, w) - auroc(y, sb, cls, w) for w in W])
    J = jackknife_weights(groups)
    dj = np.array([auroc(y, sa, cls, w) - auroc(y, sb, cls, w) for w in J])
    res = {"delta_within_auroc": theta, "ci95_cluster": percentile_ci(d),
           "p_cluster": bootstrap_p(d), "jackknife": jackknife_t(theta, dj),
           "tost": tost_equivalent(d, margin), "B": B, "seed": seed}
    if hierarchical:
        Wh = hierarchical_bootstrap_weights(cls, groups, B, seed + 1)
        dh = np.array([auroc(y, sa, cls, w) - auroc(y, sb, cls, w) for w in Wh])
        res["ci95_hierarchical"] = percentile_ci(dh)
        res["p_hierarchical"] = bootstrap_p(dh)
    return res


def one_sample_vs_chance(y, cls, groups, s, B=10000, seed=20271):
    """Within-class AUROC of one monitor vs 0.5 (cluster bootstrap)."""
    y = np.asarray(y, bool)
    theta = auroc(y, s, cls)
    W = cluster_bootstrap_weights(groups, B, seed)
    v = np.array([auroc(y, s, cls, w) for w in W])
    return {"within_auroc": theta, "ci95_cluster": percentile_ci(v),
            "p_vs_0.5": bootstrap_p(v, 0.5)}


# ===================================================== synthetic data

def synthetic_benchmark(n_classes=12, per_class=70, groups_per_class=25,
                        between=1.0, within=0.0, seed=0):
    """Synthetic monitor scores with controlled between-class signal
    (class mean score tracks class accuracy) and within-class signal
    (score shift for correct vs incorrect inside a class). Used in tests
    and power simulations; never mixed with real results."""
    rng = np.random.default_rng(seed)
    acc = np.linspace(0.3, 0.95, n_classes)
    cls, grp, y, s = [], [], [], []
    for c in range(n_classes):
        for i in range(per_class):
            g = f"c{c}_g{i % groups_per_class}"
            yi = rng.random() < acc[c]
            cls.append(f"class{c}")
            grp.append(g)
            y.append(yi)
            s.append(between * acc[c] + within * (1.0 if yi else 0.0) + rng.normal(0, 0.3))
    return np.array(y), np.array(s), np.array(cls), np.array(grp)
