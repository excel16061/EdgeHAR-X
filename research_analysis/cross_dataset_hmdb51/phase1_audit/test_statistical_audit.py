"""Tests for statistical_audit.py: metric definitions, edge cases,
resampling correctness, and reproduction of Phase 1 numbers."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import statistical_audit as A  # noqa: E402


# ------------------------------------------------------------ AURC

def test_aurc_hand_computed():
    y = np.array([True, False, True])
    s = np.array([0.9, 0.5, 0.1])
    assert A.aurc(y, s) == pytest.approx((0 + 1 / 2 + 1 / 3) / 3)


def test_aurc_perfect_and_worst():
    y = np.array([True, True, False, False])
    best = A.aurc(y, np.array([4, 3, 2, 1.0]))
    worst = A.aurc(y, np.array([1, 2, 3, 4.0]))
    assert best == pytest.approx(A.aurc_optimal(y))
    assert best < A.aurc_expected(y, np.zeros(4)) < worst


def test_constant_score_expected_aurc_equals_error_rate():
    rng = np.random.default_rng(0)
    y = rng.random(57) < 0.7
    assert A.aurc_expected(y, np.zeros(57)) == pytest.approx(np.mean(~y))


def test_tie_aware_matches_bruteforce_average():
    rng = np.random.default_rng(1)
    y = rng.random(8) < 0.5
    s = np.array([3, 3, 3, 2, 2, 1, 1, 1.0])
    vals = []
    for _ in range(20000):
        jitter = s + rng.random(8) * 1e-6
        vals.append(A.aurc(y, jitter))
    assert A.aurc_expected(y, s) == pytest.approx(np.mean(vals), abs=2e-3)


def test_stable_equals_expected_without_ties():
    rng = np.random.default_rng(2)
    y = rng.random(100) < 0.6
    s = rng.random(100)
    assert A.aurc(y, s) == pytest.approx(A.aurc_expected(y, s))


def test_aurc_counts_equals_explicit_resample():
    rng = np.random.default_rng(3)
    y = rng.random(40) < 0.6
    s = rng.random(40)
    idx = rng.integers(0, 40, 40)
    counts = np.bincount(idx, minlength=40)
    assert A.aurc(y, s, counts) == pytest.approx(A.aurc(y[idx], s[idx]))


def test_aurc_invariant_to_monotone_transform():
    rng = np.random.default_rng(4)
    y = rng.random(60) < 0.5
    s = rng.random(60)
    assert A.aurc(y, s) == pytest.approx(A.aurc(y, np.exp(3 * s)))


# ----------------------------------------------------------- AUROC

def test_auroc_edge_cases():
    y = np.array([True, True, False, False])
    assert A.auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert A.auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert A.auroc(y, np.full(4, 0.3)) == 0.5
    assert np.isnan(A.auroc(np.array([True, True]), np.array([0.1, 0.2])))


def test_auroc_matches_mann_whitney():
    rng = np.random.default_rng(5)
    y = rng.random(80) < 0.6
    s = rng.normal(size=80) + y
    U = stats.mannwhitneyu(s[y], s[~y]).statistic
    assert A.auroc(y, s) == pytest.approx(U / (y.sum() * (~y).sum()))


def test_within_class_auroc_excludes_between_class_pairs():
    y = np.array([True, True, False, False])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    cls = np.array(["A", "A", "B", "B"])
    assert A.auroc(y, s) == 1.0
    assert np.isnan(A.auroc(y, s, cls))
    y2 = np.array([True, False, True, False])
    s2 = np.array([0.6, 0.7, 0.1, 0.2])
    assert A.auroc(y2, s2, np.array(["A", "A", "B", "B"])) == 0.0


def test_weighted_auc_counts_equals_resample():
    rng = np.random.default_rng(6)
    y = rng.random(30) < 0.5
    s = rng.random(30)
    idx = rng.integers(0, 30, 30)
    c = np.bincount(idx, minlength=30)
    P, Q = A.pairs(y)
    got = A.weighted_auc(A.psi(s, P, Q), c[P] * c[Q])
    assert got == pytest.approx(A.auroc(y[idx], s[idx]))


# --------------------------------------------- Obuchowski / DeLong

def _delong_reference(y, a, b):
    """Textbook DeLong (Sun & Xu 2014 formulation) for cross-check."""
    def comps(s):
        pos, neg = s[y], s[~y]
        P = (pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])
        return P.mean(), P.mean(1), P.mean(0)
    Aa, Va10, Va01 = comps(a)
    Ab, Vb10, Vb01 = comps(b)
    S10 = np.cov(np.vstack([Va10, Vb10]))
    S01 = np.cov(np.vstack([Va01, Vb01]))
    cov = S10 / y.sum() + S01 / (~y).sum()
    return np.array([Aa, Ab]), cov


def test_obuchowski_reduces_to_delong_with_singleton_clusters():
    rng = np.random.default_rng(7)
    y = rng.random(70) < 0.55
    a = rng.normal(size=70) + y
    b = rng.normal(size=70) + 0.5 * y
    A1, C1 = A.delong_unclustered(y, a, b)
    A2, C2 = _delong_reference(y, a, b)
    assert np.allclose(A1, A2) and np.allclose(C1, C2)


def test_obuchowski_variance_grows_with_duplicated_clusters():
    rng = np.random.default_rng(8)
    y = rng.random(40) < 0.5
    a = rng.normal(size=40) + y
    b = rng.normal(size=40)
    # duplicate every unit 3x inside its own cluster: correct variance should
    # not shrink like an i.i.d. sample of 120 would
    y3, a3, b3 = np.repeat(y, 3), np.repeat(a, 3), np.repeat(b, 3)
    g3 = np.repeat(np.arange(40), 3).astype(str)
    _, C_clu = A.obuchowski(y3, [a3, b3], g3)
    _, C_iid = A.obuchowski(y3, [a3, b3], np.arange(120).astype(str))
    var = lambda C: C[0, 0] + C[1, 1] - 2 * C[0, 1]
    assert var(C_clu) > 2 * var(C_iid)


# --------------------------------------------------- inference tools

def test_holm():
    adj = A.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone step-down
    # step-down monotonicity: y adj = max(min(1, 2*0.6), 0.7) = 1.0
    assert A.holm({"x": 0.6, "y": 0.7})["y"] == pytest.approx(1.0)


def test_boot_p_and_percentile():
    v = np.r_[np.full(975, -1.0), np.full(25, 1.0)]
    assert A.boot_p(v) == pytest.approx(0.05)
    assert A.boot_p(np.full(10, -1.0)) == 0.0
    v2 = np.r_[v, np.nan]
    assert A.percentile_ci(v2) == A.percentile_ci(v)


def test_jackknife_t_mean_matches_classical_se():
    rng = np.random.default_rng(9)
    x = rng.normal(size=25)
    jack = np.array([np.delete(x, i).mean() for i in range(25)])
    jt = A.jackknife_t(x.mean(), jack)
    assert jt["se"] == pytest.approx(x.std(ddof=1) / np.sqrt(25))


def test_bca_reduces_to_percentile_when_symmetric():
    rng = np.random.default_rng(10)
    boot = rng.normal(size=200000)
    jack = np.linspace(-1, 1, 51)  # symmetric -> acceleration 0
    ci, z0, a = A.bca_ci(0.0, boot, jack)
    assert abs(a) < 1e-12 and abs(z0) < 0.01
    assert np.allclose(ci, np.percentile(boot, [2.5, 97.5]), atol=0.02)


def test_threshold_rule_and_tie_guard():
    s = np.linspace(0, 1, 10)
    t = A.threshold_rule(s)
    assert np.mean(s >= t) == pytest.approx(0.7)
    with pytest.raises(ValueError):
        A.threshold_rule(np.r_[np.ones(8), np.zeros(2)])


def test_operating_point_counts():
    y = np.array([True, False, True, False])
    s = np.array([0.9, 0.8, 0.1, 0.05])
    op = A.operating_point(y, s, 0.5)
    assert op["coverage"] == 0.5 and op["selective_risk"] == 0.5
    assert op["false_alarm_rate"] == 0.5 and op["miss_rate"] == 0.5
    op2 = A.operating_point(y, s, 0.5, counts=np.array([2, 0, 1, 1]))
    assert op2["selective_risk"] == 0.0 and op2["miss_rate"] == 0.0


def test_film_group_rule():
    assert A.film_group("Return_of_the_King_4") == "Lord_of_the_Rings"
    assert A.film_group("Two_Towers_3") == "Lord_of_the_Rings"
    assert A.film_group("Return_of_the_King_4", franchise=False) == "Return_of_the_King"
    assert A.film_group("Glory") == "Glory"
    assert A.film_group("35_pull_ups") == "35_pull_ups"
    assert A.film_group("Rapier_at_Clinton_2007") == "Rapier_at_Clinton"


# ----------------------------------------- real-data integration tests

@pytest.fixture(scope="module")
def data():
    fm = json.loads(A.FROZEN.read_text())["models"]
    hmdb = A.Data(A.build(A.load_grouped(A.HMDB_PRED, 5), fm, "hmdb51"), "hmdb51")
    return fm, hmdb


def test_frozen_hash_unchanged():
    assert A.sha256(A.FROZEN) == A.EXPECTED_SHA256


def test_manifest_official_split():
    info = A.verify_hmdb_manifest()
    assert all(v["test_clips_split1"] == 30 for v in info.values())


def test_group_structure(data):
    _, h = data
    assert len(set(h.group)) == 50
    assert len(set(h.film)) == 47
    assert len(set(h.franchise)) == 46
    # stratified resampling never moves a clip across classes
    C = A.stratified_cluster_counts(h, 50, 0)
    for c in set(h.cls):
        m = h.cls == c
        n_groups = len(set(h.group[m]))
        for row in C[:5]:
            # number of group draws in class equals number of groups
            drawn = sum(row[(h.group == g)].max() if row[(h.group == g)].size else 0
                        for g in set(h.group[m]))
            assert drawn == n_groups


def test_cluster_counts_constant_within_group(data):
    _, h = data
    for C in (A.stratified_cluster_counts(h, 20, 1),
              A.unstratified_cluster_counts(h.franchise, 20, 1),
              A.hierarchical_counts(h, 20, 1)):
        for g in set(h.group):
            m = h.group == g
            assert np.all(C[:, m] == C[:, m][:, :1])


def test_phase1_point_estimates(data):
    _, h = data
    assert A.aurc(h.y, h.s["msp"]) == pytest.approx(0.362333, abs=1e-6)
    assert A.aurc(h.y, h.s["guardian_o"]) == pytest.approx(0.358934, abs=1e-6)
    assert A.aurc(h.y, h.s["guardian_full"]) == pytest.approx(0.293768, abs=1e-6)
    assert A.auroc(h.y, h.s["msp"], h.cls) == pytest.approx(0.6007, abs=1e-4)


def test_phase1_bootstrap_reproduced_exactly(data):
    fm, h = data
    thr = {"msp": fm["baseline"]["threshold"]}
    eng = A.Engine(h, thr)
    C = A.stratified_cluster_counts(h, 10000, 2027)
    R = eng.replicate(C, ["msp", "guardian_full"])
    d = R["aurc/guardian_full"] - R["aurc/msp"]
    v2 = json.loads(A.V2_RESULTS.read_text())
    ref = v2["comparisons"]["guardian_full_minus_baseline"]["aurc"]
    assert np.allclose(A.percentile_ci(d), ref["ci95"], atol=1e-12)
    assert A.boot_p(d) == pytest.approx(0.024)
    assert int(np.sum(d >= 0)) == 120


def test_end_to_end_and_no_overwrite(tmp_path):
    A.main(["--output-dir", str(tmp_path), "--bootstraps", "200",
            "--permutations", "100", "--sensitivity-grid", "100",
            "--sensitivity-seeds", "1"])
    assert (tmp_path / "baseline_comparison.csv").exists()
    with pytest.raises(FileExistsError):
        A.main(["--output-dir", str(tmp_path), "--bootstraps", "10"])
    assert A.sha256(A.FROZEN) == A.EXPECTED_SHA256
