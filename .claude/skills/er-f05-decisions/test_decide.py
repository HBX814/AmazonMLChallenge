"""pytest tests for decide.py  (run: python -m pytest -q test_decide.py)"""
import itertools
import os
import sys

import numpy as np
import polars as pl
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decide as D  # noqa: E402
import metric as M  # noqa: E402


def rand_bg(rng, R):
    lam = rng.uniform(0, 1.5)
    return D.poisson_pmf_trunc(np.array([lam]), R)[0]


def test_exact_matches_brute_force():
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = rng.integers(1, 8)
        p = np.sort(rng.beta(0.6, 0.6, size=n))[::-1]
        bg = rand_bg(rng, 5) if trial % 2 else np.array([1.0])
        EF = D.expected_f_topm(p[None, :], bg[None, :])[0]
        for m in range(n + 1):
            bf = D.brute_force_expected_f(p, range(m), bg)
            assert abs(EF[m] - bf) < 1e-10, (trial, m, EF[m], bf)


def test_top_m_is_optimal_among_all_subsets():
    """Under independence the optimal set is always a top-m set (probability ranking principle)."""
    rng = np.random.default_rng(1)
    for trial in range(40):
        n = rng.integers(2, 8)
        p = np.sort(rng.uniform(0, 1, size=n))[::-1]
        bg = rand_bg(rng, 4)
        best_any = max(D.brute_force_expected_f(p, S, bg)
                       for r in range(n + 1) for S in itertools.combinations(range(n), r))
        best_top = D.expected_f_topm(p[None, :], bg[None, :])[0].max()
        assert best_top >= best_any - 1e-12


def test_padding_and_tail_folding():
    rng = np.random.default_rng(2)
    p = np.sort(np.concatenate([rng.uniform(0.5, 1, 4), rng.uniform(0, 0.05, 20)]))[::-1]
    full = D.expected_f_topm(p[None, :], np.array([[1.0, 0, 0, 0]]))[0]
    padded = D.expected_f_topm(np.concatenate([p, np.zeros(5)])[None, :], np.array([[1.0, 0, 0, 0]]))[0]
    assert np.allclose(full, padded[:len(full)])
    L = 8  # fold the rest into a Poisson background
    bg = D.poisson_pmf_trunc(np.array([p[L:].sum()]), 12)
    folded = D.expected_f_topm(p[None, :L], bg)[0]
    assert np.abs(folded - full[:L + 1]).max() < 2e-3
    assert folded.argmax() == full.argmax()


def test_approximations_close_to_exact():
    rng = np.random.default_rng(3)
    B, L = 5000, 12
    P = np.sort(rng.beta(0.4, 0.8, size=(B, L)), axis=1)[:, ::-1]
    lam = rng.uniform(0, 0.3, B)
    EF = D.expected_f_topm(P, D.poisson_pmf_trunc(lam, 12))
    ES = D.expected_f_approx(P, lam, kind="series")
    EP = D.expected_f_approx(P, lam, kind="plugin")
    assert np.abs(ES - EF).max() < 0.03
    agree_s = (ES.argmax(1) == EF.argmax(1)).mean()
    agree_p = (EP.argmax(1) == EF.argmax(1)).mean()
    loss_s = (EF.max(1) - EF[np.arange(B), ES.argmax(1)]).mean()
    loss_p = (EF.max(1) - EF[np.arange(B), EP.argmax(1)]).mean()
    print(f"agree series={agree_s:.3f} plugin={agree_p:.3f}  E-loss series={loss_s:.5f} plugin={loss_p:.5f}")
    assert agree_s > 0.9 and loss_s < 2e-3 and loss_s <= loss_p


def test_simple_decisions():
    ed = pl.DataFrame({"s1": ["A", "B", "C", "C", "C"], "cand": ["a", "b", "c1", "c2", "c3"],
                       "p": [0.9, 0.02, 0.95, 0.9, 0.1]})
    for method in ("exact", "series", "plugin"):
        d = D.decide(ed, method, lam_missing=0.05)
        sel = set(d.filter(pl.col("sel"))["cand"].to_list())
        assert sel == {"a", "c1", "c2"}, (method, sel)


def test_fast_equals_numpy_approx():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(300):
        n = rng.integers(1, 10)
        for j in range(n):
            rows.append((f"S1-{i}", f"S2-{i}-{j}", float(rng.beta(0.5, 0.8))))
    ed = pl.DataFrame(rows, schema=["s1", "cand", "p"], orient="row")
    d = D.decide_fast(ed, lam_missing=0.1, kind="series")
    e = D.add_rank(ed)
    P, tail, nval, code, rank = D._padded(e, 16)
    EF = D.expected_f_approx(P, 0.1 + tail, kind="series")
    m = D.best_m(EF, 1.0, nval)
    assert (d["sel"].to_numpy() == (rank < m[code])).all()


def test_exclusivity_soft_and_hard():
    ed = pl.DataFrame({"s1": ["A", "B", "C", "A"], "cand": ["x", "x", "x", "y"], "p": [0.8, 0.6, 0.1, 0.7]})
    s = D.exclusivity_soft(ed)
    tot = s.group_by("cand").agg(pl.col("p").sum())
    assert (tot["p"] <= 1 + 1e-9).all()
    assert abs(s.filter(pl.col("cand") == "y")["p"][0] - 0.7) < 1e-9  # lone edge unchanged
    h = D.exclusivity_hard(ed)
    assert h.filter(pl.col("cand") == "x")["s1"].to_list() == ["A"]
    h2 = D.exclusivity_hard(ed, margin=0.3)  # 0.8-0.6 < 0.3 -> x dropped entirely
    assert h2.filter(pl.col("cand") == "x").height == 0


def test_resolve_exclusive_and_writer(tmp_path):
    rng = np.random.default_rng(5)
    rows = []
    for i in range(400):
        for j in range(rng.integers(0, 8)):
            rows.append((f"S1-{i}", f"S{2 + (j % 2)}-{rng.integers(0, 600)}", float(rng.uniform())))
    ed = pl.DataFrame(rows, schema=["s1", "cand", "p"], orient="row").unique(subset=["s1", "cand"])
    for exc in ("soft", "hard", "none"):
        for method in ("exact", "series"):
            r = D.resolve(ed, method=method, exclusivity=exc, lam_missing=0.1)
            s = r.filter(pl.col("sel"))
            assert s["cand"].n_unique() == s.height, (exc, method)  # each record used once
    all_s1 = [f"S1-{i}" for i in range(400)]
    path = str(tmp_path / "m.tsv")
    D.write_id_list_tsv(r, all_s1, path)
    got = M.load_id_list_tsv(path)
    assert len(got) == 400 and all(len(v) == len(set(v)) for v in got.values())
    with open(path, encoding="utf-8") as f:
        assert f.readline() == "source1_entity_id\tmatched_entity_ids\n"


def test_contract_select_links_and_lam_column(tmp_path):
    ed = pl.DataFrame({"s1": ["A", "A", "B", "B", "C"], "cand": ["x", "y", "x", "z", "w"],
                       "p": [0.95, 0.85, 0.90, 0.80, 0.30]})
    links = D.select_links(ed, "expected_f", exclusivity="soft", lam_missing=0.1)
    assert set(links.columns) == {"s1", "mid"} and links["mid"].n_unique() == links.height
    assert set(map(tuple, links.rows())) >= {("A", "y"), ("B", "z")}
    ed2 = ed.with_columns(pl.when(pl.col("s1") == "C").then(5.0).otherwise(0.0).alias("lam"))
    r = D.resolve(ed2, method="exact", exclusivity="none", lam_missing="lam")
    assert "lam" in r.columns
    for m in ("expected_f_fast", "plugin"):
        assert D.select_links(ed, m, lam_missing=0.1)["mid"].is_unique().all()
    h = D.assign_exclusive(ed)
    assert h.filter(pl.col("cand") == "x")["s1"].to_list() == ["A"]
    path = str(tmp_path / "c.tsv")
    D.write_id_lists(["A", "B", "C", "D"], links, path, "candidate_entity_ids")
    got = M.load_id_list_tsv(path)
    assert list(got) == ["A", "B", "C", "D"] and got["D"] == []


def test_threshold_fallback_single_pick():
    ed = pl.DataFrame({"s1": ["A", "A", "B"], "cand": ["a1", "a2", "b1"], "p": [0.4, 0.4, 0.1]})
    d = D.decide_threshold(ed, t=0.7, fallback_t=0.3)
    assert d.filter(pl.col("sel")).height == 1 and d.filter(pl.col("sel"))["s1"][0] == "A"
