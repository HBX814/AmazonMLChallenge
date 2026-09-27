"""pytest tests for metric.py  (run:  python -m pytest -q test_metric.py)"""
import os
import random
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metric as M  # noqa: E402


def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")


# ---------------------------------------------------------------- single entity
def test_problem_statement_example():
    v = M.f_beta(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(v - 5 / 7) < 1e-12 and round(v, 3) == 0.714


def test_edge_cases():
    assert M.f_beta([], []) == 1.0                     # correct singleton
    assert M.f_beta(["S2-1"], []) == 0.0               # false merge on singleton
    assert M.f_beta([], ["S2-1", "S3-2"]) == 0.0       # empty pred on matched entity
    assert M.f_beta(["S2-9"], ["S2-1"]) == 0.0         # disjoint
    assert M.f_beta(["S2-1"], ["S2-1"]) == 1.0


def test_duplicates_are_set_semantics():
    assert M.f_beta(["S2-1", "S2-1", "S3-2"], ["S2-1", "S3-2"]) == 1.0
    assert M.f_beta(["S2-1", "S2-1"], ["S2-1", "S3-2"]) == M.f_beta(["S2-1"], ["S2-1", "S3-2"])


def test_formula_equivalence():
    for tp in range(1, 8):
        for k in range(tp, 12):
            for m in range(tp, 15):
                P, R = tp / m, tp / k
                assert abs(M.f_beta_counts(tp, k, m) - 1.25 * P * R / (0.25 * P + R)) < 1e-12


def test_partial_precise_list():
    # 2 of 4 correct, no FP -> 0.8333
    assert abs(M.f_beta(["a", "b"], ["a", "b", "c", "d"]) - 1.25 * 2 / (1 + 2)) < 1e-12


# ---------------------------------------------------------------- macro average
def test_macro_with_singletons():
    gt = {"S1-1": [], "S1-2": [], "S1-3": ["S2-1", "S3-1"], "S1-4": ["S2-2"]}
    pred = {"S1-1": [], "S1-2": ["S2-5"], "S1-3": ["S2-1", "S3-1", "S2-9"], "S1-4": []}
    r = M.score_dicts(pred, gt, warn=lambda s: None)
    exp = (1.0 + 0.0 + 5 / 7 + 0.0) / 4
    assert abs(r["macro_f"] - exp) < 1e-12
    assert r["singleton_empty_acc"] == 0.5
    assert r["nonsingleton_empty_rate"] == 0.5
    # loss decomposition sums to 1 - F
    assert abs(sum(r["loss_points"].values()) - (1 - exp)) < 1e-12


def test_missing_pred_row_scored_empty_and_warned():
    msgs = []
    gt = {"S1-1": [], "S1-2": ["S2-1"]}
    r = M.score_dicts({}, gt, warn=msgs.append)
    assert abs(r["macro_f"] - 0.5) < 1e-12
    assert r["n_missing_pred_rows"] == 2 and any("REJECT" in m for m in msgs)


def test_restrict_ids_and_extra_rows():
    gt = {"S1-1": ["S2-1"], "S1-2": ["S2-2"], "S1-3": []}
    pred = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": [], "S1-99": ["S2-7"]}
    r = M.score_dicts(pred, gt, ids=["S1-1", "S1-3"], warn=lambda s: None)
    assert r["macro_f"] == 1.0 and r["n"] == 2 and r["n_pred_rows_ignored"] == 2


# ---------------------------------------------------------------- file level / engines
@pytest.fixture()
def files(tmp_path):
    gt = [("S1-1", ""), ("S1-2", "S2-1,S3-1"), ("S1-3", "S2-2"), ("S1-4", "S2-3,S2-4,S3-5,S3-6")]
    pred = [("S1-1", ""), ("S1-2", "S2-1,S3-1,S2-9"), ("S1-3", "S2-2,S2-2"), ("S1-4", "S2-3,S2-4")]
    s1 = [("S1-1", "A", "x, y", "US"), ("S1-2", "B", "x", "India"), ("S1-3", "C", "", "US"),
          ("S1-4", "D", "q", "France")]
    g, p, s, ids = (str(tmp_path / n) for n in ("gt.tsv", "pred.tsv", "s1.tsv", "ids.txt"))
    write_tsv(g, ["source1_entity_id", "matched_entity_ids"], gt)
    write_tsv(p, ["source1_entity_id", "matched_entity_ids"], pred)
    write_tsv(s, ["entity_id", "business_name", "business_address", "country"], s1)
    with open(ids, "w") as f:
        f.write("source1_entity_id\nS1-2\nS1-4\n")
    return g, p, s, ids


def test_engines_agree_on_files(files):
    g, p, s, ids = files
    exp = (1 + 5 / 7 + 1 + 1.25 * 2 / (1 + 2)) / 4
    for eng in ("python", "polars"):
        r = M.score_files(p, g, s1_path=s, by_country=True, by_k=True, by_pred_size=True,
                          engine=eng, warn=lambda x: None)
        assert abs(r["macro_f"] - exp) < 1e-12, (eng, r["macro_f"])
        assert r["n_lists_with_duplicates"] == 1
        assert set(r["by_country"]) == {"US", "India", "France"}
        assert abs(r["by_country"]["US"]["macro_f"] - 1.0) < 1e-12
        assert set(r["by_k"]) == {"0", "1", "2", "4"}
        r2 = M.score_files(p, g, ids_path=ids, engine=eng, warn=lambda x: None)
        assert abs(r2["macro_f"] - (5 / 7 + 1.25 * 2 / 3) / 2) < 1e-12


def test_crlf_and_spaces(tmp_path):
    g = tmp_path / "g.tsv"
    p = tmp_path / "p.tsv"
    g.write_bytes(b"source1_entity_id\tmatched_entity_ids\r\nS1-1\tS2-1,S3-2\r\nS1-2\t\r\n")
    p.write_bytes(b"source1_entity_id\tmatched_entity_ids\r\nS1-1\tS2-1, S3-2\r\nS1-2\t\r\n")
    for eng in ("python", "polars"):
        assert M.score_files(str(p), str(g), engine=eng, warn=lambda x: None)["macro_f"] == 1.0


def test_duplicate_rows_rejected(tmp_path):
    g = tmp_path / "g.tsv"
    g.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-1\tS2-2\n", encoding="utf-8")
    for eng in ("python", "polars"):
        with pytest.raises(ValueError):
            M.score_files(str(g), str(g), engine=eng, warn=lambda x: None)


def test_random_engine_agreement(tmp_path):
    rnd = random.Random(0)
    gt_rows, pr_rows = [], []
    pool = [f"S{rnd.choice([2, 3])}-{i}" for i in range(20000)]
    for i in range(3000):
        k = rnd.choice([0, 0, 1, 2, 3, 4, 5, 8])
        t = rnd.sample(pool, k)
        pr = [x for x in t if rnd.random() < 0.8] + rnd.sample(pool, rnd.choice([0, 0, 1, 2]))
        gt_rows.append((f"S1-{i}", ",".join(t)))
        if rnd.random() < 0.98:
            pr_rows.append((f"S1-{i}", ",".join(pr)))
    g, p = str(tmp_path / "g.tsv"), str(tmp_path / "p.tsv")
    write_tsv(g, ["source1_entity_id", "matched_entity_ids"], gt_rows)
    write_tsv(p, ["source1_entity_id", "matched_entity_ids"], pr_rows)
    a = M.score_files(p, g, engine="python", by_k=True, warn=lambda x: None)
    b = M.score_files(p, g, engine="polars", by_k=True, warn=lambda x: None)
    assert abs(a["macro_f"] - b["macro_f"]) < 1e-12
    for kb in a["by_k"]:
        assert abs(a["by_k"][kb]["macro_f"] - b["by_k"][kb]["macro_f"]) < 1e-12


def test_cli(files, tmp_path):
    g, p, s, ids = files
    out = str(tmp_path / "r.json")
    cp = subprocess.run([sys.executable, os.path.join(HERE, "metric.py"), "--pred", p, "--gt", g,
                         "--s1", s, "--by-country", "--by-k", "--ids", ids, "--json", out],
                        capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr
    assert "F0.5=" in cp.stdout and os.path.exists(out)
    cp = subprocess.run([sys.executable, os.path.join(HERE, "metric.py"), "--selftest"],
                        capture_output=True, text=True)
    assert cp.returncode == 0 and "selftest OK" in cp.stdout


# ---------------------------------------------------------------- contract API (long frames)
def test_contract_long_frames():
    import polars as pl
    gt = pl.DataFrame({"s1": ["S1-2", "S1-2", "S1-3", "S1-4", "S1-4", "S1-4", "S1-4"],
                       "mid": ["S2-1", "S3-1", "S2-2", "S2-3", "S2-4", "S3-5", "S3-6"]})
    pred = pl.DataFrame({"s1": ["S1-2", "S1-2", "S1-2", "S1-3", "S1-4", "S1-4", "S1-9"],
                         "mid": ["S2-1", "S3-1", "S2-9", "S2-2", "S2-3", "S2-4", "S2-7"]})
    ids = ["S1-1", "S1-2", "S1-3", "S1-4"]           # S1-1 is a singleton (no gt rows, no pred rows)
    exp = (1 + 5 / 7 + 1 + 1.25 * 2 / 3) / 4
    assert abs(M.macro_f05(pred, gt, ids) - exp) < 1e-12
    meta = pl.DataFrame({"entity_id": ids, "country": ["US", "India", "US", "France"]})
    b = M.f05_breakdown(pred, gt, meta)
    row = lambda gt_, g: b.filter((pl.col("group_type") == gt_) & (pl.col("group") == g)).row(0, named=True)
    assert abs(row("all", "ALL")["macro_f"] - exp) < 1e-12
    assert abs(row("country", "US")["macro_f"] - 1.0) < 1e-12
    assert abs(row("k", "4")["macro_f"] - 1.25 * 2 / 3) < 1e-12
    assert abs(row("country_k", "India|2")["macro_f"] - 5 / 7) < 1e-12
    assert abs(b.filter(pl.col("group_type") == "country")["share_of_loss"].sum() - 1.0) < 1e-12


def test_fold_of_stable_and_balanced():
    import numpy as np
    ids = [f"S1-{i * 7919 + 13}" for i in range(50000)] + ["weird-id", "S1-abc"]
    f1, f2 = M.fold_of(ids, 5, 0), M.fold_of(ids, 5, 0)
    assert (f1 == f2).all() and f1.min() == 0 and f1.max() == 4
    share = np.bincount(f1, minlength=5) / len(f1)
    assert np.abs(share - 0.2).max() < 0.01
    assert (M.fold_of(ids, 5, 1) != f1).mean() > 0.7   # different seed -> different split
