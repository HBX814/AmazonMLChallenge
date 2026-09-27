# -*- coding: utf-8 -*-
"""Tests for error_report.py (synthetic frames with hand-computed numbers + the mini dataset with fake,
noisy predictions derived from the ground truth).

Run (from the project root):
  set PYTHONDONTWRITEBYTECODE=1 & set PYTHONIOENCODING=utf-8
  .venv/Scripts/python -m pytest -p no:cacheprovider -q .claude/skills/er-error-analysis/test_error_report.py
Env: ER_MINI_DIR (folder with train/ and test/ mini TSVs; mini tests skip if absent),
     ER_ITEST_DIR (where report files are written; default: pytest tmp_path).
"""
import math
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)
for _d in (HERE, os.path.join(SKILLS, "er-f05-decisions"), os.path.join(SKILLS, "er-text-normalization")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import error_report as er  # noqa: E402
import metric  # noqa: E402

_SP = "C:/Users/harsh/AppData/Local/Temp/claude/c--Users-harsh-Downloads-AmazonMLChallenge/" \
      "da46a70e-4037-4864-96e9-d5a67558da5d/scratchpad"
MINI = os.environ.get("ER_MINI_DIR", _SP + "/mini")
ITEST = os.environ.get("ER_ITEST_DIR", _SP + "/itest/b-err")
COLS = ["entity_id", "business_name", "business_address", "country"]


@pytest.fixture
def out_dir(tmp_path):
    if os.path.isdir(os.path.dirname(ITEST)):
        os.makedirs(ITEST, exist_ok=True)
        return ITEST
    return str(tmp_path)


# ----------------------------------------------------------------------------------------------- synthetic case
def synthetic():
    s1 = pl.DataFrame([
        ("S1-1", "Unified Pinnacle Esports LLC", "221 Seneca Street, Corning, NY", "US"),
        ("S1-2", "Primary Care Group", "10 Main St, Austin, TX", "US"),
        ("S1-3", "Primary Care Group", "55 Oak Ave, Austin, TX", "US"),
        ("S1-4", "Katz Seafood LLC", "500 Elm St, Dallas, TX", "US"),
        ("S1-5", "Fortune Finance Private Limited", "12 MG Road, Bangalore, Karnataka", "India"),
        ("S1-6", "Lone Star Bakery", "9 Pine Rd, Waco, TX", "US"),
        ("S1-7", "Meridian LLC", "300 King St, Houston, TX", "US"),
        ("S1-8", "Meridian Inc", "302 King St, Houston, TX", "US"),
        ("S1-9", "Express Data Systems LLC", "77 Lake Dr, Plano, TX", "US"),
    ], schema=COLS, orient="row")
    pool = pl.DataFrame([
        ("S2-11", "Unified Pinnacle Esports L.L.C.", "221 SENECA STREET, CRNING, NY", "US"),
        ("S3-12", "unifiedpinnacleesports.com", "221 Seneca Saint, Corning, New York", "US"),
        ("S3-13", "Unified Pinnacle", "221 Seneca Street, Corning, NY", "US"),
        ("S2-14", "Unified Pinnacle Esports", "", "US"),
        ("S2-21", "Primary Care Group", "10 MAIN ST, AUSTIN, TX", "US"),
        ("S2-31", "PRIMARY CARE GROUP", "55 OAK AVE, AUSTIN, TX", "US"),
        ("S3-41", "Belomirabrix", "500 ELM ST, DALLAS, TX", "US"),
        ("S2-51", "ಫಾರ್ಚೂನ್ ಫೈನಾನ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "12 MG Road, Bangalore, KA", "India"),
        ("S3-52", "Fortune Finance Pvt Ltd", "12 M G ROAD, BANGALORE, Karnataka", "India"),
        ("S2-71", "MERIDIAN LLC", "300 KING ST, HOUSTON, TX", "US"),
        ("S3-81", "Meridian Inc", "302 King St, Houston, TX", "US"),
        ("S2-91", "Express Data", "77 LAKE DR, PLANO, TX", "US"),
        ("S3-99", "Lone Star Bakery", "19 Pine Rd, Waco, TX", "US"),
        ("S2-98", "Unified Pinnacle Esports LLC", "221 Seneca Street, Corning, NY", "US"),
    ], schema=COLS, orient="row")
    gt = pl.DataFrame([("S1-1", "S2-11"), ("S1-1", "S3-12"), ("S1-1", "S3-13"), ("S1-1", "S2-14"),
                       ("S1-2", "S2-21"), ("S1-3", "S2-31"), ("S1-4", "S3-41"), ("S1-5", "S2-51"),
                       ("S1-5", "S3-52"), ("S1-7", "S2-71"), ("S1-8", "S3-81"), ("S1-9", "S2-91")],
                      schema=["s1", "mid"], orient="row")
    scored = pl.DataFrame([
        ("S1-1", "S2-11", .95), ("S1-1", "S3-12", .40), ("S1-1", "S3-13", .80), ("S1-1", "S2-98", .90),
        ("S1-2", "S2-21", .60), ("S1-2", "S2-31", .70), ("S1-3", "S2-31", .65), ("S1-3", "S2-21", .30),
        ("S1-4", "S3-41", .20), ("S1-5", "S2-51", .30), ("S1-5", "S3-52", .90), ("S1-6", "S3-99", .75),
        ("S1-7", "S2-71", .92), ("S1-7", "S3-81", .85), ("S1-8", "S3-81", .80), ("S1-8", "S2-71", .50),
        ("S1-9", "S2-91", .35)], schema=["s1", "cand", "p"], orient="row")
    links = pl.DataFrame([("S1-1", "S2-11"), ("S1-1", "S3-13"), ("S1-1", "S2-98"), ("S1-2", "S2-31"),
                          ("S1-5", "S3-52"), ("S1-6", "S3-99"), ("S1-7", "S2-71"), ("S1-7", "S3-81")],
                         schema=["s1", "mid"], orient="row")
    return scored, links, gt, s1, pool


def fb(tp, m, k):
    return metric.f_beta_counts(tp, k, m)


@pytest.fixture(scope="module")
def syn_rep():
    scored, links, gt, s1, pool = synthetic()
    return er.build_report(scored, links, gt, s1, pool), (scored, links, gt, s1, pool)


def test_synthetic_headline_matches_hand_computation(syn_rep):
    rep, (scored, links, gt, s1, pool) = syn_rep
    fs = [fb(2, 3, 4), 0, 0, 0, fb(1, 1, 2), 0, fb(1, 2, 1), 0, 0]
    h = rep["headline"]
    assert h["macro_f"] == pytest.approx(sum(fs) / 9)
    assert h["macro_f"] == pytest.approx(metric.macro_f05(links, gt, s1["entity_id"]))
    assert h["blocking_ceiling_f"] == pytest.approx((fb(3, 3, 4) + 8) / 9)
    # S1-1 best prefix = all 4 candidates (tp 3), S1-2 best = both (tp 1 of m 2); the rest reach 1.0
    assert h["oracle_cut_f"] == pytest.approx((fb(3, 4, 4) + fb(1, 2, 1) + 7) / 9)
    assert h["blocking_pair_recall"] == pytest.approx(11 / 12)
    assert h["singleton_rate"] == pytest.approx(1 / 9)
    assert rep["meta"]["warnings"] == []


def test_synthetic_loss_decomposition_is_exact(syn_rep):
    rep, _ = syn_rep
    loss = {r[0]: r[3] for r in rep["loss"].iter_rows()}
    one_minus_f = 1 - rep["headline"]["macro_f"]
    assert sum(loss.values()) == pytest.approx(one_minus_f)
    assert float(rep["loss_rollup"]["points"].sum()) == pytest.approx(one_minus_f)
    assert loss["singleton_fp"] == pytest.approx(1 / 9)
    assert loss["empty_model"] == pytest.approx(4 / 9)          # S1-3, S1-4, S1-8, S1-9
    assert loss["allwrong_model"] == pytest.approx(1 / 9)       # S1-2 picked the S1-3 record
    assert loss["empty_blocking"] == 0 and loss["allwrong_blocking"] == 0
    # S1-1: tp=2 m=3 k=4 mm=1 bm=1 ; S1-7: tp=1 m=2 k=1 ; S1-5: tp=1 m=1 k=2 mm=1
    pf = (fb(2, 2, 4) - fb(2, 3, 4)) + (1 - fb(1, 2, 1))
    prm = (fb(3, 3, 4) - fb(2, 2, 4)) + (1 - fb(1, 1, 2))
    assert loss["partial_fp"] == pytest.approx(pf / 9)
    assert loss["partial_recall_model"] == pytest.approx(prm / 9)
    assert loss["partial_recall_blocking"] == pytest.approx((1 - fb(3, 3, 4)) / 9)
    ladder = dict(zip(rep["ladder"]["layer"], rep["ladder"]["points"]))
    assert sum(ladder.values()) == pytest.approx(one_minus_f)


def test_synthetic_pair_counts(syn_rep):
    rep, _ = syn_rep
    ec = {r[0]: r for r in rep["error_counts"].iter_rows()}
    assert ec["FP"][1] == 4 and ec["FP"][3] == 1 and ec["FP"][5] == 2
    assert ec["FN_model"][1] == 7 and ec["FN_model"][3] == 2
    assert ec["FN_blocking"][1] == 1
    src = {r[0]: r for r in rep["pairs_by_source"].iter_rows()}
    assert set(src) == {"S2", "S3"}
    assert float(rep["p_bins_errors"]["FP"].sum()) == 4


def test_synthetic_tags(syn_rep):
    rep, _ = syn_rep
    t = rep["tagged"]

    def tags(s1, cand):
        r = t.filter((pl.col("s1") == s1) & (pl.col("cand") == cand))
        assert r.height == 1, (s1, cand)
        return set(r["tags"][0].split(",")) - {""}

    assert "cross_duplicate" in tags("S1-1", "S2-98")                       # duplicate of linked S2-11
    assert {"generic_name_same_city", "belongs_to_other_s1", "house_number_mismatch"} <= tags("S1-2", "S2-31")
    assert tags("S1-6", "S3-99") == {"house_number_mismatch"}                # FP on the singleton: 9 vs 19
    assert {"legal_form_only_diff", "belongs_to_other_s1", "generic_name_same_city"} <= tags("S1-7", "S3-81")
    assert "domain_form" in tags("S1-1", "S3-12")
    assert "competition_loss" in tags("S1-3", "S2-31")
    assert "competition_loss" in tags("S1-8", "S3-81")
    assert "competition_loss" not in tags("S1-2", "S2-21")                   # linked nowhere
    assert tags("S1-4", "S3-41") == {"alias_name"}
    assert "native_script" in tags("S1-5", "S2-51")
    assert tags("S1-9", "S2-91") == {"truncated_name"}
    assert "empty_address" in tags("S1-1", "S2-14")
    tc = rep["tag_counts"]
    assert set(t.filter(pl.col("s1") == "S1-1")["etype"]) == {"FP", "FN_model", "FN_blocking", "TP"}
    assert t.filter(pl.col("etype") == "TP").height == 4                    # S2-11, S3-13, S3-52, S2-71
    fp_rows = dict(zip(tc["tag"], tc["FP_lift"]))
    assert math.isinf(fp_rows["belongs_to_other_s1"])                        # never on TPs, 2 of 4 FPs
    tc = rep["tag_counts"]
    assert tc.height == len(er.TAGS) + 1


def test_render_markdown_sections(syn_rep, out_dir):
    rep, _ = syn_rep
    md = er.render_markdown(rep, "synthetic")
    for head in ("## 1. Headline", "## 2. Loss ladder", "## 3. Loss decomposition", "## 4. Where the points are",
                 "## 5. Pair-level errors", "## 6. Calibration", "## 7. Error tags", "## 8. Examples"):
        assert head in md
    assert "Belomirabrix" in md and "\t" not in md
    with open(os.path.join(out_dir, "syn_report.md"), "w", encoding="utf-8") as f:
        f.write(md)


def test_warnings_for_contract_violations():
    scored, links, gt, s1, pool = synthetic()
    bad = pl.concat([links, pl.DataFrame([("S1-4", "S2-11"), ("S1-9", "S2-77")], schema=["s1", "mid"], orient="row")])
    rep = er.build_report(scored, bad, gt, s1, pool)
    w = " ".join(rep["meta"]["warnings"])
    assert "NOT in the scored candidates" in w and "exclusivity" in w
    lab = scored.with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    rep2 = er.build_report(lab, links, gt, s1, pool, s1_ids=["S1-1", "S1-2"])
    assert rep2["meta"]["n_s1"] == 2
    assert any("label != ground truth" in x for x in rep2["meta"]["warnings"])


def test_s1_ids_subset_and_missing_candidates():
    scored, links, gt, s1, pool = synthetic()
    rep = er.build_report(scored.filter(pl.col("s1") != "S1-9"), links, gt, s1, pool, s1_ids=["S1-9", "S1-6"])
    per = rep["per_s1"].sort("s1")
    assert per["s1"].to_list() == ["S1-6", "S1-9"]
    r9 = per.filter(pl.col("s1") == "S1-9").row(0, named=True)
    assert r9["n_cand"] == 0 and r9["f_ceiling"] == 0 and r9["empty_blocking"] == 1.0


def test_load_long_from_submission_tsv(out_dir):
    scored, links, gt, s1, pool = synthetic()
    p = os.path.join(out_dir, "links_sub.tsv")
    metric_rows = links.group_by("s1", maintain_order=True).agg(pl.col("mid").str.join(","))
    base = pl.DataFrame({"source1_entity_id": s1["entity_id"]})
    base.join(metric_rows, left_on="source1_entity_id", right_on="s1", how="left") \
        .with_columns(pl.col("mid").fill_null("")).rename({"mid": "matched_entity_ids"}) \
        .write_csv(p, separator="\t", quote_style="never")
    back = er.load_long(p).sort(["s1", "mid"])
    assert back.equals(links.sort(["s1", "mid"]))


def test_cli_report_from_parquet(out_dir):
    scored, links, gt, s1, pool = synthetic()
    paths = {}
    for name, df in (("scored", scored), ("links", links), ("gt", gt), ("s1", s1), ("pool", pool)):
        paths[name] = os.path.join(out_dir, f"cli_{name}.parquet")
        df.write_parquet(paths[name])
    out = os.path.join(out_dir, "cli_report.md")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "error_report.py"), "report",
                        *sum([[f"--{k}", v] for k, v in paths.items()], []), "--out", out, "--title", "cli"],
                       capture_output=True, text=True, env=env, cwd=HERE, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "macro F0.5=0.22377" in r.stdout
    assert os.path.isfile(out) and os.path.isfile(out.replace(".md", "_tagged.parquet"))


def test_psi_and_feature_psi():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    assert er.psi(x, rng.normal(size=20000)) < 0.01
    assert er.psi(x, x + 1.0) > 0.25
    assert math.isnan(er.psi([], x))
    n = 4000
    ids = [f"S1-{i}" for i in range(n)]
    ref = pl.DataFrame({"s1": ids, "cand": ids, "f_name": rng.random(n), "f_addr": rng.random(n)})
    cur = pl.DataFrame({"s1": ids, "cand": ids, "f_name": rng.random(n) ** 3, "f_addr": rng.random(n)})
    meta = pl.DataFrame({"entity_id": ids, "country": ["France"] * n})
    ps = er.feature_psi(ref, cur, meta, frac=1.0)
    top = ps.filter(pl.col("country") == "France").row(0, named=True)
    assert top["feature"] == "f_name" and top["psi"] > 0.25


# ----------------------------------------------------------------------------------------------- mini dataset
def _read(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False)


def _mini(split):
    d = os.path.join(MINI, split)
    if not os.path.isdir(d):
        pytest.skip(f"mini dataset not found at {d}")
    s1 = _read(os.path.join(d, f"{split}_source1.tsv"))
    pool = pl.concat([_read(os.path.join(d, f"{split}_source2.tsv")), _read(os.path.join(d, f"{split}_source3.tsv"))])
    gt = er.load_long(os.path.join(d, f"{split}_ground_truth.tsv")) if split == "train" else None
    return s1, pool, gt


def fake_run(s1, pool, gt, seed=0, drop=0.04, n_distr=8, noise=1.2, t=0.5):
    """Candidates = GT links minus `drop` (blocking misses) + random same-country distractors + a true link of
    another S1 for 10% of S1 (competition); p = noisy logistic of the label; links = threshold + exclusivity."""
    import decide
    rng = np.random.default_rng(seed)
    parts = []
    if gt is not None:
        g = gt.sort(["s1", "mid"])
        parts.append(g.filter(pl.Series(rng.random(g.height) >= drop)).rename({"mid": "cand"}))
    for cty in sorted(s1["country"].unique().to_list()):
        ids = s1.filter(pl.col("country") == cty)["entity_id"].sort()
        pids = pool.filter(pl.col("country") == cty)["entity_id"].sort()
        if pids.len() == 0:
            continue
        parts.append(pl.DataFrame({"s1": np.repeat(ids.to_numpy(), n_distr),
                                   "cand": pids.to_numpy()[rng.integers(0, pids.len(), ids.len() * n_distr)]}))
        if gt is not None:
            gc = gt.join(s1.filter(pl.col("country") == cty).select(pl.col("entity_id").alias("s1")), on="s1",
                         how="semi").sort(["s1", "mid"])
            pick = ids.to_numpy()[rng.random(ids.len()) < 0.10]
            parts.append(pl.DataFrame({"s1": pick, "cand": gc["mid"].to_numpy()[rng.integers(0, gc.height, len(pick))]}))
    sc = pl.concat(parts, how="vertical").unique().sort(["s1", "cand"])
    if gt is not None:
        sc = sc.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("y")), on=["s1", "cand"], how="left") \
               .with_columns(pl.col("y").fill_null(0))
    else:
        sc = sc.with_columns(pl.lit(0).alias("y"))
    y = sc["y"].to_numpy()
    logit = np.where(y == 1, 2.0, -2.5) + rng.normal(0, noise, sc.height)
    sc = sc.with_columns(pl.Series("p", 1 / (1 + np.exp(-logit)))).drop("y")
    links = decide.select_links(sc, method="threshold", t=t)
    return sc, links


@pytest.fixture(scope="module")
def mini_run():
    s1, pool, gt = _mini("train")
    sc, links = fake_run(s1, pool, gt)
    return s1, pool, gt, sc, links


def test_mini_report_invariants(mini_run, out_dir):
    import time
    s1, pool, gt, sc, links = mini_run
    t0 = time.time()
    rep = er.write_report(os.path.join(out_dir, "mini_report.md"), scored=sc, links=links, gt=gt, s1=s1, pool=pool,
                          max_tag_pairs=1500, title="mini fake run")
    dt = time.time() - t0
    h = rep["headline"]
    assert h["macro_f"] == pytest.approx(metric.macro_f05(links, gt, s1["entity_id"]))
    assert float(rep["loss"]["points"].sum()) == pytest.approx(1 - h["macro_f"])
    per = rep["per_s1"]
    assert per.height == s1.height
    assert (per["f_ceiling"] >= per["f_oracle_cut"] - 1e-12).all()
    assert h["blocking_ceiling_f"] >= h["oracle_cut_f"]
    assert 0.95 < h["blocking_pair_recall"] < 0.97                          # 4% of GT links dropped
    n_drop = gt.join(sc.select("s1", pl.col("cand").alias("mid")), on=["s1", "mid"], how="anti").height
    ec = {r[0]: r[1] for r in rep["error_counts"].iter_rows()}
    assert ec["FN_blocking"] == n_drop
    assert set(rep["by_country"]["country"]) == {"US", "India"}
    assert int(rep["by_country"]["n"].sum()) == s1.height
    assert rep["by_k"]["k_bucket"].to_list()[0] == "0"
    t = rep["tagged"]
    assert set(t["etype"].unique()) == {"FP", "FN_model", "FN_blocking", "TP"}
    fnm = t.filter(pl.col("etype") == "FN_model")
    assert fnm["native_script"].mean() > 0.02                                # India native-script links exist
    assert t.filter(pl.col("etype") == "FP")["belongs_to_other_s1"].sum() > 0  # injected competition
    print(f"\n[mini] n_s1={per.height} pairs={sc.height} macroF={h['macro_f']:.4f} ceiling={h['blocking_ceiling_f']:.4f} "
          f"oracle_cut={h['oracle_cut_f']:.4f} report_time={dt:.1f}s")


def test_mini_shift_report_flags_inflated_country(mini_run, out_dir):
    import decide
    s1, pool, gt, sc, links = mini_run
    ind = set(s1.filter(pl.col("country") == "India")["entity_id"].to_list())
    ind_ids = pl.DataFrame({"s1": sorted(ind), "_ind": [True] * len(ind)})
    sc2 = sc.join(ind_ids, on="s1", how="left").with_columns(
        pl.when(pl.col("_ind").fill_null(False)).then(pl.col("p").sqrt()).otherwise(pl.col("p")).alias("p")).drop("_ind")
    links2 = decide.select_links(sc2, method="threshold", t=0.5)
    md, res = er.shift_report({"scored": sc, "links": links, "s1": s1}, {"scored": sc2, "links": links2, "s1": s1})
    fl = res["flags"]
    fi = fl.filter(pl.col("country") == "India")
    assert fi.height >= 2 and set(fi["basis"]) == {"same country"}
    assert {"sel_p_lt_0.7", "conflict_rate"} <= set(fi["stat"].to_list())
    assert fl.filter(pl.col("country") == "US").height == 0                  # untouched country: no flag
    fr = s1.filter(pl.col("country") == "US").head(500).with_columns(pl.lit("France").alias("country"))
    sc3 = sc.join(fr.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    _, res3 = er.shift_report({"scored": sc, "links": links, "s1": s1},
                              {"scored": sc3, "links": links.join(sc3.select("s1").unique(), on="s1", how="semi"),
                               "s1": fr})
    assert set(res3["flags"]["basis"].to_list()) <= {"pooled ref"}
    with open(os.path.join(out_dir, "mini_shift.md"), "w", encoding="utf-8") as f:
        f.write(md)


def test_mini_test_france_inspection_sheet(out_dir):
    s1, pool, _ = _mini("test")
    fr = s1.filter(pl.col("country") == "France").sort("entity_id").head(400)
    sc, links = fake_run(fr, pool, None, seed=1, n_distr=5, t=0.2)
    md = er.inspection_sheet(sc, links, fr, pool, country="France", n=50, top=3, seed=0)
    assert md.count("\n### S1-") == 50
    assert "verdict" in md and "[uncertain]" in md and "[empty]" in md
    st = er.unlabeled_stats(sc, links, fr, "cur")
    assert st["country"].to_list() == ["France"] and st["cands_per_s1"][0] == pytest.approx(5, abs=0.1)
    with open(os.path.join(out_dir, "mini_france_inspect.md"), "w", encoding="utf-8") as f:
        f.write(md)
