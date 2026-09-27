"""pytest for stage2.py:  python -m pytest -q -p no:cacheprovider test_stage2.py"""
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MD   # noqa: E402
import stage2 as S2  # noqa: E402

ORIG = ["f1", "f2"] + S2.GUARD_COLS


def _world(n_s1=240, n_true=3, n_dis=7, seed=0):
    """S1 i owns records S2-(1000 i + k); distractors are other S1's records (-> competition) or free records."""
    rng = np.random.default_rng(seed)
    rows, names = [], {}
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "zeta", "theta", "iota"]
    for i in range(n_s1):
        nm = f"{words[i % 10]} {words[(i * 7) % 10]} {i}"
        for k in range(rng.integers(0, n_true + 1)):
            c = f"S2-{1000 * i + k}"
            names[c] = (nm, f"{i} main street", nm.split()[0])
            rows.append((f"S1-{i}", c, 1))
        for _ in range(n_dis):
            j = int(rng.integers(0, n_s1))
            c = f"S2-{1000 * j + int(rng.integers(0, n_true))}" if rng.random() < 0.5 else f"S3-{int(rng.integers(0, 10 ** 6))}"
            if c not in names:
                names[c] = (f"other {c}", f"{j} side road", "other")
            if j != i or c.startswith("S3"):
                rows.append((f"S1-{i}", c, 0))
    df = pl.DataFrame(rows, schema=["s1", "cand", "label"], orient="row").unique(["s1", "cand"], keep="first").sort(["s1", "cand"])
    lab = df["label"].to_numpy()
    n = df.height
    f1 = rng.normal(2.0 * lab, 1.0).astype(np.float32)
    f2 = rng.normal(1.0 * lab, 1.0).astype(np.float32)
    raw = 1 / (1 + np.exp(-(2.2 * f1 + 0.8 * f2 - 3.0)))
    df = df.with_columns(pl.col("label").cast(pl.Int8), pl.Series("p", raw.astype(np.float32)),
                         pl.Series("p_raw", raw.astype(np.float32)), pl.Series("f1", f1), pl.Series("f2", f2),
                         pl.Series("hn_both", (rng.random(n) < 0.8).astype(np.float32)),
                         pl.Series("hn_exact", ((rng.random(n) < 0.5) | (lab == 1)).astype(np.float32)),
                         pl.Series("hn_base", np.zeros(n, np.float32)))
    pool = pl.DataFrame({"entity_id": list(names), "name_core": [v[0] for v in names.values()],
                         "addr_norm": [v[1] for v in names.values()], "name_key": [v[2] for v in names.values()]})
    scored = df.select("s1", "cand", "p", "p_raw", "label")
    feats = df.select(["s1", "cand", "label"] + ORIG)
    return scored, feats, pool


def _cfg(**kw):
    params = dict(S2.S2_PARAMS, num_leaves=15, min_data_in_leaf=20)
    base = dict(orig_features=["f1", "f2"], floor=1e-3, verbose=False, params=params, max_rounds=200,
                early_stopping=20, n_folds=3)
    base.update(kw)
    return S2.Stage2Config(**base)


def test_feature_columns_groups():
    cfg = _cfg()
    cols = S2.feature_columns(cfg)
    assert cols[:2] == ["f1", "f2"] and cols[2:] == S2.W_FEATS + S2.D_FEATS + S2.C_FEATS + S2.S_FEATS
    assert S2.feature_columns(_cfg(groups=["within", "competition"])) == S2.W_FEATS + S2.C_FEATS
    with pytest.raises(ValueError):
        S2.feature_columns(_cfg(groups=["nope"]))


def test_guard_columns_that_are_also_stage1_features():
    scored, feats, pool = _world(n_s1=80, seed=5)
    cfg = _cfg(orig_features=ORIG)
    fr = S2.build_frame(scored, feats, pool, cfg)
    assert fr.columns.count("hn_both") == 1 and S2.feature_columns(cfg)[:5] == ORIG


def test_build_frame_rows_features_and_determinism():
    scored, feats, pool = _world()
    cfg = _cfg()
    fr = S2.build_frame(scored, feats, pool, cfg)
    want = scored.filter(pl.col("p") >= cfg.floor)
    assert fr.height == want.height and fr["s1"].dtype == pl.Utf8
    assert set(S2.feature_columns(cfg) + S2.GUARD_COLS + ["label"]) <= set(fr.columns)
    for c in S2.feature_columns(cfg):
        assert fr[c].null_count() == 0, c
    # shuffled inputs -> identical frame
    fr2 = S2.build_frame(scored.sample(fraction=1.0, shuffle=True, seed=3), feats.sample(fraction=1.0, shuffle=True, seed=4),
                         pool.sample(fraction=1.0, shuffle=True, seed=5), cfg)
    assert fr.equals(fr2)
    # within: list length counts ALL candidates (also those below the floor)
    n = scored.group_by("s1").len()
    chk = fr.select("s1", "w_ncand").unique().join(n, on="s1")
    assert (chk["w_ncand"] == chk["len"]).all()
    # competition by hand for one record with several claimants
    multi = scored.group_by("cand").len().filter(pl.col("len") >= 3)["cand"][0]
    cl = scored.filter(pl.col("cand") == multi).sort("p", descending=True)
    top = fr.filter((pl.col("cand") == multi) & (pl.col("s1") == cl["s1"][0]))
    if top.height:
        assert top["c_n_other"][0] == cl.height - 1
        assert top["c_other_max"][0] == pytest.approx(float(cl["p"][1]), rel=1e-6)
        assert top["c_is_best"][0] == 1.0


def test_population_subset_keeps_full_competition_and_string_ids():
    scored, feats, pool = _world(seed=1)
    cfg = _cfg()
    full = S2.build_frame(scored, feats, pool, cfg)
    pop = scored["s1"].unique().sort().gather_every(3)
    sub = S2.build_frame(scored, feats, pool, cfg, population=pop)
    assert set(sub["s1"]) <= set(pop)
    j = sub.join(full, on=["s1", "cand"], suffix="_f")
    assert j.height == sub.height
    for c in S2.C_FEATS + S2.W_FEATS + S2.D_FEATS + S2.S_FEATS:
        assert np.allclose(j[c].to_numpy(), j[f"{c}_f"].to_numpy()), c
    # ids that are not S<d>-<digits> are kept as strings (no int encoding) and give the same features
    ren = lambda df, cols: df.with_columns([pl.format("X{}", pl.col(c)).alias(c) for c in cols])  # noqa: E731
    fx = S2.build_frame(ren(scored, ["s1", "cand"]), ren(feats, ["s1", "cand"]), ren(pool, ["entity_id"]), cfg)
    fx = fx.with_columns(pl.col("s1").str.slice(1), pl.col("cand").str.slice(1))
    jj = fx.join(full, on=["s1", "cand"], suffix="_f")
    assert jj.height == full.height
    for c in S2.C_FEATS + S2.S_FEATS + ["p1", "w_sum", "w_ncand", "w_sel1"]:
        assert np.allclose(jj[c].to_numpy(), jj[f"{c}_f"].to_numpy()), c


def test_fit_predict_merge_guard_and_roundtrip(tmp_path):
    scored, feats, pool = _world(n_s1=400, seed=2)
    cfg = _cfg()
    fr = S2.build_frame(scored, feats, pool, cfg)
    bundle, oof = S2.fit(fr, cfg)
    assert oof.height == fr.height and oof["p"].is_between(0, 1).all()
    assert bundle["report"]["auc_ll_stage2"][0] > 0.9
    assert abs(float(oof["p2"].mean()) - float(oof["label"].mean())) < 0.02      # isotonic on OOF
    # G1: house-number-mismatch rows can only go down
    mm = fr.join(oof, on=["s1", "cand"]).filter((pl.col("hn_both") > 0) & (pl.col("hn_exact") == 0) & (pl.col("hn_base") == 0))
    assert (mm["p"] <= mm["p1_right"] + 1e-7).all() if "p1_right" in mm.columns else (mm["p"] <= mm["p1"] + 1e-7).all()
    res = S2.predict(bundle, fr)
    merged = S2.predict(bundle, fr, scored)
    assert merged["cand"].to_list() == scored["cand"].to_list() and merged.height == scored.height
    low = merged.filter(pl.col("p_raw") < cfg.floor)
    assert np.allclose(low["p"].to_numpy(), scored.filter(pl.col("p_raw") < cfg.floor)["p"].to_numpy())
    g0 = S2.predict(bundle, fr, guard="G0")
    assert np.allclose(g0["p"].to_numpy(), g0["p2"].to_numpy())
    S2.save_bundle(bundle, str(tmp_path / "s2"))
    b2 = S2.load_bundle(str(tmp_path / "s2"))
    assert np.allclose(S2.predict(b2, fr)["p"].to_numpy(), res["p"].to_numpy(), atol=1e-6)
    cfg2 = S2.stage2_config_from_bundle(b2)
    assert cfg2.floor == cfg.floor and cfg2.orig_features == ["f1", "f2"]
    # deterministic training
    bundle_b, oof_b = S2.fit(fr.sample(fraction=1.0, shuffle=True, seed=9), cfg)
    assert np.array_equal(oof["p2"].to_numpy(), oof_b["p2"].to_numpy())


def test_stage1_stream_and_raw_from_bundle(tmp_path):
    scored, feats, pool = _world(seed=3)
    mcfg = MD.ModelConfig(n_folds=3, max_rounds=100, early_stopping=10, feature_columns=["f1", "f2"], verbose=False,
                          params=dict(MD.DEFAULT_PARAMS, num_leaves=15, min_data_in_leaf=20))
    b1, oof1 = MD.train_matcher(feats, mcfg)
    path = str(tmp_path / "feats.parquet")
    feats.write_parquet(path)
    skip = feats["s1"].unique().sort().head(50)
    st = S2.stage1_predict_stream(b1, path, skip_s1=skip, batch=500)
    assert st.height == feats.filter(~pl.col("s1").is_in(skip.implode())).height
    ref = MD.predict_matcher(b1, feats).join(st.select("s1", "cand", pl.col("p").alias("p_s")), on=["s1", "cand"])
    assert np.allclose(ref["p"].to_numpy(), ref["p_s"].to_numpy(), atol=1e-6)
    # no p_raw column -> the stage-1 raw score is re-predicted for the re-scored rows
    sc = MD.predict_matcher(b1, feats)
    with_raw = sc.join(st.select("s1", "cand", "p_raw"), on=["s1", "cand"], how="inner")
    cfg = _cfg(guard="G0")
    f_a = S2.build_frame(with_raw, path, pool, cfg)
    f_b = S2.build_frame(with_raw.drop("p_raw"), path, pool, cfg, stage1_bundle=b1)
    j = f_a.join(f_b, on=["s1", "cand"], suffix="_b")
    assert j.height == f_a.height and np.allclose(j["p1_logit"].to_numpy(), j["p1_logit_b"].to_numpy(), atol=1e-4)
