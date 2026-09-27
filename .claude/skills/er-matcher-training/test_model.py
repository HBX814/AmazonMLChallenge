"""pytest for model.py:  python -m pytest -q -p no:cacheprovider test_model.py"""
import os
import sys
import tempfile

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model as MD  # noqa: E402


def _synthetic(n_s1=600, k=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_s1):
        n_true = rng.integers(0, 4)
        for j in range(k):
            y = int(j < n_true)
            rows.append((f"S1-{i}", f"S2-{i * 100 + j}", y, rng.normal(1.5 * y, 1.0), rng.normal(0.8 * y, 1.0)))
    df = pl.DataFrame(rows, schema=["s1", "cand", "label", "f1", "f2"], orient="row")
    return df.with_columns(pl.col("label").cast(pl.Int8), pl.col("f1").cast(pl.Float32), pl.col("f2").cast(pl.Float32))


def _cfg():
    return MD.ModelConfig(n_folds=3, max_rounds=200, early_stopping=20, feature_columns=["f1", "f2"], verbose=False,
                          params=dict(MD.DEFAULT_PARAMS, num_leaves=15, min_data_in_leaf=20))


def test_oof_is_group_disjoint_and_calibrated():
    df = _synthetic()
    bundle, oof = MD.train_matcher(df, _cfg())
    assert oof.height == df.height and oof["p"].is_between(0, 1).all()
    assert bundle["report"]["oof_auc"] > 0.8
    # calibration: mean predicted ~ positive rate
    assert abs(float(oof["p"].mean()) - float(df["label"].mean())) < 0.02


def test_save_load_roundtrip():
    df = _synthetic(200)
    bundle, _ = MD.train_matcher(df, _cfg())
    with tempfile.TemporaryDirectory() as d:
        MD.save_bundle(bundle, d)
        b2 = MD.load_bundle(d)
        p1 = MD.predict_matcher(bundle, df)["p"].to_numpy()
        p2 = MD.predict_matcher(b2, df)["p"].to_numpy()
        assert np.allclose(p1, p2, atol=1e-6)


def test_loco():
    a, b = _synthetic(300, seed=1), _synthetic(300, seed=2)
    res = MD.loco_eval({"A": a, "B": b}, _cfg())
    assert set(res) == {"A->B", "B->A"} and all(v["auc"] > 0.8 for v in res.values())
