"""pytest for features.py:  python -m pytest -q -p no:cacheprovider test_features.py"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F  # noqa: E402


def _frames():
    s1 = pl.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "name_norm": ["unified pinnacle esports llc", "caouette cardiology"],
        "name_core": ["unified pinnacle esports", "caouette cardiology"],
        "name_compact": ["unifiedpinnacleesports", "caouettecardiology"],
        "addr_norm": ["221 seneca st corning ny", "1804 stone brook ln birmingham al"],
        "house_nums": [["221"], ["1804"]], "city_key": ["corning", "birmingham"], "state_key": ["ny", "al"],
        "script": ["latin", "latin"], "name_is_domain": [False, False], "country": ["US", "US"]})
    pool = pl.DataFrame({
        "entity_id": ["S2-10", "S3-11", "S2-12", "S3-13"],
        "name_norm": ["unified pinnacle esports", "unifiedpinnacleesports", "caouette cardiology ltd", "katz seafood"],
        "name_core": ["unified pinnacle esports", "unifiedpinnacleesports", "caouette cardiology", "katz seafood"],
        "name_compact": ["unifiedpinnacleesports", "unifiedpinnacleesports", "caouettecardiology", "katzseafood"],
        "addr_norm": ["221 seneca st crning ny", "221 seneca st corning ny", "1804a stone brook ln birmingham al", ""],
        "house_nums": [["221"], ["221"], ["1804a"], []], "city_key": ["crning", "corning", "birmingham", ""],
        "state_key": ["ny", "ny", "al", ""], "script": ["latin"] * 4, "name_is_domain": [False, True, False, False],
        "country": ["US"] * 4})
    cands = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2"],
                          "cand": ["S2-10", "S3-11", "S3-13", "S2-12", "S3-13"],
                          "label": [1, 1, 0, 1, 0], "p_key_name": [True, False, False, True, False],
                          "p_both_score": [0.9, 0.8, None, 0.95, None]})
    return cands, s1, pool


def test_shape_order_and_finite():
    cands, s1, pool = _frames()
    out = F.compute_features(cands, s1, pool, F.FeatureConfig(verbose=False))
    assert out.height == cands.height
    assert out.select("s1", "cand").rows() == cands.select("s1", "cand").rows()
    X = out.select(F.FEATURE_COLUMNS).to_numpy()
    assert X.dtype == np.float32 and np.isfinite(X).all()
    assert "country" not in F.FEATURE_COLUMNS


def test_true_pairs_score_higher():
    cands, s1, pool = _frames()
    out = F.compute_features(cands, s1, pool, F.FeatureConfig(verbose=False))
    d = {(r["s1"], r["cand"]): r for r in out.iter_rows(named=True)}
    assert d[("S1-1", "S2-10")]["c_tset"] > d[("S1-1", "S3-13")]["c_tset"]
    assert d[("S1-1", "S3-11")]["g_ratio"] == 1.0            # domain-glued form matches the glued S1 name
    assert d[("S1-2", "S2-12")]["hn_base"] == 1.0            # 1804 vs 1804a share the base number
    assert d[("S1-1", "S2-10")]["p_both_rank"] == F.RANK_MISSING   # missing blocking column -> neutral default
    assert d[("S1-2", "S2-12")]["comp_is_best"] == 1.0
    assert d[("S1-1", "S3-13")]["comp_nclaim"] == 2.0
