"""pytest for adapt.py:  python -m pytest -q -p no:cacheprovider test_adapt.py"""
import os
import random
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapt as A  # noqa: E402


CASES = [
    ("12", "12", "exact"), ("12a", "12", "letter_change"), ("12", "12b", "letter_change"),
    ("1234", "234", "digit_dropped"), ("1234", "123", "digit_dropped"), ("234", "1234", "digit_added"),
    ("1234", "1235", "subst_d1_2"), ("1234", "1232", "subst_d1_2"), ("1234", "1237", "subst_d3_9"),
    ("1230", "1239", "subst_d3_9"), ("1231", "1237", "subst_d6_8"), ("1230", "1238", "subst_d6_8"),
    ("1234", "1264", "subst_d10_99"), ("1234", "1934", "subst_d100plus"), ("99", "101", "diff_1_2"),
    ("100", "95", "diff_3_10"), ("6167", "6180", "diff_11_100"), ("1566", "1570", "diff_3_10"),
    ("1566", "2570", "diff_100plus"), ("12", None, "one_missing"), ("", "7", "one_missing"), (None, "", "none"),
    ("A", "12", "other"), ("12", "B", "other"), ("1234", "1243", "diff_3_10"),
]


def test_scalar_relation_cases():
    for a, b, want in CASES:
        assert A.hn_relation(a, b) == want, (a, b, A.hn_relation(a, b), want)
    assert set(A.hn_relation(a, b) for a, b, _ in CASES) <= set(A.REL_CLASSES)


def _rand_hn(rng):
    k = rng.random()
    if k < 0.08:
        return None
    n = str(rng.randint(1, 99999))
    if k < 0.15:
        n += rng.choice("abc")
    if k < 0.18:
        n = rng.choice(["A", "B-", "Unit "]) + n
    return n


def _mutate(rng, a):
    if a is None or rng.random() < 0.1:
        return _rand_hn(rng)
    d = "".join(ch for ch in a if ch.isdigit())
    op = rng.randrange(7)
    if op == 0 or not d:
        return a
    if op == 1:
        return d + rng.choice("xyz")
    if op == 2 and len(d) > 1:
        return d[1:] if rng.random() < 0.5 else d[:-1]
    if op == 3:
        return str(rng.randint(1, 9)) + d
    if op == 4:
        i = rng.randrange(len(d))
        return d[:i] + str(rng.randint(0, 9)) + d[i + 1:]
    return str(max(0, int(d[:12]) + rng.randint(-150, 150)))


def test_vectorised_matches_scalar():
    rng = random.Random(0)
    rows = []
    for _ in range(4000):
        a = _rand_hn(rng)
        rows.append((a, _mutate(rng, a)))
    rows += [(a, b) for a, b, _ in CASES]
    df = pl.DataFrame(rows, schema=["h1", "h2"], orient="row")
    got = df.select(A.hn_relation_expr("h1", "h2").alias("r"))["r"].to_list()
    want = [A.hn_relation(a, b) for a, b in rows]
    bad = [(r, g, w) for r, g, w in zip(rows, got, want) if g != w]
    assert not bad, bad[:10]
    assert len(set(got)) >= 12


def test_tag_pairs_first_house_number_and_order():
    s1 = pl.DataFrame({"entity_id": ["S1-1", "S1-2"], "house_nums": [["12", "7"], []]})
    pool = pl.DataFrame({"entity_id": ["S2-1", "S3-2", "S2-3"], "house_nums": [["12"], ["15"], None]})
    sc = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-2", "S1-1"], "cand": ["S2-3", "S2-1", "S2-1", "S3-2"],
                       "p": [0.5, 0.9, 0.0001, 0.4]})
    t = A.tag_pairs(sc, s1, pool, p_min=0.001)
    assert t["cand"].to_list() == sc["cand"].to_list()
    assert t["hn_rel"].to_list() == ["one_missing", "exact", None, "subst_d3_9"]


# ------------------------------------------------------------------------------------------------ EM
def _synthetic(rng, n, prior, prior_s, mu=1.5):
    """scores ~ N(+-mu, 1) per class; p = calibrated posterior under the SOURCE prior prior_s."""
    y = rng.random(n) < prior
    s = rng.normal(np.where(y, mu, -mu), 1.0)
    lr = np.exp(2 * mu * s)                      # N(mu,1)/N(-mu,1)
    o = lr * prior_s / (1 - prior_s)
    return o / (1 + o), y.astype(np.int8)


def _frame(p, y, cls="diff_3_10", country="US", start=0):
    n = len(p)
    return pl.DataFrame({"s1": [f"S1-{(start + i) // 4}" for i in range(n)],
                         "cand": [f"S2-{start + i}" for i in range(n)], "p": p.astype(np.float32),
                         "label": y, "hn_rel": [cls] * n, "country": [country] * n})


def test_em_recovers_label_shift_and_is_neutral_on_source():
    rng = np.random.default_rng(0)
    ps = 0.3
    p_src, y_src = _synthetic(rng, 200_000, ps, ps)
    src = A.fit_source(_frame(p_src, y_src), A.AdaptConfig(p_min=0.0))
    cfg = A.AdaptConfig(p_min=0.0, shrink_n0=0.0)
    same = A.estimate(_frame(p_src, y_src), src, cfg, "US")
    assert abs(same["factor"][0] - 1.0) < 1e-6
    p_t, y_t = _synthetic(rng, 200_000, 0.15, ps)           # target prior halved-ish
    est = A.estimate(_frame(p_t, y_t), src, cfg, "US")
    assert abs(est["prior_raw"][0] - y_t.mean()) < 0.01, est
    assert est["factor"][0] < 0.6
    adapted, _ = A.adapt(_frame(p_t, y_t), src, cfg, "US")
    # adapted posteriors are calibrated on the target again
    assert abs(float(adapted["p"].mean()) - float(y_t.mean())) < 0.01
    assert float(_frame(p_t, y_t)["p"].mean()) - float(y_t.mean()) > 0.03


def test_em_numpy_core_matches_histogram():
    rng = np.random.default_rng(1)
    p, _ = _synthetic(rng, 50_000, 0.1, 0.25)
    full, _ = A.em_prior(p, 0.25)
    w, pm = A._hist(p, 4096)
    binned, _ = A.em_prior(pm, 0.25, w)
    assert abs(full - binned) < 1e-4


def test_safeguards_raise_block_shrink_cap_and_small_classes():
    rng = np.random.default_rng(2)
    p_src, y_src = _synthetic(rng, 100_000, 0.2, 0.2)
    src = A.fit_source(_frame(p_src, y_src), A.AdaptConfig(p_min=0.0))
    p_up, y_up = _synthetic(rng, 100_000, 0.5, 0.2)
    blocked = A.estimate(_frame(p_up, y_up), src, A.AdaptConfig(p_min=0.0), "US")
    assert blocked["factor"][0] == 1.0 and blocked["reason"][0] == "raise blocked"
    raised = A.estimate(_frame(p_up, y_up), src, A.AdaptConfig(p_min=0.0, allow_raise=True, raise_llr=10), "US")
    assert 1.0 < raised["factor"][0] <= 4.0
    p_dn, y_dn = _synthetic(rng, 100_000, 0.01, 0.2)
    capped = A.estimate(_frame(p_dn, y_dn), src, A.AdaptConfig(p_min=0.0, max_factor=2.0, shrink_n0=0), "US")
    assert capped["factor"][0] == pytest.approx(0.5) and "capped" in capped["reason"][0]
    small = A.estimate(_frame(p_dn[:100], y_dn[:100]), src, A.AdaptConfig(p_min=0.0), "US")
    assert small["factor"][0] == 1.0
    shr = A.estimate(_frame(p_dn[:3000], y_dn[:3000]), src, A.AdaptConfig(p_min=0.0, shrink_n0=3000), "US")
    raw_pt = shr["prior_raw"][0]
    assert shr["prior_t"][0] == pytest.approx(0.5 * raw_pt + 0.5 * src["priors"]["US"]["diff_3_10"]["prior"], rel=1e-6)


def test_density_estimator_and_country_fallback():
    rng = np.random.default_rng(3)
    ps = 0.2
    p_src, y_src = _synthetic(rng, 40_000, ps, ps)
    src_frame = _frame(p_src, y_src)                       # 10k S1, 4 pairs each
    src = A.fit_source(src_frame, A.AdaptConfig(p_min=0.0))
    # target: same positives per S1, twice the negatives per S1 -> prior ~ 0.2 / (0.2 + 2 * 0.8) = 0.111
    p_neg, y_neg = _synthetic(rng, 400_000, 0.0, ps)
    extra = _frame(p_neg[: int((y_src == 0).sum())], y_neg[: int((y_src == 0).sum())], start=0)
    extra = extra.with_columns(pl.format("X{}", pl.col("cand")).alias("cand"))
    tgt = pl.concat([src_frame, extra])
    est = A.estimate(tgt, src, A.AdaptConfig(method="density", p_min=0.0, shrink_n0=0), "France")
    assert est["src"][0] == A.POOLED                         # unseen country -> pooled prior
    ps_fit = src["priors"][A.POOLED]["diff_3_10"]["prior"]
    assert est["prior_raw"][0] == pytest.approx(ps_fit * 4 / (tgt.height / 10_000), rel=1e-6)
    em = A.estimate(tgt, src, A.AdaptConfig(p_min=0.0, shrink_n0=0), "France")
    assert abs(em["prior_raw"][0] - float(tgt["label"].mean())) < 0.01


def test_apply_rescales_odds_and_keeps_order():
    t = pl.DataFrame({"s1": ["a", "a", "b", "b"], "cand": ["x", "y", "z", "w"], "p": [0.8, 0.0005, 0.5, 0.9],
                      "hn_rel": ["diff_3_10", "diff_3_10", "exact", None]})
    est = pl.DataFrame({"hn_rel": ["diff_3_10", "exact"], "factor": [0.25, 1.0]})
    out = A.apply(t, est, p_min=1e-3)
    assert out["cand"].to_list() == ["x", "y", "z", "w"]
    assert out["p"][0] == pytest.approx(0.25 * 0.8 / (0.25 * 0.8 + 0.2), rel=1e-6)
    assert out["p"].to_list()[1:] == pytest.approx([0.0005, 0.5, 0.9])
    assert out["p_unadapted"].to_list() == pytest.approx(t["p"].to_list())


def test_targeted_rule():
    tagged = pl.DataFrame({"s1": ["a", "a", "a", "b", "b"], "cand": ["x", "y", "z", "u", "v"],
                           "p": [0.99, 0.9, 0.995, 0.9, 0.8], "hn_rel": ["exact", "diff_3_10", "subst_d3_9",
                                                                         "diff_11_100", "letter_change"]})
    links = tagged.select("s1", pl.col("cand").alias("mid"))
    out = A.targeted_rule(links, tagged)
    # a/y dropped (HARD, p<0.99, S1 has exact); a/z kept (p>=0.99); b/u kept (b has no exact link)
    assert sorted(out.rows()) == [("a", "x"), ("a", "z"), ("b", "u"), ("b", "v")]


def test_class_profile_and_source_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    p, y = _synthetic(rng, 1000, 0.3, 0.3)
    f = _frame(p, y)
    prof = A.class_profile(f, f.filter(pl.col("p") > 0.5).select("s1", pl.col("cand").alias("mid")), p_min=0.0)
    assert set(prof["hn_rel"]) == {"diff_3_10", "HARD"}
    src = A.fit_source(f)
    A.save_source(src, str(tmp_path / "src.json"))
    assert A.load_source(str(tmp_path / "src.json"))["priors"]["US"]["diff_3_10"]["n"] == src["priors"]["US"]["diff_3_10"]["n"]
