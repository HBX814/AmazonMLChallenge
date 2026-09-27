# -*- coding: utf-8 -*-
"""Tests for blocking.py (+ measure_blocking.normalize_frame). Run:
  set PYTHONDONTWRITEBYTECODE=1 & set PYTHONIOENCODING=utf-8
  .venv\\Scripts\\python -m pytest -q -p no:cacheprovider .claude\\skills\\er-blocking-candidates\\test_blocking.py
The integration test uses the mini dataset if present (MINI_DIR env var overrides the default path)."""
import os
import sys

import polars as pl
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blocking as B  # noqa: E402

MINI_DIR = os.environ.get("MINI_DIR", r"C:/Users/harsh/AppData/Local/Temp/claude/c--Users-harsh-Downloads-AmazonMLChallenge/"
                                      r"da46a70e-4037-4864-96e9-d5a67558da5d/scratchpad/mini/train")


def _frame(rows):
    """rows: (entity_id, country, name_core, addr_norm, state_key[, name_norm])"""
    out = []
    for r in rows:
        eid, c, core, addr, st = r[:5]
        norm = r[5] if len(r) > 5 else core
        out.append({"entity_id": eid, "country": c, "name_norm": norm, "name_core": core,
                    "name_compact": core.replace(" ", ""), "addr_norm": addr, "house_nums": [],
                    "city_key": "", "state_key": st})
    return pl.DataFrame(out, schema={"entity_id": pl.Utf8, "country": pl.Utf8, "name_norm": pl.Utf8, "name_core": pl.Utf8,
                                     "name_compact": pl.Utf8, "addr_norm": pl.Utf8, "house_nums": pl.List(pl.Utf8),
                                     "city_key": pl.Utf8, "state_key": pl.Utf8})


S1 = _frame([
    ("S1-1", "US", "unified pinnacle esports", "221 seneca st corning ny", "ny"),
    ("S1-2", "US", "katz seafood", "15 harbor rd boston ma", "ma"),
    ("S1-3", "India", "girwa projects", "niwasa near sbi atm punjawati bhuwana girwa udaipur rajasthan", "rajasthan"),
    ("S1-4", "Atlantis", "poseidon trident works", "7 coral way atlantis city", "deep"),
    ("S1-5", "US", "primary care group", "900 elm st dallas tx", "tx"),
])
POOL = _frame([
    ("S2-10", "US", "unified pinnacle esports", "221 seneca st crning ny", "ny"),        # exact name key
    ("S3-11", "US", "unifiedpinnacleesports", "221 seneca st corning ny", "ny"),         # compact / domain form
    ("S3-12", "US", "pinnacle unified", "", ""),                                         # empty address -> fallback block
    ("S2-13", "US", "belomirabrix", "15 harbor rd boston ma", "ma"),                     # alias: address only
    ("S2-14", "US", "katz seafood", "", ""),
    ("S2-15", "India", "girwa projects", "niwasa near sbi atm punjawati bhuwana girwa rajasthan", "rajasthan"),
    ("S3-16", "India", "unified pinnacle esports", "221 seneca st corning ny", "ny"),   # other country: never a candidate of S1-1
    ("S2-17", "Atlantis", "poseidon trident works", "7 coral way atlantis", "deep"),
    ("S2-18", "US", "totally different shop", "1 main st austin tx", "tx"),
    ("S2-19", "US", "primary care group", "900 elm st dallas tx", "tx"),
])
GT = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-4", "S1-5"],
                   "mid": ["S2-10", "S3-11", "S3-12", "S2-13", "S2-14", "S2-15", "S2-17", "S2-19"]})


def _cfg(**kw):
    base = dict(max_cands_per_s1=None, verbose=False, n_threads=2, max_df_min_docs=1000)
    base.update(kw)
    return B.BlockingConfig(**base)


@pytest.fixture(scope="module")
def cands():
    return B.generate_candidates(S1, POOL, _cfg())


def test_output_schema(cands):
    for c in ["s1", "cand", "p_key_hit", "n_passes", "prio"] + [f"p_key_{k}" for k in B.KEY_PASSES] + \
             [f"p_{f}_{s}" for f in ("name", "addr", "both") for s in ("score", "rank", "rrank")]:
        assert c in cands.columns, c
    assert cands.select(pl.struct("s1", "cand").is_duplicated().any()).item() is False
    assert cands["s1"].dtype == pl.Utf8 and cands["cand"].dtype == pl.Utf8


def test_all_true_links_found(cands):
    r = B.candidate_recall(cands, GT, S1.select("entity_id", "country"))
    assert r["pair_recall"] == 1.0
    assert r["s1_full_recall"] == 1.0
    assert set(r["by_country"]) == {"US", "India", "Atlantis"}


def test_country_is_a_hard_partition(cands):
    ctry = dict(zip(POOL["entity_id"], POOL["country"])) | dict(zip(S1["entity_id"], S1["country"]))
    assert all(ctry[a] == ctry[b] for a, b in cands.select("s1", "cand").iter_rows())
    assert cands.filter((pl.col("s1") == "S1-1") & (pl.col("cand") == "S3-16")).height == 0


def test_key_passes_fire(cands):
    row = lambda s, c: cands.filter((pl.col("s1") == s) & (pl.col("cand") == c)).row(0, named=True)
    assert row("S1-1", "S2-10")["p_key_name"]
    assert row("S1-1", "S3-11")["p_key_compact"]
    assert row("S1-2", "S2-13")["p_key_addr"] and row("S1-2", "S2-13")["p_key_hn"]
    assert row("S1-2", "S2-13")["p_key_name"] is False
    assert row("S1-1", "S3-12")["p_name_score"] is not None       # fallback block (empty address, unknown state)


def test_unseen_country_works(cands):
    assert cands.filter((pl.col("s1") == "S1-4") & (pl.col("cand") == "S2-17")).height == 1


def test_reverse_direction_adds_links():
    # many near-duplicate S1 names crowd the forward top-1; the pool record's own best S1 is still kept
    s1 = _frame([(f"S1-{i}", "US", f"alpha beta gamma {i}", f"{i} x st", "tx") for i in range(8)] +
                [("S1-99", "US", "alpha beta gamma", "5 y st", "tx")])
    pool = _frame([("S2-1", "US", "alpha beta gamma", "5 y st", "tx")] +
                  [(f"S3-{i}", "US", f"alpha beta gamma {i}", f"{i} x st", "tx") for i in range(8)])
    c = B.generate_candidates(s1, pool, _cfg(key_passes=(), tfidf_passes=("name",), k_fwd={"name": 1}, k_rev={"name": 1}))
    assert c.filter((pl.col("s1") == "S1-99") & (pl.col("cand") == "S2-1")).height == 1
    assert c["p_name_rrank"].is_not_null().sum() > 0


def test_key_cap_and_block_retry():
    s1 = _frame([("S1-1", "US", "primary care group", "1 a st", "tx"), ("S1-2", "US", "primary care group", "2 b st", "ny")])
    pool = _frame([(f"S2-{i}", "US", "primary care group", f"{i} q st", "tx" if i < 3 else "ny") for i in range(40)])
    c = B.generate_candidates(s1, pool, _cfg(key_passes=("name",), tfidf_passes=(), key_cap=10, key_block_cap=5))
    # 40 pool rows share the key (> key_cap) -> retried inside state: tx has 3 (<= 5) -> kept, ny has 37 -> dropped
    assert set(c.filter(pl.col("s1") == "S1-1")["cand"]) == {"S2-0", "S2-1", "S2-2"}
    assert c.filter(pl.col("s1") == "S1-2").height == 0


def test_cap_keeps_highest_priority(cands):
    capped = B.cap_candidates(cands, 1)
    assert capped.group_by("s1").len()["len"].max() == 1
    best = cands.sort("prio", descending=True).group_by("s1", maintain_order=True).first()
    assert dict(zip(capped["s1"], capped["prio"])) == dict(zip(best["s1"], best["prio"]))
    assert B.cap_candidates(cands, None).height == cands.height


def test_candidate_recall_numbers():
    gt = pl.DataFrame({"s1": ["a", "a", "b"], "mid": ["x", "y", "z"]})
    c = pl.DataFrame({"s1": ["a", "a", "c"], "cand": ["x", "q", "w"]})
    r = B.candidate_recall(c, gt, ["a", "b", "c"])
    assert r["n_true_pairs"] == 3 and abs(r["pair_recall"] - 1 / 3) < 1e-12
    assert r["s1_full_recall"] == 0.0 and r["s1_zero_recall"] == 0.5
    # oracle: a -> tp 1 of k 2: 1.25*1/(0.25*2+1) = 0.8333; b -> 0; c singleton -> 1.0
    assert abs(r["oracle_f05"] - (1.25 / 1.5 + 0 + 1) / 3) < 1e-9
    assert r["cands_per_s1_mean"] == 1.0 and r["cands_per_s1_max"] == 2


def test_empty_inputs_do_not_crash():
    c = B.generate_candidates(S1.head(0), POOL, _cfg())
    assert c.height == 0 and "s1" in c.columns


def test_geo_shards_cover_every_s1_once():
    shards = list(B.geo_shards(S1, POOL, _cfg(), max_pool_rows=2))
    ids = [i for _, _, s, _ in shards for i in s["entity_id"].to_list()]
    assert sorted(ids) == sorted(S1["entity_id"].to_list())
    for c, _, s, p in shards:
        unk = POOL.filter((pl.col("country") == c) & (pl.col("state_key") == ""))["entity_id"].to_list()
        assert set(unk) <= set(p["entity_id"].to_list())


@pytest.mark.skipif(not os.path.exists(os.path.join(MINI_DIR, "train_source1.tsv")), reason="mini dataset not present")
def test_mini_integration_recall():
    import measure_blocking as M
    rd = lambda n: M._read(os.path.join(MINI_DIR, n))
    s1 = rd("train_source1.tsv")
    s1 = pl.concat([s1.filter(pl.col("country") == "US").head(300), s1.filter(pl.col("country") == "India").head(300)])
    gtw = rd("train_ground_truth.tsv")
    gt = (gtw.select(pl.col(gtw.columns[0]).alias("s1"), pl.col(gtw.columns[1]).str.split(",").alias("mid")).explode("mid")
             .filter(pl.col("mid").fill_null("") != "").join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi"))
    pool = pl.concat([rd("train_source2.tsv"), rd("train_source3.tsv")])
    pool = pl.concat([pool.join(gt.select(pl.col("mid").alias("entity_id")), on="entity_id", how="semi"),
                      pool.join(gt.select(pl.col("mid").alias("entity_id")), on="entity_id", how="anti").head(2000)])
    s1n, pooln = M.normalize_frame(s1, n_jobs=1), M.normalize_frame(pool, n_jobs=1)
    for col in ("name_norm", "name_core", "name_compact", "addr_norm", "house_nums", "city_key", "state_key", "script", "name_is_domain"):
        assert col in s1n.columns and col in pooln.columns
    stats = {}
    cfg = B.BlockingConfig(verbose=False, n_threads=2)
    c = B.generate_candidates(s1n, pooln, cfg, stats)
    r = B.candidate_recall(c, gt, s1n.select("entity_id", "country"))
    assert r["pair_recall"] >= 0.97, r
    assert r["cands_per_s1_max"] <= cfg.max_cands_per_s1 + cfg.reserve_slots
    assert set(stats["by_country"]) == {"US", "India"}


# ---------------------------------------------------------------------------------------------------------------
# same-address passes (key_akey / key_stname), cap reserve, legacy priority-model compatibility
# ---------------------------------------------------------------------------------------------------------------
def _frame2(rows):
    """rows: (entity_id, country, name_core, name_compact, addr_norm, house_nums, city_key, state_key)"""
    out = [{"entity_id": e, "country": c, "name_norm": core, "name_core": core, "name_compact": comp, "addr_norm": a,
            "house_nums": hn, "city_key": city, "state_key": st} for e, c, core, comp, a, hn, city, st in rows]
    return pl.DataFrame(out, schema={"entity_id": pl.Utf8, "country": pl.Utf8, "name_norm": pl.Utf8, "name_core": pl.Utf8,
                                     "name_compact": pl.Utf8, "addr_norm": pl.Utf8, "house_nums": pl.List(pl.Utf8),
                                     "city_key": pl.Utf8, "state_key": pl.Utf8})


HDF = "hauts de france"
FR_S1 = _frame2([
    ("S1-1", "France", "lille club", "lilleclub", "30 rue meuniers lille hauts de france", ["30"], "lille", HDF),
    ("S1-2", "France", "roubaix college", "roubaixcollege", "7 rue ange michel roubaix hauts de france", ["7"], "roubaix", HDF),
])
FR_POOL = _frame2([
    ("S2-1", "France", "lilleclub", "lilleclub", "30 rue meuniers lille hauts de france", ["30"], "lille", HDF),   # handle form
    ("S2-2", "France", "club lille", "clublille", "32 rue meuniers lille", ["32"], "lille", HDF),                  # other number
    ("S3-3", "France", "zeta optique", "zetaoptique", "030 bd meuniers lille", ["030"], "lille", HDF),              # same address
    ("S3-4", "France", "lille club", "lilleclub", "30 rue meuniers roubaix hauts de france", ["30"], "roubaix", HDF),  # other city
    ("S2-5", "France", "roubaix college", "roubaixcollege", "", [], "", ""),                                         # empty address
    ("S2-6", "France", "roubaix college", "roubaixcollege", "7 rue ange michel roubaix", ["7"], "roubaix", HDF),
])


def test_street_words_drop_numbers_geo_and_street_types():
    d = _frame2([
        ("a", "France", "x", "x", "30 rue des meuniers lille hauts de france", ["30"], "lille", HDF),
        ("b", "US", "x", "x", "4880 vale dr unit 5 north royalton oh", ["4880"], "north royalton", "oh"),
        ("c", "US", "x", "x", "100 main st springfield il", ["100"], "springfield", "il"),
        ("d", "US", "x", "x", "12 st", ["12"], "", ""),
        ("e", "India", "x", "x", "", [], "", ""),
    ])
    assert d.select(B.street_words().alias("sw"))["sw"].to_list() == ["meuniers", "vale", "main", "", ""]
    assert "rue" in B.STREET_STOP and "st" in B.STREET_STOP and "main" not in B.STREET_STOP


def test_street_passes_find_same_address_and_same_street_copies():
    c = B.generate_candidates(FR_S1, FR_POOL, _cfg(key_passes=("akey", "stname"), tfidf_passes=()))
    got = {(r["s1"], r["cand"]): (r["p_key_akey"], r["p_key_stname"]) for r in c.iter_rows(named=True)}
    assert got[("S1-1", "S2-1")] == (True, True)            # same address, glued name == glued core
    assert got[("S1-1", "S2-2")] == (False, True)           # same street + name, house number differs
    assert got[("S1-1", "S3-3")] == (True, False)           # same address (street type ignored, 030 -> 30), other name
    assert ("S1-1", "S3-4") not in got                      # same street words in another city
    assert got[("S1-2", "S2-6")] == (True, True)
    assert ("S1-2", "S2-5") not in got                      # no address -> no street key
    assert all(r["p_key_hit"] == int(r["p_key_akey"]) + int(r["p_key_stname"]) for r in c.iter_rows(named=True))


def test_street_cap_and_block_retry():
    s1 = _frame2([("S1-1", "US", "acme", "acme", "5 elm st springfield a", ["5"], "springfield", "a"),
                  ("S1-2", "US", "acme", "acme", "5 elm st springfield b", ["5"], "springfield", "b")])
    pool = _frame2([(f"S2-{i}", "US", f"shop {i}", f"shop{i}", "5 elm st springfield", ["5"], "springfield", "a" if i < 5 else "b")
                    for i in range(25)])
    c = B.generate_candidates(s1, pool, _cfg(key_passes=("akey",), tfidf_passes=(), street_cap=10, street_block_cap=6))
    # 25 pool rows share the akey (> street_cap) -> retried per state: a has 5 (<= 6) kept, b has 20 -> dropped
    assert set(c.filter(pl.col("s1") == "S1-1")["cand"]) == {f"S2-{i}" for i in range(5)}
    assert c.filter(pl.col("s1") == "S1-2").height == 0


def test_cap_reserve_keeps_cut_street_hits():
    d = pl.DataFrame({"s1": ["a"] * 6 + ["b"] * 2, "cand": list("uvwxyz") + ["p", "q"],
                      "prio": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.1], "r": [0, 0, 0, 1, 0, 2, 1, 1]})
    plain = B.cap_candidates(d, 3)
    assert B.cap_candidates(d, 3, "r", 0).equals(plain)
    one = B.cap_candidates(d, 3, "r", 1)
    assert set(one.filter(pl.col("s1") == "a")["cand"]) == {"u", "v", "w", "z"}     # more reserve passes first
    two = B.cap_candidates(d, 3, pl.col("r") > 0, 2)
    assert set(two.filter(pl.col("s1") == "a")["cand"]) == {"u", "v", "w", "x", "z"}
    assert set(two.filter(pl.col("s1") == "b")["cand"]) == {"p", "q"}                # cap not binding
    assert two.columns == d.columns and two.group_by("s1").len()["len"].max() <= 5
    assert two.sort(["s1", "cand"]).equals(B.cap_candidates(d.reverse(), 3, pl.col("r") > 0, 2).sort(["s1", "cand"]))


def test_reserve_slots_in_generate_candidates():
    s1 = _frame2([("S1-1", "France", "lille club", "lilleclub", "30 rue meuniers lille", ["30"], "lille", HDF)])
    pool = _frame2([(f"S2-{i}", "France", "lille club", "lilleclub", "30 rue meuniers lille", [], "", HDF) for i in range(3)]
                   + [("S3-9", "France", "zeta optique", "zetaoptique", "30 rue meuniers lille", ["30"], "lille", HDF)])
    base = dict(tfidf_passes=("name", "addr"), max_df_min_docs=1000)
    c0 = B.generate_candidates(s1, pool, _cfg(max_cands_per_s1=3, reserve_slots=0, **base))
    c1 = B.generate_candidates(s1, pool, _cfg(max_cands_per_s1=3, reserve_slots=1, **base))
    assert "S3-9" not in c0["cand"].to_list() and c0.height == 3                    # akey-only hit ranks last
    assert "S3-9" in c1["cand"].to_list() and c1.height == 4
    st = {}
    B.generate_candidates(s1, pool, _cfg(max_cands_per_s1=3, reserve_slots=1, **base), st)
    assert st["by_country"]["France"]["reserve_added"] == 1


def test_street_passes_are_deterministic():
    cfg = _cfg(max_cands_per_s1=2)
    assert B.generate_candidates(FR_S1, FR_POOL, cfg).equals(B.generate_candidates(FR_S1, FR_POOL, cfg))
    cols = ["s1", "cand", "p_key_akey", "p_key_stname", "p_key_hit", "n_passes"]
    a = B.generate_candidates(FR_S1, FR_POOL, _cfg()).select(cols)
    b = B.generate_candidates(FR_S1.reverse(), FR_POOL.reverse(), _cfg()).select(cols)    # uncapped: input-order free
    assert a.equals(b)


def test_prio_unknown_passes_for_the_v3_model_layout():
    v3 = B.prio_feature_names(_cfg(key_passes=("name", "tok", "hn", "skel", "compact", "addr")))
    assert B.prio_unknown_passes(v3, _cfg()) == ("akey", "stname")
    assert B.prio_unknown_passes(B.prio_feature_names(_cfg()), _cfg()) == ()
    assert B.prio_unknown_passes(v3[:-1], _cfg()) is None


def test_legacy_prio_model_compatibility(tmp_path):
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]
    extra_s1 = [(f"S1-x{i}", "France", f"lille club {w}", f"lilleclub{w}", f"{i + 40} rue x lille", [str(i + 40)], "lille", HDF)
                for i, w in enumerate(words)]
    extra_pool = [(f"S2-x{i}", "France", f"lille club {w}", f"lilleclub{w}", f"{i + 40} rue x lille", [str(i + 40)], "lille", HDF)
                  for i, w in enumerate(words)]
    s1, pool = pl.concat([FR_S1, _frame2(extra_s1)]), pl.concat([FR_POOL, _frame2(extra_pool)])
    old = dict(key_passes=("name",), tfidf_passes=("name",))                   # the "legacy" model knows these
    new = dict(key_passes=("name", "akey", "stname"), tfidf_passes=("name",))
    cfg_old = _cfg(**old)
    u = B.generate_candidates(s1, pool, cfg_old)
    gt = pl.DataFrame({"s1": ["S1-1", "S1-1", "S1-2", "S1-2"] + [f"S1-x{i}" for i in range(len(words))],
                       "cand": ["S2-1", "S2-2", "S2-5", "S2-6"] + [f"S2-x{i}" for i in range(len(words))]})
    u = u.join(gt.with_columns(pl.lit(1).alias("label")), on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
    path = str(tmp_path / "prio_old.txt")
    B.train_prio_model(u, cfg_old, path, rounds=3)
    names = B._prio_booster(path).feature_name()
    assert B.prio_unknown_passes(names, cfg_old) == ()
    assert B.prio_unknown_passes(names, _cfg(**new)) == ("akey", "stname")
    c_old = B.generate_candidates(s1, pool, _cfg(prio_model=path, **old))
    c_new = B.generate_candidates(s1, pool, _cfg(prio_model=path, **new))     # model predates akey / stname
    # legacy features of rows found by an old pass are bit-identical to the old run
    f_old = c_old.select("s1", "cand").hstack(B.prio_features(c_old, cfg_old))
    f_new = c_new.select("s1", "cand").hstack(B.prio_features(c_new, _cfg(**new), ("akey", "stname")))
    j = f_old.join(f_new, on=["s1", "cand"], how="left", suffix="_n")
    assert j.height == c_old.height
    for col in B.prio_feature_names(cfg_old):
        assert (j[col] == j[f"{col}_n"]).all(), col
    both = c_old.select("s1", "cand", "prio").join(c_new.select("s1", "cand", pl.col("prio").alias("prio_n")), on=["s1", "cand"])
    assert both.height == c_old.height and (both["prio"] == both["prio_n"]).all()
    only_new = c_new.join(c_old.select("s1", "cand"), on=["s1", "cand"], how="anti")
    assert only_new.height > 0 and (only_new["prio"] == 0).all()                # e.g. S1-1 x S3-3 (same address only)
    assert (only_new["p_key_akey"] | only_new["p_key_stname"]).all()
    with pytest.raises(ValueError):
        B.generate_candidates(s1, pool, _cfg(prio_model=path, key_passes=("tok",), tfidf_passes=("name",)))
