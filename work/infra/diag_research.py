# -*- coding: utf-8 -*-
"""diag_research.py -- where the remaining loss of the best stage-2 model (v1v2, OOF 0.98870) sits.
A. OOF pairs (p >= 0.01 or true) by category (empty candidate address x name-core relation x ambiguity), with the
   macro-F0.5 gain if the category's model misses were added / its false links dropped; blocking misses likewise.
B. Backward exact-core pass for empty-address pool records (all S1 of the country with the same name_core, only when
   <= 3 S1 share it): new pairs, true rate, recoverable blocking misses, F if the unique-core ones were linked.
C. Label-free blocking coverage per split/country: pool records that are no S1's candidate (train: share linked).
    python infra/diag_research.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

W4, S2D = Path("/vol/work_v4"), Path("/vol/exp/ce2/s2")
T0 = time.time()
OUT = {}


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


def s1_frame(split, c):
    return pl.read_parquet(W4 / "norm" / f"{split}_{c}_s1.parquet",
                           columns=["entity_id", "name_norm", "name_core", "city_key"]).with_columns(
        pl.col("name_norm").fill_null(""), pl.col("name_core").fill_null(""), pl.col("city_key").fill_null(""))


def pool_frame(split, c):
    return pl.read_parquet(W4 / "norm" / f"{split}_{c}_pool.parquet",
                           columns=["entity_id", "name_norm", "name_core", "city_key", "business_address"]).with_columns(
        pl.col("name_norm").fill_null(""), pl.col("name_core").fill_null(""), pl.col("city_key").fill_null(""),
        (pl.col("business_address").fill_null("").str.strip_chars() == "").alias("empty")).drop("business_address")


def core_counts(s1):
    return s1.filter(pl.col("name_core") != "").group_by("name_core").agg(pl.len().alias("n_nc"))


def attrs(pairs, s1, pool, cnt):
    """pairs (s1, cand, ...) + candidate empty address, same core / full name / city, #S1 sharing the cand's core."""
    return (pairs.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("name_norm").alias("s_nn"),
                                 pl.col("name_core").alias("s_nc"), pl.col("city_key").alias("s_city")), on="s1", how="left")
                 .join(pool.select(pl.col("entity_id").alias("cand"), pl.col("name_norm").alias("c_nn"),
                                   pl.col("name_core").alias("c_nc"), pl.col("city_key").alias("c_city"), "empty"),
                       on="cand", how="left")
                 .join(cnt.rename({"name_core": "c_nc"}), on="c_nc", how="left")
                 .with_columns(pl.col("n_nc").fill_null(0), pl.col("empty").fill_null(False),
                               ((pl.col("s_nc") == pl.col("c_nc")) & (pl.col("c_nc") != "")).alias("same_nc"),
                               ((pl.col("s_nn") == pl.col("c_nn")) & (pl.col("c_nn") != "")).alias("same_nn"),
                               ((pl.col("s_city") == pl.col("c_city")) & (pl.col("c_city") != "")).alias("same_city")))


CAT = (pl.when(pl.col("empty") & pl.col("same_nc") & (pl.col("n_nc") == 1)).then(pl.lit("E1 empty addr | core = S1 core, unique"))
         .when(pl.col("empty") & pl.col("same_nc")).then(pl.lit("E2 empty addr | core = S1 core, >=2 S1"))
         .when(pl.col("empty") & (pl.col("n_nc") == 0)).then(pl.lit("E3 empty addr | core = no S1 (noised)"))
         .when(pl.col("empty")).then(pl.lit("E4 empty addr | core = other S1 only"))
         .when(pl.col("same_nc") & (pl.col("n_nc") == 1)).then(pl.lit("N1 addr | core = S1 core, unique"))
         .when(pl.col("same_nc")).then(pl.lit("N2 addr | core = S1 core, >=2 S1"))
         .when(pl.col("n_nc") == 0).then(pl.lit("N3 addr | core = no S1 (noised)"))
         .otherwise(pl.lit("N4 addr | core = other S1 only")).alias("cat"))

# ------------------------------------------------------------------------------------------------ A. OOF breakdown
tune = json.load(open(S2D / "tune_v1v2.json"))
oof = pl.read_parquet(S2D / "oof_v1v2.parquet")                         # s1, cand, label, p, country
ids = oof["s1"].unique()
gt_all = bio.scan_split(W4, "train")["gt"].collect()
gt = gt_all.join(ids.to_frame("s1"), on="s1", how="semi")
links = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft",
                       lam_missing=float(tune["lam_missing"]), empty_bias=float(tune["eb_best_oof"]))
base = M.macro_f05(links, gt, ids)
log(f"base F {base:.5f} ({ids.len():,} S1, {links.height:,} links)")
s1c = oof.select("s1", "country").unique("s1")
sel = links.select("s1", pl.col("mid").alias("cand"), pl.lit(True).alias("sel"))
claimed = links.select(pl.col("mid").alias("cand")).unique().with_columns(pl.lit(True).alias("claimed_any"))
fnb = (gt.rename({"mid": "cand"}).join(oof.select("s1", "cand"), on=["s1", "cand"], how="anti")
         .join(s1c, on="s1", how="left").with_columns(pl.lit(1).alias("label"), pl.lit(None, pl.Float64).alias("p")))
focus = oof.filter((pl.col("p") >= 0.01) | (pl.col("label") == 1))
S1F, POOLF, CNT = {}, {}, {}
parts, bparts = [], []
for c in ("India", "US"):
    S1F[c], POOLF[c] = s1_frame("train", c), pool_frame("train", c)
    CNT[c] = core_counts(S1F[c])
    parts.append(attrs(focus.filter(pl.col("country") == c), S1F[c], POOLF[c], CNT[c]))
    bparts.append(attrs(fnb.filter(pl.col("country") == c), S1F[c], POOLF[c], CNT[c]))
    log(f"{c}: attrs done")
fa = (pl.concat(parts, how="vertical_relaxed").join(sel, on=["s1", "cand"], how="left")
        .join(claimed, on="cand", how="left")
        .with_columns(pl.col("sel").fill_null(False), pl.col("claimed_any").fill_null(False), CAT))
ba = pl.concat(bparts, how="vertical_relaxed").with_columns(CAT)
fn = fa.filter(~pl.col("sel") & (pl.col("label") == 1))
fp = fa.filter(pl.col("sel") & (pl.col("label") == 0))


def f_add(extra):
    return M.macro_f05(pl.concat([links, extra.select("s1", pl.col("cand").alias("mid"))]), gt, ids) - base


def f_drop(bad):
    return M.macro_f05(links.join(bad.select("s1", pl.col("cand").alias("mid")), on=["s1", "mid"], how="anti"), gt, ids) - base


rows = []
for cat in sorted(fa["cat"].unique().to_list()):
    x, fnc, fpc = fa.filter(pl.col("cat") == cat), fn.filter(pl.col("cat") == cat), fp.filter(pl.col("cat") == cat)
    rows.append({"cat": cat, "pairs": x.height, "true": int(x["label"].sum()), "selected": int(x["sel"].sum()),
                 "fn": fnc.height, "fn_claimed_by_other": int(fnc["claimed_any"].sum()),
                 "fn_p_lt_0.2": int((fnc["p"] < 0.2).sum()), "fn_p_0.2_0.5": int(((fnc["p"] >= 0.2) & (fnc["p"] < 0.5)).sum()),
                 "fn_p_ge_0.5": int((fnc["p"] >= 0.5).sum()), "fp": fpc.height,
                 "gain_fix_fn": f_add(fnc), "gain_drop_fp": f_drop(fpc)})
    log(json.dumps(rows[-1]))
OUT["A_model"] = {"base": base, "fn": fn.height, "fp": fp.height, "rows": rows}
brows = []
for cat in sorted(ba["cat"].unique().to_list()):
    b = ba.filter(pl.col("cat") == cat)
    brows.append({"cat": cat, "blocking_misses": b.height, "same_city": int(b["same_city"].sum()),
                  "gain_if_linked": f_add(b)})
    log("blocking", json.dumps(brows[-1]))
OUT["A_blocking"] = {"misses": ba.height, "gain_all": f_add(ba), "rows": brows}

# ------------------------------------------------------------------------------------------------ B. backward pass
bres = {}
new_all = []
for c in ("India", "US"):
    pool, s1, cnt = POOLF[c], S1F[c], CNT[c]
    ep = pool.filter(pl.col("empty") & (pl.col("name_core") != "")).join(cnt.filter(pl.col("n_nc") <= 3), on="name_core", how="inner")
    pairs = ep.select(pl.col("entity_id").alias("cand"), "name_core", "n_nc").join(
        s1.select(pl.col("entity_id").alias("s1"), "name_core"), on="name_core", how="inner")
    cands = pl.scan_parquet(W4 / "cands" / f"train_{c}.parquet").select("s1", "cand").collect()
    new = pairs.join(cands, on=["s1", "cand"], how="anti")
    newo = new.join(ids.to_frame("s1"), on="s1", how="semi").join(
        gt_all.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("truth")), on=["s1", "cand"], how="left").with_columns(
        pl.col("truth").fill_null(0))
    new_all.append(newo)
    bres[c] = {"empty_pool_records_core_le3": ep.height, "pairs": pairs.height, "new_pairs_all_s1": new.height,
               "new_pairs_per_s1": new.height / s1.height, "new_pairs_oof": newo.height,
               "new_true_oof": int(newo["truth"].sum()),
               "new_true_rate_unique": float(newo.filter(pl.col("n_nc") == 1)["truth"].mean() or 0)}
    log(c, "backward", json.dumps(bres[c]))
    del cands
newo = pl.concat(new_all)
bres["F_if_true_new_added (upper bound)"] = f_add(newo.filter(pl.col("truth") == 1))
bres["F_if_unique_core_new_added (rule)"] = f_add(newo.filter(pl.col("n_nc") == 1))
bres["share_of_blocking_misses_recovered"] = int(newo["truth"].sum()) / max(1, ba.height)
OUT["B_backward_empty_core_pass"] = bres
log("B", json.dumps(bres))

# ------------------------------------------------------------------------------------------------ C. coverage
cres = {}
for split, cs in (("train", ("India", "US")), ("test", ("France", "India", "US"))):
    for c in cs:
        pool = POOLF[c] if split == "train" else pool_frame(split, c)
        s1 = S1F[c] if split == "train" else s1_frame(split, c)
        cnt = core_counts(s1)
        cc = pl.scan_parquet(W4 / "cands" / f"{split}_{c}.parquet").select("cand").unique().collect()
        never = pool.join(cc.rename({"cand": "entity_id"}), on="entity_id", how="anti")
        never = never.join(cnt, on="name_core", how="left").with_columns(pl.col("n_nc").fill_null(0))
        n = s1.height
        r = {"n_s1": n, "pool_per_s1": pool.height / n, "never_cand_per_s1": never.height / n,
             "never_cand_share_of_pool": never.height / pool.height,
             "never_empty_per_s1": never.filter("empty").height / n,
             "never_core_matches_1_s1_per_s1": never.filter(pl.col("n_nc") == 1).height / n,
             "never_core_matches_2plus_s1_per_s1": never.filter(pl.col("n_nc") >= 2).height / n}
        if split == "train":
            lk = gt_all.select(pl.col("mid").alias("entity_id")).unique()
            nl = never.join(lk, on="entity_id", how="semi")
            r["never_linked_per_s1 (= blocking misses/S1, all S1)"] = nl.height / n
            r["linked_share_never"] = nl.height / max(1, never.height)
            r["linked_share_never_empty"] = nl.filter("empty").height / max(1, never.filter("empty").height)
            r["linked_share_never_core1"] = nl.filter(pl.col("n_nc") == 1).height / max(1, never.filter(pl.col("n_nc") == 1).height)
        cres[f"{split}_{c}"] = r
        log(split, c, json.dumps(r))
OUT["C_coverage"] = cres
json.dump(OUT, open(S2D / "diag_research.json", "w"), indent=1, default=str)
log("done")
