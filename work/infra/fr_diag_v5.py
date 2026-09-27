# -*- coding: utf-8 -*-
"""fr_diag_v5.py -- label-free comparison France vs US / India on the v5 test predictions (work_v5/pred) and a
label-backed check of a "unique exact name" link-adding rule on the OOF sample.
Test (per country): empty-list rate, links/S1, unclaimed pool records per S1, and unclaimed records whose full
normalized name (name_norm) equals the name of EXACTLY ONE S1 of the country with the same city (or an empty record
address) -- "unique-name orphans" -- split by whether that S1 had the record as a candidate and by its p.
Train (OOF sample, labels known): for the same record/S1 relation where the S1 is in the OOF sample and the pair is not
selected: share that are true links, and macro F0.5 if they were added (all / only candidates with p >= t).
    python infra/fr_diag_v5.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

W4, W5, S2D = Path("/vol/work_v4"), Path("/vol/work_v5"), Path("/vol/exp/ce2/s2")
OUT = {}


def unique_name_pairs(split, c):
    """(s1, cand) for pool records whose name_norm equals exactly one S1's name_norm in the country, same city_key
    (or empty pool address)."""
    s1 = pl.read_parquet(W4 / "norm" / f"{split}_{c}_s1.parquet", columns=["entity_id", "name_norm", "city_key"])
    pool = pl.read_parquet(W4 / "norm" / f"{split}_{c}_pool.parquet", columns=["entity_id", "name_norm", "city_key", "addr_norm"])
    s1 = s1.filter(pl.col("name_norm").fill_null("") != "")
    cnt = s1.group_by("name_norm").agg(pl.len().alias("n_s1"), pl.col("entity_id").first().alias("s1"),
                                       pl.col("city_key").first().alias("s1_city"))
    uniq = cnt.filter(pl.col("n_s1") == 1)
    x = pool.join(uniq, on="name_norm", how="inner")
    empty = pl.col("addr_norm").fill_null("").str.strip_chars() == ""
    x = x.filter(empty | (pl.col("city_key").fill_null("") == pl.col("s1_city").fill_null("")))
    return x.select("s1", pl.col("entity_id").alias("cand"), empty.alias("empty_addr"))


# ------------------------------------------------------------------------------------------------ test, per country
for c in ("France", "India", "US"):
    s1_ids = pl.read_parquet(W4 / "norm" / f"test_{c}_s1.parquet", columns=["entity_id"])["entity_id"]
    n = s1_ids.len()
    pool_n = pl.scan_parquet(W4 / "norm" / f"test_{c}_pool.parquet").select(pl.len()).collect().item()
    links = pl.read_parquet(W5 / "pred" / f"test_{c}_links.parquet")
    scored = pl.read_parquet(W5 / "pred" / f"test_{c}_scored.parquet")
    per = links.group_by("s1").len()
    up = unique_name_pairs("test", c)
    linked = links.select(pl.col("mid").alias("cand")).unique()
    orphan_up = up.join(linked, on="cand", how="anti")
    orphan_up = orphan_up.join(scored, on=["s1", "cand"], how="left")
    OUT[f"test_{c}"] = {
        "n_s1": n, "pool_per_s1": pool_n / n, "links_per_s1": links.height / n,
        "empty_rate": 1 - per.height / n,
        "unclaimed_pool_per_s1": (pool_n - linked.height) / n,
        "unique_name_pairs_per_s1": up.height / n,
        "unique_name_linked_to_that_s1": up.join(links.rename({"mid": "cand"}), on=["s1", "cand"], how="semi").height / n,
        "unique_name_orphans_per_s1": orphan_up.height / n,
        "  of which not a candidate": orphan_up.filter(pl.col("p").is_null()).height / n,
        "  of which candidate p<0.1": orphan_up.filter(pl.col("p") < 0.1).height / n,
        "  of which candidate p>=0.1": orphan_up.filter(pl.col("p") >= 0.1).height / n,
        "  of which empty address": orphan_up.filter("empty_addr").height / n}
    print(c, json.dumps(OUT[f"test_{c}"], indent=1), flush=True)

# ------------------------------------------------------------------------------------------------ train OOF check
tune = json.load(open(S2D / "tune_v1v2.json"))
oof = pl.read_parquet(S2D / "oof_v1v2.parquet")                      # s1, cand, label, p, country (OOF sample)
ids = oof["s1"].unique()
gt = bio.scan_split(W4, "train")["gt"].collect().join(ids.to_frame("s1"), on="s1", how="semi")
links = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft",
                       lam_missing=float(tune["lam_missing"]), empty_bias=float(tune["eb_best_oof"]))
base = M.macro_f05(links, gt, ids)
up = pl.concat([unique_name_pairs("train", c) for c in ("India", "US")]).join(ids.to_frame("s1"), on="s1", how="semi")
gtl = gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("truth"))
allsel = links.select(pl.col("mid").alias("cand")).unique()
cand = up.join(links.rename({"mid": "cand"}), on=["s1", "cand"], how="anti").join(allsel, on="cand", how="anti")
cand = cand.join(oof.select("s1", "cand", "p"), on=["s1", "cand"], how="left").join(gtl, on=["s1", "cand"], how="left").with_columns(
    pl.col("truth").fill_null(0))
res = {"F_base": base, "n_s1": ids.len(), "unique_name_pairs": up.height, "orphans_not_selected": cand.height,
       "orphans_true_rate": float(cand["truth"].mean()) if cand.height else None}
for name, f in (("all", pl.lit(True)), ("not_candidate", pl.col("p").is_null()), ("cand_p<0.1", pl.col("p") < 0.1),
                ("cand_p>=0.1", pl.col("p") >= 0.1), ("cand_p>=0.3", pl.col("p") >= 0.3), ("empty_addr", pl.col("empty_addr"))):
    sub = cand.filter(f)
    add = pl.concat([links, sub.select("s1", pl.col("cand").alias("mid"))])
    res[name] = {"n": sub.height, "true_rate": float(sub["truth"].mean()) if sub.height else None,
                 "F_if_added": M.macro_f05(add, gt, ids), "gain": M.macro_f05(add, gt, ids) - base}
OUT["train_oof_rule"] = res
print(json.dumps(res, indent=1))
json.dump(OUT, open(S2D / "fr_diag_v5.json", "w"), indent=1, default=str)
