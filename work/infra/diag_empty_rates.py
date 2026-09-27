# -*- coding: utf-8 -*-
"""diag_empty_rates.py -- label-free: how the v5 predictions treat EMPTY-address pool records per test country
(France vs US / India). In train 97.7% of empty-address pool records are true copies of some S1, so a country whose
empty records are claimed much less often is either under-linking them or has a different generator.
Per country: empty records per S1; share selected by some S1 / candidate but unselected / never a candidate;
split by how many S1 of the country share the record's FULL normalized name (0 / 1 / 2+) and its core (0 / 1 / 2+);
best-claimant p of the unselected ones. Train reference: same split on the OOF population with truth.
    python infra/diag_empty_rates.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402

W4, W5, S2D = Path("/vol/work_v4"), Path("/vol/work_v5"), Path("/vol/exp/ce2/s2")
OUT = {}


def frames(split, c):
    s1 = pl.read_parquet(W4 / "norm" / f"{split}_{c}_s1.parquet", columns=["entity_id", "name_norm", "name_core"])
    pool = pl.read_parquet(W4 / "norm" / f"{split}_{c}_pool.parquet",
                           columns=["entity_id", "name_norm", "name_core", "business_address"])
    pool = pool.filter(pl.col("business_address").fill_null("").str.strip_chars() == "").drop("business_address")
    nn = s1.filter(pl.col("name_norm").fill_null("") != "").group_by("name_norm").agg(pl.len().alias("n_nn"))
    nc = s1.filter(pl.col("name_core").fill_null("") != "").group_by("name_core").agg(pl.len().alias("n_nc"))
    pool = (pool.join(nn, on="name_norm", how="left").join(nc, on="name_core", how="left")
                .with_columns(pl.col("n_nn").fill_null(0).clip(0, 2), pl.col("n_nc").fill_null(0).clip(0, 2)))
    return s1, pool


def summarize(pool, scored, links, n_s1, truth=None):
    best = scored.group_by("cand").agg(pl.col("p").max().alias("pmax"))
    sel = links.select(pl.col("mid").alias("cand")).unique().with_columns(pl.lit(True).alias("sel"))
    x = (pool.rename({"entity_id": "cand"}).join(best, on="cand", how="left").join(sel, on="cand", how="left")
             .with_columns(pl.col("sel").fill_null(False)))
    if truth is not None:
        x = x.join(truth, on="cand", how="left").with_columns(pl.col("linked").fill_null(False))
    st = pl.when(pl.col("sel")).then(pl.lit("selected")).when(pl.col("pmax").is_null()).then(pl.lit("never_cand")) \
           .otherwise(pl.lit("cand_unsel"))
    x = x.with_columns(st.alias("state"))
    r = {"empty_per_s1": x.height / n_s1,
         "state_share": {k: v for k, v in x.group_by("state").len().with_columns((pl.col("len") / x.height).alias("s"))
                         .select("state", "s").iter_rows()}}
    for key in ("n_nn", "n_nc"):
        g = (x.group_by(key).agg(pl.len().alias("n"), pl.col("sel").mean().alias("sel_rate"),
                                 (pl.col("state") == "never_cand").mean().alias("never_rate"),
                                 *([pl.col("linked").mean().alias("true_linked_rate")] if truth is not None else []))
              .sort(key))
        r[f"by_{key}"] = [dict(zip(g.columns, row)) for row in g.iter_rows()]
    u = x.filter(pl.col("state") == "cand_unsel")
    bins = [0.0, 0.1, 0.3, 0.5, 0.77, 1.01]
    r["unsel_pmax_hist"] = {f"{lo}-{hi}": u.filter((pl.col("pmax") >= lo) & (pl.col("pmax") < hi)).height / n_s1
                            for lo, hi in zip(bins[:-1], bins[1:])}
    r["unsel_nn1_pmax_ge_0.3_per_s1"] = u.filter((pl.col("n_nn") == 1) & (pl.col("pmax") >= 0.3)).height / n_s1
    return r


for c in ("France", "India", "US"):
    s1, pool = frames("test", c)
    scored = pl.read_parquet(W5 / "pred" / f"test_{c}_scored.parquet", columns=["s1", "cand", "p"])
    links = pl.read_parquet(W5 / "pred" / f"test_{c}_links.parquet")
    OUT[f"test_{c}"] = summarize(pool, scored, links, s1.height)
    print(c, json.dumps(OUT[f"test_{c}"], indent=1), flush=True)

# train reference: OOF population (331k S1) -- pool records whose candidates are OOF S1 only
tune = json.load(open(S2D / "tune_v1v2.json"))
oof = pl.read_parquet(S2D / "oof_v1v2.parquet")
links = D.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft",
                       lam_missing=float(tune["lam_missing"]), empty_bias=float(tune["eb_best_oof"]))
gt = bio.scan_split(W4, "train")["gt"].collect()
ids = oof["s1"].unique().to_frame("s1")
for c in ("India", "US"):
    s1, pool = frames("train", c)
    sc = oof.filter(pl.col("country") == c).select("s1", "cand", "p")
    owned = gt.join(ids, on="s1", how="semi").select(pl.col("mid").alias("cand")).unique()
    # empty records that are linked to an OOF S1 (true copies of the population) or never linked at all (distractors)
    linked_any = gt.select(pl.col("mid").alias("cand")).unique()
    pool_oof = pool.join(owned.rename({"cand": "entity_id"}), on="entity_id", how="semi")   # true copies only -> sel_rate = recall
    n_oof = s1.join(ids.rename({"s1": "entity_id"}), on="entity_id", how="semi").height
    OUT[f"train_oof_{c}"] = summarize(pool_oof, sc, links.join(ids, on="s1", how="semi"), n_oof)
    OUT[f"train_oof_{c}"]["distractor_empty_per_s1 (all train)"] = pool.join(linked_any.rename({"cand": "entity_id"}), on="entity_id", how="anti").height / s1.height
    print("train OOF", c, json.dumps(OUT[f"train_oof_{c}"], indent=1), flush=True)
json.dump(OUT, open(S2D / "diag_empty_rates.json", "w"), indent=1, default=str)
