# -*- coding: utf-8 -*-
"""diag_struct_g.py -- label-free ambiguity structure per country (train US/India vs test US/India/France):
S1-S1 key collisions (another S1 shares name_core@state / hn+city+state / name+hn+state / normalized address),
same-source identical raw address groups in the pool, share of pool records whose normalized name equals an S1 name.
    python infra/diag_struct_g.py
"""
import os
import time

import polars as pl

NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
T0 = time.time()


def keys(df):
    hn = pl.col("house_nums").list.first()
    nc = pl.col("name_core").fill_null("")
    st = pl.col("state_key").fill_null("")
    ci = pl.col("city_key").fill_null("")
    return df.with_columns(
        pl.when(nc.str.len_chars() >= 2).then(nc + "|" + st).alias("k_nst"),
        pl.when(hn.is_not_null() & (ci != "")).then(hn + "|" + ci + "|" + st).alias("k_addr"),
        pl.when((nc.str.len_chars() >= 2) & hn.is_not_null()).then(nc + "|" + hn + "|" + st).alias("k_na"),
        pl.when(pl.col("addr_norm").fill_null("") != "").then(pl.col("addr_norm")).alias("k_an"),
        pl.col("business_address").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("acf"))


COLS = ["entity_id", "business_address", "name_norm", "name_core", "addr_norm", "house_nums", "city_key", "state_key"]
for split, cs in (("train", ("US", "India")), ("test", ("US", "India", "France"))):
    for c in cs:
        s1 = keys(pl.read_parquet(f"{NORM}/{split}_{c}_s1.parquet", columns=COLS))
        pool = keys(pl.read_parquet(f"{NORM}/{split}_{c}_pool.parquet", columns=COLS)).with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
        r = {}
        for kc in ("k_nst", "k_addr", "k_na", "k_an"):
            m = s1.filter(pl.col(kc).is_not_null()).group_by(kc).agg(pl.len().alias("m"))
            x = s1.join(m, on=kc, how="left").with_columns(pl.col("m").fill_null(0))
            r[f"S1_shares_{kc}_with_other_S1"] = round(float((x["m"] > 1).mean()), 4)
        r["S1_distinct_states"] = s1["state_key"].n_unique()
        r["S1_distinct_cities"] = s1["city_key"].n_unique()
        r["mean_S1_per_city"] = round(s1.height / max(1, s1["city_key"].n_unique()), 1)
        nm = s1.select("name_norm").unique().with_columns(pl.lit(True).alias("hit"))
        r["pool_name_norm_eq_some_S1"] = round(float(pool.join(nm, on="name_norm", how="left")["hit"].fill_null(False).mean()), 4)
        g = pool.filter(pl.col("acf") != "").group_by("src", "acf").agg(pl.len().alias("n"))
        r["pool_nonempty_in_same_src_addr_group>=2"] = round(float(g.filter(pl.col("n") >= 2)["n"].sum() / max(1, g["n"].sum())), 4)
        r["pool_same_src_addr_groups>=5"] = g.filter(pl.col("n") >= 5).height
        tok = s1.select(pl.col("name_core").str.split(" ").alias("t")).explode("t")
        r["S1_name_core_vocab"] = tok["t"].n_unique()
        r["S1_name_core_tokens_mean"] = round(tok.height / s1.height, 2)
        print(f"[G {split} {c}] S1 {s1.height:,} pool {pool.height:,}:", r, flush=True)
print(f"done {time.time() - T0:.0f}s")
