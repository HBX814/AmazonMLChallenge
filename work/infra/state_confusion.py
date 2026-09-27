# -*- coding: utf-8 -*-
"""state_confusion.py -- TRAIN true links whose S2/S3 record sits in a different (non-empty) state block than
the S1: counts per (s1_state, pool_state) pair and share of the S1 state's links. Label use: train only.
    python infra/state_confusion.py
"""
import polars as pl

W = "work"


def _safe(c):
    return "".join(ch if ch.isalnum() else "_" for ch in (c or "EMPTY"))


gt = pl.scan_parquet(f"{W}/cache/train_gt_long.parquet")
for c in sorted(pl.scan_parquet(f"{W}/cache/train_source1.parquet").select("country").unique().collect()["country"]):
    s1 = pl.scan_parquet(f"{W}/norm/train_{_safe(c)}_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("state_key").alias("st1"))
    pool = pl.scan_parquet(f"{W}/norm/train_{_safe(c)}_pool.parquet").select(pl.col("entity_id").alias("mid"), pl.col("state_key").alias("st2"),
                                                                             pl.col("business_address").alias("addr2"))
    j = gt.join(s1, on="s1").join(pool, on="mid").collect()
    tot = j.group_by("st1").agg(pl.len().alias("n_st1"))
    cross = j.filter((pl.col("st2") != "") & (pl.col("st2") != pl.col("st1")))
    print(f"\n=== {c}: links {j.height:,}, cross-block {cross.height:,} ({cross.height / j.height:.4f}), "
          f"empty pool addr {(j['st2'] == '').sum():,}")
    pairs = (cross.group_by("st1", "st2").agg(pl.len().alias("n")).join(tot, on="st1")
             .with_columns((pl.col("n") / pl.col("n_st1")).alias("share_of_st1")).sort("n", descending=True))
    with pl.Config(tbl_rows=25, fmt_str_lengths=40):
        print(pairs.head(25))
    print("examples:", cross.select("st1", "st2", "addr2").head(8).rows())
