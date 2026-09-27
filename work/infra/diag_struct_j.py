# -*- coding: utf-8 -*-
"""diag_struct_j.py -- label-free density of hard-negative 'branch' records: pairs (S1, pool record) with the same
name_core + state + city, per S1, by house-number relation, TRAIN vs TEST (same key definition as diag_struct_e).
    python infra/diag_struct_j.py
"""
import os
import re
import time

import polars as pl

NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
T0 = time.time()
HARD = {"diff_3_10", "diff_11_100", "one_digit_subst_d3", "one_digit_subst_d4", "one_digit_subst_d5", "one_digit_subst_d7", "one_digit_subst_d9"}


def hn_rel(h1, h2):
    if not h1 or not h2:
        return "one_missing"
    a, b = h1[0], h2[0]
    if a == b:
        return "exact"
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    if da == db and da:
        return "letter_change"
    if not da or not db:
        return "other"
    if db in da and len(db) < len(da):
        return "digit_dropped"
    if da in db and len(da) < len(db):
        return "digit_added"
    try:
        d = abs(int(da[:12]) - int(db[:12]))
    except ValueError:
        return "other"
    if len(da) == len(db) and sum(x != y for x, y in zip(da, db)) == 1:
        return f"one_digit_subst_d{d}" if d < 10 else "one_digit_subst_big"
    if d <= 2:
        return "diff_1_2"
    if d <= 10:
        return "diff_3_10"
    if d <= 100:
        return "diff_11_100"
    return "diff_100+"


kk = ["name_core", "state_key", "city_key"]
res = {}
for split, cs in (("train", ("US", "India")), ("test", ("US", "India", "France"))):
    for c in cs:
        s1 = pl.read_parquet(f"{NORM}/{split}_{c}_s1.parquet", columns=["entity_id"] + kk + ["house_nums"])
        pool = pl.read_parquet(f"{NORM}/{split}_{c}_pool.parquet", columns=["entity_id"] + kk + ["house_nums", "business_address"])
        n_s1, n_pool = s1.height, pool.height
        k1 = s1.filter((pl.col("name_core").str.len_chars() >= 2) & (pl.col("city_key").fill_null("") != "")).select(pl.col("entity_id").alias("s1"), *kk, pl.col("house_nums").alias("h1"))
        k2 = pool.filter((pl.col("name_core").str.len_chars() >= 2) & (pl.col("city_key").fill_null("") != "")).select(pl.col("entity_id").alias("cand"), *kk, pl.col("house_nums").alias("h2"))
        sz = k1.group_by(kk).agg(pl.len().alias("n1")).join(k2.group_by(kk).agg(pl.len().alias("n2")), on=kk).with_columns((pl.col("n1") * pl.col("n2")).alias("np"))
        k1 = k1.join(sz.filter(pl.col("np") <= 2000).select(kk), on=kk, how="semi")
        pr = k1.join(k2, on=kk)
        pr = pr.with_columns(pl.Series("hrel", [hn_rel(x, y) for x, y in zip(pr["h1"].to_list(), pr["h2"].to_list())]))
        pr = pr.with_columns(pl.when(pl.col("hrel").is_in(list(HARD))).then(pl.lit("HARDNEG_shift")).otherwise(pl.col("hrel")).alias("grp"))
        t = pr.group_by("grp").agg((pl.len() / n_s1).alias("per_s1"))
        res[(split, c)] = dict(t.rows())
        hs = pr.filter(pl.col("grp") == "HARDNEG_shift").select("s1").n_unique() / n_s1
        print(f"[J {split} {c}] S1 {n_s1:,} pool/S1 {n_pool / n_s1:.3f}; same name+state+city pairs per S1 {pr.height / n_s1:.4f}; "
              f"share of S1 with a HARDNEG_shift record {hs:.4f}; per-S1 by relation:",
              {k: round(v, 4) for k, v in sorted(res[(split, c)].items(), key=lambda kv: -kv[1])}, flush=True)
for c in ("US", "India"):
    tr, te = res[("train", c)], res[("test", c)]
    print(f"[J ratio test/train {c}]", {k: round(te.get(k, 0) / tr[k], 3) for k in sorted(tr, key=lambda k: -tr[k]) if tr[k] > 0.001})
print(f"done {time.time() - T0:.0f}s")
