# -*- coding: utf-8 -*-
"""diag_struct_i.py -- which selected links would a prior-shift recalibration drop? House-number relation mix
(true-copy-typical vs hard-negative-typical) of v3 selected links per S1 on TRAIN OOF vs TEST, and of the TEST links
dropped by the odds rescaling (w from diag_struct_h). OOF rows also get their FP rate per relation.
    python infra/diag_struct_i.py
"""
import os
import re
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide  # noqa: E402

NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
PRED = os.environ.get("DIAG_PRED", "/vol/work_v3/pred")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
T0 = time.time()
KW = dict(exclusivity="soft", lam_missing=0.05303060038344832, empty_bias=2.0)
W = {"US": 0.2306, "India": 0.5423, "France": 0.2964}
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


def tag_rel(df, split, c):
    a1 = pl.scan_parquet(f"{NORM}/{split}_{c}_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("house_nums").alias("h1")).join(
        df.select("s1").unique().lazy(), on="s1", how="semi").collect()
    a2 = pl.scan_parquet(f"{NORM}/{split}_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), pl.col("house_nums").alias("h2"),
                                                                    (pl.col("business_address").str.strip_chars() == "").alias("aempty")).join(
        df.select("cand").unique().lazy(), on="cand", how="semi").collect()
    x = df.join(a1, on="s1").join(a2, on="cand")
    x = x.with_columns(pl.Series("hrel", [hn_rel(u, v) for u, v in zip(x["h1"].to_list(), x["h2"].to_list())])).drop("h1", "h2")
    return x.with_columns(pl.when(pl.col("aempty")).then(pl.lit("empty_addr")).when(pl.col("hrel").is_in(list(HARD))).then(pl.lit("HARDNEG_shift"))
                          .otherwise(pl.col("hrel")).alias("grp"))


def mix(tag, x, n_s1, extra_cols=()):
    t = x.group_by("grp").agg((pl.len() / n_s1).alias("per_s1"), *extra_cols).sort("per_s1", descending=True)
    print(f"[I {tag}] per-S1 rate of selected links by relation group:", [tuple(round(v, 5) if isinstance(v, float) else v for v in r) for r in t.rows()], flush=True)


oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", **KW).rename({"mid": "cand"})
cty = pl.concat([pl.scan_parquet(f"{NORM}/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).with_columns(pl.lit(c).alias("country")).collect()
                 for c in ("US", "India")])
sel = links.join(oof, on=["s1", "cand"]).join(cty, on="s1")
for c in ("US", "India"):
    x = tag_rel(sel.filter(pl.col("country") == c), "train", c)
    n = oof.join(cty, on="s1").filter(pl.col("country") == c)["s1"].n_unique()
    mix(f"OOF {c}", x, n, extra_cols=((pl.col("label") == 0).mean().alias("fp_rate"),))
    # what the same rescaling would drop on OOF, and its FP rate
    oc = oof.join(cty, on="s1").filter(pl.col("country") == c)
    p = oc["p"].to_numpy().astype(np.float64)
    w = W[c]
    o2 = oc.select("s1", "cand").with_columns(pl.Series("p", w * p / (w * p + 1 - p)).cast(pl.Float32))
    l2 = decide.select_links(o2, "expected_f", **KW).rename({"mid": "cand"})
    dropped = sel.filter(pl.col("country") == c).join(l2, on=["s1", "cand"], how="anti")
    print(f"[I OOF {c}] rescaling w={w} would drop {dropped.height:,} OOF links ({dropped.height / n:.4f}/S1); FP rate among dropped {float((dropped['label'] == 0).mean()):.4f}; "
          f"FP rate among all selected {float((sel.filter(pl.col('country') == c)['label'] == 0).mean()):.4f}", flush=True)
for c in ("US", "India", "France"):
    sc = pl.read_parquet(f"{PRED}/test_{c}_scored.parquet", columns=["s1", "cand", "p"])
    tl = pl.read_parquet(f"{PRED}/test_{c}_links.parquet").rename({"mid": "cand"}).join(sc, on=["s1", "cand"])
    n = pl.scan_parquet(f"{NORM}/test_{c}_s1.parquet").select(pl.len()).collect().item()
    x = tag_rel(tl, "test", c)
    mix(f"TEST {c}", x, n)
    p = sc["p"].to_numpy().astype(np.float64)
    w = W[c]
    l2 = decide.select_links(sc.select("s1", "cand").with_columns(pl.Series("p", w * p / (w * p + 1 - p)).cast(pl.Float32)), "expected_f", **KW).rename({"mid": "cand"})
    dr = x.join(l2, on=["s1", "cand"], how="anti")
    mix(f"TEST {c} DROPPED by w={w}", dr, n)
    print(f"[I TEST {c}] dropped p quantiles (10/50/90%):", np.quantile(dr["p"].to_numpy(), [0.1, 0.5, 0.9]).round(3), flush=True)
    del sc, p
print(f"done {time.time() - T0:.0f}s")
