# -*- coding: utf-8 -*-
"""diag_struct_h.py -- (1) sibling structure of confident links (p>=0.99 selected) on TRAIN OOF vs TEST: do same-source
siblings share the identical address string as often in test as in train? (2) label-free prior-shift recalibration:
find the odds factor w that brings TEST sum-p per S1 down to the train OOF value (copies per S1 are equal in train and
test), re-run the v3 decision with p' and report links/S1; apply the same w on the OOF to measure the cost if wrong.
    python infra/diag_struct_h.py
"""
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide, metric  # noqa: E402

NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
PRED = os.environ.get("DIAG_PRED", "/vol/work_v3/pred")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
GTP = os.environ.get("DIAG_GT", "/vol/work/cache/train_gt_long.parquet")
OUT = os.environ.get("DIAG_OUT", "/vol/diag/struct")
T0 = time.time()
KW = dict(exclusivity="soft", lam_missing=0.05303060038344832, empty_bias=2.0)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def cf(col):
    return pl.col(col).str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ")


def sib_stats(tag, links_p, split, c):
    """links_p: s1, cand, p (selected). Uses p>=0.99 links only."""
    L = links_p.filter(pl.col("p") >= 0.99)
    pool = (pl.scan_parquet(f"{NORM}/{split}_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), "business_address", "business_name")
            .join(L.select("cand").lazy(), on="cand", how="semi").collect()
            .with_columns(cf("business_address").alias("acf"), cf("business_name").alias("ncf"), pl.col("cand").str.slice(0, 2).alias("src")))
    s1 = (pl.scan_parquet(f"{NORM}/{split}_{c}_s1.parquet").select(pl.col("entity_id").alias("s1"), "business_address", "business_name")
          .join(L.select("s1").unique().lazy(), on="s1", how="semi").collect()
          .with_columns(cf("business_address").alias("acf1"), cf("business_name").alias("ncf1")).drop("business_address", "business_name"))
    x = L.join(pool.drop("business_address", "business_name"), on="cand").join(s1, on="s1")
    a = x.select("s1", "cand", "src", "acf", "ncf")
    pr = a.join(a, on="s1", suffix="_b").filter(pl.col("cand") < pl.col("cand_b"))
    pr = pr.with_columns(pl.when(pl.col("src") == pl.col("src_b")).then(pl.col("src") + "-" + pl.col("src_b")).otherwise(pl.lit("S2-S3")).alias("pt"))
    ne = (pl.col("acf") != "") & (pl.col("acf_b") != "")
    t = pr.filter(ne).group_by("pt").agg(pl.len().alias("n"), (pl.col("acf") == pl.col("acf_b")).mean().alias("addr_eq"),
                                          (pl.col("ncf") == pl.col("ncf_b")).mean().alias("name_eq")).sort("pt")
    s = x.filter(pl.col("acf") != "").group_by("src").agg((pl.col("acf") == pl.col("acf1")).mean().alias("addr_eq_S1"),
                                                           (pl.col("ncf") == pl.col("ncf1")).mean().alias("name_eq_S1")).sort("src")
    print(f"[H1 {tag}] confident links {L.height:,} over {L['s1'].n_unique():,} S1; sibling pairs (non-empty addr):",
          [(r[0], r[1], round(r[2], 4), round(r[3], 4)) for r in t.rows()], "| copy vs S1:", [(r[0], round(r[1], 4), round(r[2], 4)) for r in s.rows()], flush=True)


def odds(p, w):
    return w * p / (w * p + (1 - p))


# ---------------------------------------------------------------- OOF (train)
oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
s1_ids = oof.select("s1").unique()
gt = pl.read_parquet(GTP).join(s1_ids, on="s1", how="semi")
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", **KW)
F0 = metric.macro_f05(links, gt, s1_ids)
cty = pl.concat([pl.scan_parquet(f"{NORM}/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).with_columns(pl.lit(c).alias("country")).collect()
                 for c in ("US", "India")])
lp = links.rename({"mid": "cand"}).join(oof.select("s1", "cand", "p"), on=["s1", "cand"]).join(cty, on="s1")
for c in ("US", "India"):
    sib_stats(f"TRAIN-OOF {c}", lp.filter(pl.col("country") == c), "train", c)
target = {}
o2 = oof.join(cty, on="s1")
for c in ("US", "India"):
    oc = o2.filter(pl.col("country") == c)
    target[c] = float(oc["p"].sum() / oc["s1"].n_unique())
log("OOF sum-p per S1 targets:", target, f"F0 {F0:.5f}")

# ---------------------------------------------------------------- TEST
res = {}
for c in ("US", "India", "France"):
    sc = pl.read_parquet(f"{PRED}/test_{c}_scored.parquet", columns=["s1", "cand", "p"])
    tl = pl.read_parquet(f"{PRED}/test_{c}_links.parquet").rename({"mid": "cand"}).join(sc, on=["s1", "cand"])
    sib_stats(f"TEST {c}", tl, "test", c)
    n = pl.scan_parquet(f"{NORM}/test_{c}_s1.parquet").select(pl.len()).collect().item()
    p = sc["p"].to_numpy().astype(np.float64)
    tgt = target.get(c, float(np.mean(list(target.values()))))
    lo, hi = 0.05, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if odds(p, mid).sum() / n > tgt:
            hi = mid
        else:
            lo = mid
    w = (lo + hi) / 2
    res[c] = w
    sc2 = sc.with_columns(pl.Series("p", odds(p, w)).cast(pl.Float32))
    l2 = decide.select_links(sc2, "expected_f", **KW)
    k2 = l2.group_by("s1").agg(pl.len())
    log(f"[H2 TEST {c}] sum-p per S1 {p.sum() / n:.4f} -> target {tgt:.4f} with odds factor w={w:.4f}; links/S1 v3 {tl.height / n:.4f} -> "
        f"{l2.height / n:.4f}; empty S1 v3 {1 - tl['s1'].n_unique() / n:.4f} -> {1 - k2.height / n:.4f}; links dropped {tl.height - l2.height:,}")
    del sc, sc2, p
# cost on OOF if the same w were applied where no shift exists
for c, w in res.items():
    o3 = oof.with_columns(pl.Series("p", odds(oof["p"].to_numpy().astype(np.float64), w)).cast(pl.Float32))
    l3 = decide.select_links(o3.select("s1", "cand", "p"), "expected_f", **KW)
    log(f"[H2 OOF] w={w:.4f} (from test {c}) applied to OOF: macro F0.5 {F0:.5f} -> {metric.macro_f05(l3, gt, s1_ids):.5f}; links {links.height:,} -> {l3.height:,}")
log("done")
