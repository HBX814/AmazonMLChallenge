# -*- coding: utf-8 -*-
"""diag_struct_d.py -- label-free structure checks of the submitted v3 TEST predictions:
  * empty-address pool records: share linked by v3 vs the ~96% expected from train structure (distractors almost never
    have an empty address), their best / second-best p and how many S1 compete
  * same-source raw-address twins: unlinked records whose identical same-source address string is linked to an S1 that
    also has the record as a candidate
  * per-S1 predicted copy counts vs the train caps (n2<=5, n3<=6) and the train k distribution
    python infra/diag_struct_d.py
"""
import os
import sys
import time

import numpy as np
import polars as pl

OUT = os.environ.get("DIAG_OUT", "/vol/diag/struct")
NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
PRED = os.environ.get("DIAG_PRED", "/vol/work_v3/pred")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(70)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# expected linked fraction from the diag_struct_b mixture estimate (k_na key), printed for reference
for c in ["US", "India", "France"]:
    log(f"==================== test {c}")
    pool = (pl.scan_parquet(f"{NORM}/test_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), "business_address", "business_name")
            .collect().with_columns(pl.col("cand").str.slice(0, 2).alias("src"),
                                    pl.col("business_address").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("acf"))
            .with_columns((pl.col("acf") == "").alias("aempty")))
    n_s1 = pl.scan_parquet(f"{NORM}/test_{c}_s1.parquet").select(pl.len()).collect().item()
    links = pl.read_parquet(f"{PRED}/test_{c}_links.parquet").rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel"))
    sc = pl.scan_parquet(f"{PRED}/test_{c}_scored.parquet").filter(pl.col("p") >= 0.01).select("s1", "cand", "p").collect()
    log("pool", pool.height, "S1", n_s1, "links", links.height, "scored p>=0.01", sc.height)
    sc = sc.join(links, on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
    per = (sc.sort("p", descending=True).group_by("cand", maintain_order=True)
           .agg(pl.col("p").first().alias("p1"), pl.col("p").slice(1, 1).first().alias("p2"), pl.col("p").sum().alias("psum"),
                (pl.col("p") >= 0.05).sum().alias("n_p05"), pl.col("sel").any().alias("linked"), pl.col("s1").first().alias("s1_top")))
    pool = pool.join(per, on="cand", how="left").with_columns(pl.col("linked").fill_null(False), pl.col("p1").fill_null(0.0),
                                                              pl.col("p2").fill_null(0.0), pl.col("psum").fill_null(0.0), pl.col("n_p05").fill_null(0))
    lf = float(pool["linked"].mean())
    print(f"[D {c}] v3 linked fraction of pool {lf:.4f}; links/S1 {links.height / n_s1:.3f}; empty-address share of pool {pool['aempty'].mean():.4f}")
    e = pool.filter(pl.col("aempty"))
    print(f"[D {c}] EMPTY-address records {e.height:,}: v3 linked {e['linked'].mean():.4f} (train structure: ~0.96-0.98 of empty-address records are true copies)")
    print(f"[D {c}] unlinked empty-address records: p1 quantiles (10/50/90%)",
          np.quantile(e.filter(~pl.col("linked"))["p1"].to_numpy(), [0.1, 0.5, 0.9]).round(3),
          "| n with p1>=0.2:", e.filter(~pl.col("linked") & (pl.col("p1") >= 0.2)).height,
          "| p1>=0.2 & p2<0.05 (one clear S1):", e.filter(~pl.col("linked") & (pl.col("p1") >= 0.2) & (pl.col("p2") < 0.05)).height,
          "| #S1 with p>=0.05 dist:", e.filter(~pl.col("linked")).group_by(pl.col("n_p05").clip(0, 5)).agg(pl.len()).sort("n_p05").rows())
    ne = pool.filter(~pl.col("aempty"))
    print(f"[D {c}] non-empty records: v3 linked {ne['linked'].mean():.4f}")
    # same-source raw address twins
    tw = pool.filter(~pl.col("aempty")).select("cand", "src", "acf", "linked")
    lt = (tw.filter(pl.col("linked")).join(links.select("s1", "cand"), on="cand").select("src", "acf", pl.col("s1").alias("s1_twin"))
          .unique(["src", "acf", "s1_twin"]))
    ut = tw.filter(~pl.col("linked")).join(lt, on=["src", "acf"])
    ut = ut.join(sc.select(pl.col("s1").alias("s1_twin"), "cand", "p"), on=["s1_twin", "cand"], how="left")
    print(f"[D {c}] unlinked records with a same-source identical-address twin linked to S1 X: {ut.select('cand').n_unique():,} "
          f"({ut.select('cand').n_unique() / max(1, tw.filter(~pl.col('linked')).height):.4f} of unlinked non-empty); of which X scored it (p>=0.01): "
          f"{ut.filter(pl.col('p').is_not_null()).select('cand').n_unique():,}; p quantiles",
          np.quantile(ut.filter(pl.col("p").is_not_null())["p"].to_numpy(), [0.1, 0.5, 0.9]).round(3) if ut.filter(pl.col("p").is_not_null()).height else None)
    # linked records sharing a same-source address string with a record linked to a DIFFERENT S1 (conflict)
    ll = tw.filter(pl.col("linked")).join(links.select("s1", "cand"), on="cand")
    g = ll.group_by("src", "acf").agg(pl.col("s1").n_unique().alias("ns1"), pl.len().alias("n"))
    print(f"[D {c}] same-source address strings linked to >1 S1: {g.filter(pl.col('ns1') > 1).height:,} groups "
          f"({g.filter(pl.col('ns1') > 1)['n'].sum():,} links) of {g.height:,}")
    # predicted per-S1 copy counts
    cnt = links.with_columns(pl.col("cand").str.slice(0, 2).alias("src")).group_by("s1").agg(
        (pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
    print(f"[D {c}] predicted S1 with n2>5: {cnt.filter(pl.col('n2') > 5).height:,}; n3>6: {cnt.filter(pl.col('n3') > 6).height:,}; "
          f"k dist:", cnt.group_by((pl.col("n2") + pl.col("n3")).clip(0, 11).alias("k")).agg(pl.len()).sort("k").rows(),
          f"empty {1 - cnt.height / n_s1:.4f}")
    pool.select("cand", "src", "aempty", "linked", "p1", "p2", "psum", "n_p05").write_parquet(f"{OUT}/d_test_pool_{c}.parquet")
log("done")
