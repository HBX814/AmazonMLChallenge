# -*- coding: utf-8 -*-
"""diag_struct_f.py -- label-free calibration check of v3 on TEST vs the train OOF:
  sum of calibrated p over all candidates per S1 (= expected #true links among candidates if p is calibrated),
  expected FP per S1 among the selected links (sum 1-p), p-mass by bin, per country.
  Train truth: 3.46 links/S1, blocking pair recall 0.985 -> sum p per S1 should be ~3.41 if calibrated.
    python infra/diag_struct_f.py
"""
import os
import time

import numpy as np
import polars as pl

PRED = os.environ.get("DIAG_PRED", "/vol/work_v3/pred")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
T0 = time.time()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)


def summ(tag, sc, links, n_s1, extra=""):
    sel = sc.join(links, on=["s1", "cand"], how="semi")
    bins = [0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
    b = sc.with_columns(pl.col("p").cut(bins).alias("b")).group_by("b").agg((pl.len() / n_s1).alias("cands_per_s1"), (pl.col("p").sum() / n_s1).alias("pmass_per_s1")).sort("b")
    bs = sel.with_columns(pl.col("p").cut(bins).alias("b")).group_by("b").agg((pl.len() / n_s1).alias("sel_per_s1")).sort("b")
    b = b.join(bs, on="b", how="left")
    print(f"\n[F {tag}] S1 {n_s1:,}: sum p over candidates per S1 = {sc['p'].sum() / n_s1:.4f}; selected links per S1 = {sel.height / n_s1:.4f}; "
          f"expected TP per S1 (sum p over selected) = {sel['p'].sum() / n_s1:.4f}; expected FP per S1 (sum 1-p over selected) = {(1 - sel['p']).sum() / n_s1:.4f} {extra}")
    print(b.select("b", pl.col("cands_per_s1").round(4), pl.col("pmass_per_s1").round(4), pl.col("sel_per_s1").round(4)).rows())


oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
import sys
sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide  # noqa: E402
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=0.05303060038344832, empty_bias=2.0)
links = links.rename({"mid": "cand"})
cty = []
for c in ("US", "India"):
    cty.append(pl.scan_parquet(f"/vol/work_v3/norm/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).with_columns(pl.lit(c).alias("country")).collect())
cty = pl.concat(cty)
oof = oof.join(cty, on="s1")
for c in ("US", "India"):
    o = oof.filter(pl.col("country") == c)
    n = o.select("s1").n_unique()
    l = links.join(o.select("s1").unique(), on="s1", how="semi")
    tp = o.join(l, on=["s1", "cand"], how="semi")
    summ(f"OOF {c}", o.select("s1", "cand", "p"), l, n,
         extra=f"| TRUE labels among candidates per S1 {o['label'].sum() / n:.4f}; actual TP per S1 {tp['label'].sum() / n:.4f}; actual FP per S1 {(tp.height - tp['label'].sum()) / n:.4f}")
for c in ("US", "India", "France"):
    sc = pl.read_parquet(f"{PRED}/test_{c}_scored.parquet", columns=["s1", "cand", "p"])
    l = pl.read_parquet(f"{PRED}/test_{c}_links.parquet").rename({"mid": "cand"})
    n = pl.scan_parquet(f"/vol/work_v3/norm/test_{c}_s1.parquet").select(pl.len()).collect().item()
    summ(f"TEST {c}", sc, l, n)
    # per-record sum of p over S1 (competition); share of pool records whose p-sum exceeds 1
    pr = sc.group_by("cand").agg(pl.col("p").sum().alias("ps"), (pl.col("p") >= 0.5).sum().alias("n05"))
    print(f"[F TEST {c}] per-record sum_p over S1: share >1.0 {float((pr['ps'] > 1.0).mean()):.4f}, >1.2 {float((pr['ps'] > 1.2).mean()):.4f}; "
          f"records with >=2 S1 at p>=0.5: {int((pr['n05'] >= 2).sum()):,}")
print(f"done {time.time() - T0:.0f}s")
