# -*- coding: utf-8 -*-
"""ce_data_v2.py -- pair files for the Kaggle GPU cross-encoder (CE v2), from the v4 artifacts + ce_data.py outputs.
    kg_train.parquet : TRAIN S1 of hash buckets [300, 1000) (never in the OOF / HB populations), pairs whose final-model
                       stage-1 p is in the band [LO, HI]; columns s1, cand, label, p, ta, tb
    kg_eval.parquet  : band rows of the OOF (buckets < 150), HB (150-299) and test sets; columns set, s1, cand, p, ta, tb
Text of a record = "business_name | business_address" (raw strings). Writes /vol/exp/ce2/.
    python infra/ce_data_v2.py
"""
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import metric as M         # noqa: E402

W, CE, OUT = "/vol/work_v4", "/vol/exp/ce", "/vol/exp/ce2"
LO, HI = 0.01, 0.99
T0 = time.time()
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def texts(split, c):
    def one(path):
        d = pl.read_parquet(path, columns=["entity_id", "business_name", "business_address"])
        return d.select("entity_id", (pl.col("business_name").fill_null("") + " | "
                                      + pl.col("business_address").fill_null("")).alias("t"))
    return one(f"{W}/norm/{split}_{c}_s1.parquet"), one(f"{W}/norm/{split}_{c}_pool.parquet")


def attach(df, s1t, poolt):
    return (df.join(s1t.rename({"entity_id": "s1", "t": "ta"}), on="s1", how="left", maintain_order="left")
              .join(poolt.rename({"entity_id": "cand", "t": "tb"}), on="cand", how="left", maintain_order="left")
              .with_columns(pl.col("ta").fill_null(""), pl.col("tb").fill_null("")))


band = pl.col("p").is_between(LO, HI)
tr_parts, ev_parts = [], []
for c in ("India", "US"):
    pf = pl.read_parquet(f"{CE}/pfull_train_{c}.parquet", columns=["s1", "cand", "label", "p"]).filter(band)
    pf = pf.filter(pl.Series(M.fold_of(pf["s1"], n_folds=1000, seed=7919) >= 300))
    s1t, poolt = texts("train", c)
    tr_parts.append(attach(pf.with_columns(pl.col("label").cast(pl.Int8), pl.col("p").cast(pl.Float32)), s1t, poolt))
    log(f"train {c}: {pf.height:,} band pairs, positives {int(pf['label'].sum()):,}")
    for k in ("oof", "hb"):
        e = pl.read_parquet(f"{CE}/{k}_{c}.parquet", columns=["s1", "cand", "p", "ta", "tb"]).filter(band)
        ev_parts.append(e.select(pl.lit(f"{k}_{c}").alias("set"), "s1", "cand", pl.col("p").cast(pl.Float32), "ta", "tb"))
for c in ("France", "India", "US"):
    e = pl.read_parquet(f"{CE}/test_{c}.parquet", columns=["s1", "cand", "p", "ta", "tb"]).filter(band)
    ev_parts.append(e.select(pl.lit(f"test_{c}").alias("set"), "s1", "cand", pl.col("p").cast(pl.Float32), "ta", "tb"))
tr = pl.concat(tr_parts).sort(["s1", "cand"])
ev = pl.concat(ev_parts)
tr.write_parquet(f"{OUT}/kg_train.parquet", compression="zstd", compression_level=9)
ev.write_parquet(f"{OUT}/kg_eval.parquet", compression="zstd", compression_level=9)
log(f"kg_train {tr.height:,} rows ({int(tr['label'].sum()):,} pos), {os.path.getsize(f'{OUT}/kg_train.parquet') / 1e6:.0f} MB")
log(f"kg_eval {ev.height:,} rows, {os.path.getsize(f'{OUT}/kg_eval.parquet') / 1e6:.0f} MB; per set:\n"
    + str(ev.group_by("set").len().sort("set")))
log("done")
