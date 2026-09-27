# -*- coding: utf-8 -*-
"""ce_data.py -- pair sets (with raw record text) for the cross-encoder experiment, from the v4 artifacts.

Train S1 populations by hash bucket fold_of(s1, 1000, seed=7919) (the pipeline's sample hash):
    [0, 150)   OOF population of the v4 stage-1 / stage-2 models  -> oof_<c>   (rows with OOF p >= FLOOR)
    [150, 300) honest holdout HB (no model saw it)                -> hb_<c>    (final-model p >= FLOOR)
    [300, 450) cross-encoder TRAINING S1 (unused by every model)   -> cetrain_<c> (p >= FLOOR or positive,
               + EASY_FRAC of the remaining easy negatives)
Test: every test pair with final-model stage-1 p >= FLOOR           -> test_<c>
Also writes pfull_{train,test}_<c>.parquet (stage-1 p / p_raw of EVERY pair; needed for stage-2 competition).
Text of a record = "business_name | business_address" (raw strings, as provided).
"""
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import metric as M         # noqa: E402
from ber import model as MD         # noqa: E402
from ber import stage2 as S2        # noqa: E402

W, OUT = "/vol/work_v4", "/vol/exp/ce"
FLOOR, EASY_FRAC = 1e-3, 0.03
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


b1 = MD.load_bundle(f"{W}/model")
oof = pl.read_parquet(f"{W}/model/oof.parquet").select("s1", "cand", "label", "p", "p_raw")
for c in ("India", "US"):
    pf = S2.stage1_predict_stream(b1, f"{W}/feats/train_{c}.parquet", keep_label=True)
    pf.write_parquet(f"{OUT}/pfull_train_{c}.parquet")
    pf = pf.with_columns(pl.Series("bucket", M.fold_of(pf["s1"], n_folds=1000, seed=7919)))
    s1t, poolt = texts("train", c)
    o = oof.join(pf.filter(pl.col("bucket") < 150).select("s1").unique(), on="s1", how="semi").filter(pl.col("p") >= FLOOR)
    hb = pf.filter((pl.col("bucket") >= 150) & (pl.col("bucket") < 300) & (pl.col("p") >= FLOOR))
    tr = pf.filter((pl.col("bucket") >= 300) & (pl.col("bucket") < 450))
    hard = tr.filter((pl.col("p") >= FLOOR) | (pl.col("label") == 1))
    easy = tr.filter((pl.col("p") < FLOOR) & (pl.col("label") == 0))
    easy = easy.filter(pl.Series(M.fold_of(easy["cand"], n_folds=1000, seed=31) < int(EASY_FRAC * 1000)))
    ce = pl.concat([hard, easy]).sort(["s1", "cand"])
    for name, df in (("oof", o), ("hb", hb), ("cetrain", ce)):
        x = attach(df.select("s1", "cand", "label", "p"), s1t, poolt)
        x.write_parquet(f"{OUT}/{name}_{c}.parquet")
        log(f"{name}_{c}: {x.height:,} rows, positives {int(x['label'].sum()):,}")
    del pf, o, hb, tr, hard, easy, ce
for c in ("France", "India", "US"):
    pt = S2.stage1_predict_stream(b1, f"{W}/feats/test_{c}.parquet", keep_label=False)
    pt.write_parquet(f"{OUT}/pfull_test_{c}.parquet")
    s1t, poolt = texts("test", c)
    x = attach(pt.filter(pl.col("p") >= FLOOR).select("s1", "cand", "p"), s1t, poolt)
    x.write_parquet(f"{OUT}/test_{c}.parquet")
    log(f"test_{c}: {x.height:,} rows (of {pt.height:,} pairs)")
log("done")
