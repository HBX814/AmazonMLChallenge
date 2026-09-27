# -*- coding: utf-8 -*-
"""prio_exp.py -- can a learned candidate priority beat blocking's hand-made `prio` at the same per-S1 cap?
Uses a slice's saved labelled candidates (measure_blocking --cap 80 output): fit LightGBM on blocking columns for
S1 with hash bucket < 50%, evaluate recall at caps on the other half. Lower bound (re-ranks within the top 80).
    python infra/prio_exp.py US_IL IN_KA IN_TGA
"""
import sys
import os

import numpy as np
import polars as pl
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "business_entity_resolution", "src"))
from ber import metric as M   # noqa: E402

PASSES_T = ("name", "addr", "both")


def feats(c: pl.DataFrame) -> pl.DataFrame:
    cols = [pl.col(k).cast(pl.Float32).fill_null(0.0) for k in c.columns if k.startswith("p_key_") and k != "p_key_hit"]
    for f in PASSES_T:
        cols += [pl.col(f"p_{f}_score").fill_null(0.0).cast(pl.Float32),
                 pl.col(f"p_{f}_rank").cast(pl.Float32).fill_null(99.0),
                 pl.col(f"p_{f}_rrank").cast(pl.Float32).fill_null(99.0)]
    cols += [pl.col("n_passes").cast(pl.Float32), pl.col("p_key_hit").cast(pl.Float32), pl.col("prio").cast(pl.Float32)]
    x = c.select(cols)
    # within-S1 context: best cosine relative to the S1's best, hand-prio rank inside the S1
    best = pl.max_horizontal([pl.col(f"p_{f}_score").fill_null(0.0) for f in PASSES_T])
    ctx = c.select((best / best.max().over("s1").clip(1e-6)).alias("best_rel"),
                   pl.col("prio").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("prio_rank"),
                   pl.len().over("s1").cast(pl.Float32).alias("n_cand"))
    return pl.concat([x, ctx], how="horizontal")


def recall_at(c: pl.DataFrame, score: str, caps, n_true: int):
    out = {}
    r = c.with_columns(pl.col(score).rank("ordinal", descending=True).over("s1").alias("_r"))
    for k in caps:
        out[k] = round(int(r.filter(pl.col("_r") <= k)["label"].sum()) / n_true, 5)
    return out


for name in sys.argv[1:]:
    d = f"work/dev/{name}"
    c = pl.read_parquet(f"{d}/{name}_cands.parquet")
    gt = pl.read_parquet(f"{d}/{name}_links.parquet")
    half = M.fold_of(c["s1"], n_folds=2, seed=4242) == 0
    X = feats(c)
    tr, te = pl.Series(half), pl.Series(~half)
    Xtr, ytr = X.filter(tr).to_numpy(), c.filter(tr)["label"].to_numpy()
    b = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
                   "verbose": -1, "num_threads": 16, "seed": 0}, lgb.Dataset(Xtr, ytr, feature_name=X.columns), 300)
    ce = c.filter(te).with_columns(pl.Series("lprio", b.predict(X.filter(te).to_numpy()).astype(np.float32)))
    s1_te = pl.Series(c["s1"].unique().to_list())
    s1_te = s1_te.filter(pl.Series(M.fold_of(s1_te, n_folds=2, seed=4242) == 1))
    n_true = gt.join(pl.DataFrame({"s1": s1_te}), on="s1", how="semi").height
    caps = (20, 30, 40, 60, 80)
    print(f"[{name}] eval half: {s1_te.len():,} S1, {n_true:,} true links, in top-80 file: "
          f"{int(ce['label'].sum()) / n_true:.5f}")
    print(f"   hand prio   recall@cap {recall_at(ce, 'prio', caps, n_true)}")
    print(f"   learned prio recall@cap {recall_at(ce, 'lprio', caps, n_true)}", flush=True)
