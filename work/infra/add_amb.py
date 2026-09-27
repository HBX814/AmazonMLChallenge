# -*- coding: utf-8 -*-
"""add_amb.py -- append the name-ambiguity features (features.AMB_FEATS) to existing feature files, so an A/B test
does not recompute the other 68 columns. Keys come from features._prep_side -> identical to compute_features.
    python infra/add_amb.py /vol/work_v2 /vol/work_v3
Copies cache/, norm/, native_token_dict.tsv from SRC and writes DST/feats/<split>_<country>.parquet.
"""
import os
import shutil
import sys
import time

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "business_entity_resolution", "src"))
from ber import features as F   # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(dst, "feats"), exist_ok=True)
for d in ("cache", "norm"):
    if not os.path.exists(os.path.join(dst, d)):
        shutil.copytree(os.path.join(src, d), os.path.join(dst, d))
shutil.copy2(os.path.join(src, "native_token_dict.tsv"), os.path.join(dst, "native_token_dict.tsv"))

for fn in sorted(os.listdir(os.path.join(src, "feats"))):
    t = time.time()
    stem = fn[:-len(".parquet")]
    split, country = stem.split("_", 1)
    S = F._prep_side(pl.read_parquet(os.path.join(src, "norm", f"{stem}_s1.parquet")))
    Q = F._prep_side(pl.read_parquet(os.path.join(src, "norm", f"{stem}_pool.parquet")))
    ks = S.filter(pl.col("key") != "").group_by("key").agg(pl.len().cast(pl.Float32).alias("n_s1k"))
    kp = Q.filter(pl.col("key") != "").group_by("key").agg(pl.len().cast(pl.Float32).alias("n_poolk"))

    def counts(side, idname):
        return (side.select(pl.col("id").alias(idname), "key").join(ks, on="key", how="left").join(kp, on="key", how="left")
                .select(idname, pl.col("n_s1k").fill_null(0.0).log1p().cast(pl.Float32).alias("s"),
                        pl.col("n_poolk").fill_null(0.0).log1p().cast(pl.Float32).alias("p")))
    cq, cc = counts(S, "s1"), counts(Q, "cand")
    fe = pl.read_parquet(os.path.join(src, "feats", fn))
    fe = (fe.join(cq.rename({"s": "amb_s1_q", "p": "amb_pool_q"}), on="s1", how="left", maintain_order="left")
            .join(cc.rename({"s": "amb_s1_c", "p": "amb_pool_c"}), on="cand", how="left", maintain_order="left")
            .with_columns([pl.col(c).fill_null(0.0) for c in F.AMB_FEATS]))
    fe.write_parquet(os.path.join(dst, "feats", fn))
    print(f"{fn}: {fe.height:,} rows, +{F.AMB_FEATS} in {time.time() - t:.0f}s; "
          f"mean amb_s1_c {fe['amb_s1_c'].mean():.3f}", flush=True)
    del fe, S, Q
