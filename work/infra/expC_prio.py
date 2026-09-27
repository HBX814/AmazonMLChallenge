# -*- coding: utf-8 -*-
"""expC_prio.py -- retrain the learned candidate priority WITH the new pass flags (p_key_akey, p_key_stname), exactly
like run_pipeline.stage_prio (v3 config: prio_block_frac 0.2 of each train country's S1 as whole geo blocks, chosen by
crc32(seed|block); S1 in the main-model hash sample prio_exclude_frac 0.3 excluded), current BlockingConfig defaults.
Writes /vol/exp/expC/prio_model_new.txt (+ .json report)."""
import copy
import json
import os
import sys
import time
import zlib

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl  # noqa: E402

import run_pipeline as RP  # noqa: E402
from ber import blocking as BL  # noqa: E402
from ber.config import PipelineConfig  # noqa: E402

OUT = "/vol/exp/expC"
cfg = PipelineConfig.from_json("/root/proj/code/business_entity_resolution/configs/submission_v3.json")
bcfg = copy.deepcopy(cfg.blocking)
bcfg.max_cands_per_s1, bcfg.prio_model = None, None
print("key passes", bcfg.key_passes, "prio_block_frac", cfg.prio_block_frac, "exclude", cfg.prio_exclude_frac, flush=True)
gt = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"])
countries = sorted(f[len("train_"):-len("_s1.parquet")] for f in os.listdir("/vol/work_v3/norm")
                   if f.startswith("train_") and f.endswith("_s1.parquet"))
parts, rep = [], {"countries": {}}
for c in countries:
    s1 = pl.read_parquet(f"/vol/work_v3/norm/train_{c}_s1.parquet")
    pool = pl.read_parquet(f"/vol/work_v3/norm/train_{c}_pool.parquet")
    cnt = s1.group_by(bcfg.block_col).len().filter(pl.col(bcfg.block_col) != "")
    order = sorted(cnt.rows(), key=lambda r: zlib.crc32(f"{cfg.seed}|{r[0]}".encode()))
    chosen, tot = [], 0
    for st, n in order:
        if tot >= cfg.prio_block_frac * s1.height:
            break
        chosen.append(st)
        tot += n
    s1s = s1.filter(pl.col(bcfg.block_col).is_in(chosen))
    t = time.time()
    u = BL.generate_candidates(s1s, pool, bcfg)
    u = u.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, pl.Int8).alias("label")),
               on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
    excl = cfg.prio_exclude_frac if cfg.prio_exclude_frac is not None else cfg.train_s1_frac
    main = s1s["entity_id"].filter(RP._train_sample(s1s["entity_id"], excl, cfg.seed))
    u = u.filter(~pl.col("s1").is_in(main.implode()))
    rep["countries"][c] = {"blocks": chosen, "s1": s1s.height, "rows": u.height, "pos": int(u["label"].sum()),
                           "seconds": round(time.time() - t, 1)}
    print(c, rep["countries"][c], flush=True)
    parts.append(u)
    del s1, pool
rep["train"] = BL.train_prio_model(pl.concat(parts, how="diagonal_relaxed"), bcfg, f"{OUT}/prio_model_new.txt")
print("PRIO", json.dumps(rep, default=str), flush=True)
with open(f"{OUT}/prio_model_new.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, default=str)
