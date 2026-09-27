# -*- coding: utf-8 -*-
"""expC_block.py -- full-scale blocking run with the v3 settings (configs/submission_v3.json blocking section +
learned prio /vol/work_v2/model/prio_model.txt) for one split/country and one variant:
  base  = v3 passes only (key_passes = the 6 old passes, reserve_slots 0)
  new   = current BlockingConfig defaults (+ akey / stname, reserve_slots) unless overridden on the CLI
Writes /vol/exp/expC/cands/<split>_<country>_<variant>.parquet and /vol/exp/expC/block_<split>_<country>_<variant>.json
(time, peak RSS, per-pass stats, recall on train).
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl  # noqa: E402

from ber import blocking as BL  # noqa: E402
from ber.config import PipelineConfig  # noqa: E402

OLD = ("name", "tok", "hn", "skel", "compact", "addr")
ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--country", required=True)
ap.add_argument("--variant", default="new")
ap.add_argument("--street-cap", type=int, default=None)
ap.add_argument("--street-block-cap", type=int, default=None)
ap.add_argument("--reserve", type=int, default=None)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--prio", default="/vol/work_v2/model/prio_model.txt")
a = ap.parse_args()
OUT = "/vol/exp/expC"
os.makedirs(f"{OUT}/cands", exist_ok=True)

cfg = PipelineConfig.from_json("/root/proj/code/business_entity_resolution/configs/submission_v3.json").blocking
cfg.prio_model = a.prio
cfg.n_threads = a.threads
if a.variant == "base":
    cfg.key_passes, cfg.reserve_slots = OLD, 0
if a.street_cap is not None:
    cfg.street_cap = a.street_cap
if a.street_block_cap is not None:
    cfg.street_block_cap = a.street_block_cap
if a.reserve is not None:
    cfg.reserve_slots = a.reserve
print("config:", {k: getattr(cfg, k) for k in ("key_passes", "tfidf_passes", "max_cands_per_s1", "city_alt_blocks", "street_cap",
                                               "street_block_cap", "reserve_slots", "reserve_passes", "prio_model", "n_threads")}, flush=True)

s1 = pl.read_parquet(f"/vol/work_v3/norm/{a.split}_{a.country}_s1.parquet")
pool = pl.read_parquet(f"/vol/work_v3/norm/{a.split}_{a.country}_pool.parquet")
stats = {}
t = time.time()
c = BL.generate_candidates(s1, pool, cfg, stats)
secs = time.time() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6          # KB -> GB (Linux)
rep = {"split": a.split, "country": a.country, "variant": a.variant, "seconds": round(secs, 1), "peak_rss_gb": round(peak, 2),
       "pairs": c.height, "n_s1": s1.height, "cands_per_s1": c.height / s1.height, "stats": stats}
if a.split == "train":
    gt = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"]).join(
        s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    c = c.join(gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, pl.Int8).alias("label")), on=["s1", "cand"], how="left",
               maintain_order="left").with_columns(pl.col("label").fill_null(0))
    rec = BL.candidate_recall(c, gt, s1.select("entity_id", "country"))
    rec.pop("by_country", None)
    rep["recall"] = rec
    print(f"RECALL {a.split}/{a.country}/{a.variant}: pair {rec['pair_recall']:.5f} cands/S1 {rec['cands_per_s1_mean']:.2f} "
          f"full {rec['s1_full_recall']:.5f} oracle {rec['oracle_f05']:.5f}", flush=True)
c.write_parquet(f"{OUT}/cands/{a.split}_{a.country}_{a.variant}.parquet")
with open(f"{OUT}/block_{a.split}_{a.country}_{a.variant}.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, default=str)
print("REPORT", json.dumps({k: v for k, v in rep.items() if k != "stats"}, default=str), flush=True)
print("STATS", json.dumps(stats, default=str), flush=True)
