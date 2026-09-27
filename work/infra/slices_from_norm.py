# -*- coding: utf-8 -*-
"""slices_from_norm.py -- geo dev slices cut from the pipeline's normalized frames (work/norm/*), so slice
measurements see exactly the normalization the full run uses. Same definition as er-data-loading
make_dev_slice.py geo: all train S1 with state_key == X, all same-country pool rows with state_key in {X, ""}.

    python work/infra/slices_from_norm.py states --work work --country US
    python work/infra/slices_from_norm.py geo --work work --country US --state texas --name US_TX

Writes work/dev/<name>/<name>_{s1,pool,links}.parquet (raw columns, measure_blocking input) and
<name>_{s1,pool}_norm.parquet (normalized; measure_blocking reuses them instead of re-normalizing).
"""
import argparse
import json
import os
import time

import polars as pl

RAW = ["entity_id", "business_name", "business_address", "country"]


def _safe(c):
    return "".join(ch if ch.isalnum() else "_" for ch in (c or "EMPTY"))


def states(work, country, split="train"):
    s1 = pl.scan_parquet(os.path.join(work, "norm", f"{split}_{_safe(country)}_s1.parquet"))
    pool = pl.scan_parquet(os.path.join(work, "norm", f"{split}_{_safe(country)}_pool.parquet"))
    a = s1.group_by("state_key").agg(pl.len().alias("n_s1"))
    b = pool.group_by("state_key").agg(pl.len().alias("n_pool"))
    return a.join(b, on="state_key", how="full", coalesce=True).fill_null(0).sort("n_s1", descending=True).collect()


def geo(work, country, state, name, split="train", extra_states=()):
    t = time.time()
    out = os.path.join(work, "dev", name)
    os.makedirs(out, exist_ok=True)
    s1 = pl.read_parquet(os.path.join(work, "norm", f"{split}_{_safe(country)}_s1.parquet")).filter(pl.col("state_key") == state)
    pool = (pl.scan_parquet(os.path.join(work, "norm", f"{split}_{_safe(country)}_pool.parquet"))
            .filter(pl.col("state_key").is_in([state, "", *extra_states])).collect())
    gt = (pl.scan_parquet(os.path.join(work, "cache", "train_gt_long.parquet"))
          .join(s1.lazy().select(pl.col("entity_id").alias("s1")), on="s1", how="semi").collect())
    in_pool = gt.join(pool.select(pl.col("entity_id").alias("mid")), on="mid", how="semi").height
    s1.select(RAW).write_parquet(os.path.join(out, f"{name}_s1.parquet"))
    pool.select(RAW).write_parquet(os.path.join(out, f"{name}_pool.parquet"))
    gt.write_parquet(os.path.join(out, f"{name}_links.parquet"))
    s1.write_parquet(os.path.join(out, f"{name}_s1_norm.parquet"))
    pool.write_parquet(os.path.join(out, f"{name}_pool_norm.parquet"))
    meta = {"name": name, "country": country, "state": state, "n_s1": s1.height, "n_pool": pool.height,
            "n_links": gt.height, "links_target_in_pool": in_pool, "recall_ceiling": in_pool / max(1, gt.height),
            "singleton_share": 1 - gt["s1"].n_unique() / max(1, s1.height), "seconds": round(time.time() - t, 1)}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["states", "geo"])
    ap.add_argument("--work", default="work")
    ap.add_argument("--country", required=True)
    ap.add_argument("--state")
    ap.add_argument("--name")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--extra-states", default="", help="comma list: pool states added to the slice (neighbours)")
    a = ap.parse_args()
    if a.cmd == "states":
        with pl.Config(tbl_rows=a.top + 5):
            print(states(a.work, a.country).head(a.top))
    else:
        geo(a.work, a.country, a.state, a.name, extra_states=[s for s in a.extra_states.split(",") if s])


if __name__ == "__main__":
    main()
