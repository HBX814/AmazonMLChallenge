# -*- coding: utf-8 -*-
"""cluster_eval.py -- OOF macro F0.5 of the v1v2 stage-2 model on the "France-like" subset: sample S1 that share
name_key + city_key with at least one other S1 of the same country (France test: 29.8% of S1; US 2.1%, India 24%),
vs the rest, per country; plus the same split for the stage-1 and CE-v1 models (how much each step helped there).
    python infra/cluster_eval.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402

W4, S2D, S2V1 = Path("/vol/work_v4"), Path("/vol/exp/ce2/s2"), Path("/vol/exp/ce/s2")
gt_all = bio.scan_split(W4, "train")["gt"].collect()
shared = []
for split in ("train", "test"):
    for c in (("India", "US") if split == "train" else ("France", "India", "US")):
        s1 = pl.read_parquet(W4 / "norm" / f"{split}_{c}_s1.parquet", columns=["entity_id", "name_key", "city_key"])
        g = s1.group_by("name_key", "city_key").agg(pl.len().alias("n"))
        s1 = s1.join(g, on=["name_key", "city_key"], how="left").with_columns((pl.col("n") > 1).alias("shared"))
        print(f"{split} {c}: share of S1 sharing name_key+city with another S1: {s1['shared'].mean():.4f}", flush=True)
        if split == "train":
            shared.append(s1.select(pl.col("entity_id").alias("s1"), "shared", pl.lit(c).alias("country")))
shared = pl.concat(shared)
lam = json.load(open(S2D / "tune_v1v2.json"))["lam_missing"]
models = {"stage1": (pl.read_parquet(W4 / "model" / "oof.parquet").select("s1", "cand", "p"), 2.0),
          "ce_v1": (pl.read_parquet(S2V1 / "oof_ce.parquet").select("s1", "cand", "p"), 1.5),
          "ce_v1v2": (pl.read_parquet(S2D / "oof_v1v2.parquet").select("s1", "cand", "p"), 1.0)}
out = {}
for name, (sc, eb) in models.items():
    links = D.select_links(sc, "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=eb)
    ids = sc.select("s1").unique().join(shared, on="s1", how="left")
    per = M.per_s1_scores(links, gt_all.join(ids.select("s1"), on="s1", how="semi"), ids["s1"]).join(ids, on="s1", how="left")
    r = per.group_by("country", "shared").agg(pl.len().alias("n"), pl.col("f").mean().alias("F")).sort("country", "shared")
    out[name] = r.rows()
    print(name, "ALL", round(float(per["f"].mean()), 5), r.rows(), flush=True)
json.dump(out, open(S2D / "cluster_eval.json", "w"), indent=1, default=str)
