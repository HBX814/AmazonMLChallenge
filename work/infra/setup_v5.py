# -*- coding: utf-8 -*-
"""setup_v5.py -- work dir for the FINAL pipeline run (/vol/work_v5): reuses the finished v4 stages (prepare / prio /
block / features / stage-1 train, bit-identical when re-run) via symlinks and installs the two cross-encoders' outputs
in the pipeline's own format, so `run_pipeline.py --stage crossenc / predict / write` (configs/submission_v5.json)
trains stage 2 + the adaptation priors and writes the submission with the shipped code.
  crossenc/pfull_train_<c>.parquet  final-model stage-1 p of every train pair (same model; ce_data.py)
  crossenc/model_<name>/crossenc_report.json   training reports (ce: work/infra/ce_train.py on Modal CPU;
                                               ce2: Kaggle kernel harsh0814/mbolt-ce-v2), same recipe as ber/crossenc.train
  crossenc/scores_<name>.parquet    split (oof|test), country, s1, cand, logit  for the band pairs
    python infra/setup_v5.py [--ce2-scores /vol/exp/ce2/kg_scores.parquet] [--ce2-report <json>]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--ce2-scores", default="/vol/exp/ce2/kg_scores.parquet")
ap.add_argument("--ce2-report", default="/vol/exp/ce2/report_v2.json")
a = ap.parse_args()
W4, W5, CE1 = Path("/vol/work_v4"), Path("/vol/work_v5"), Path("/vol/exp/ce")
for d in ("model", "crossenc", "logs", "pred"):
    (W5 / d).mkdir(parents=True, exist_ok=True)


def link(src: Path, dst: Path):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


for d in ("cache", "norm", "cands", "feats", "native_token_dict.tsv"):
    link(W4 / d, W5 / d)
for f in ("pipeline_config.json", "decision.json", "oof.parquet", "model_final_0.txt", "bundle.json", "prio_model.txt"):
    link(W4 / "model" / f, W5 / "model" / f)
cdir = W5 / "crossenc"
for c in ("India", "US"):
    link(CE1 / f"pfull_train_{c}.parquet", cdir / f"pfull_train_{c}.parquet")
# ce: 6-layer e5-small (Modal CPU)
m1 = cdir / "model_ce"
m1.mkdir(exist_ok=True)
for f in glob.glob(str(CE1 / "model" / "*")):
    link(Path(f), m1 / Path(f).name)
rep1 = json.load(open(CE1 / "model" / "train_report.json"))
rep1["trained_by"] = "work/infra/ce_train.py on a 32-core Modal CPU container (recipe = ber/crossenc.train)"
json.dump(rep1, open(m1 / "crossenc_report.json", "w"), indent=1)
parts = []
for split, sets in (("oof", "oof"), ("test", "test")):
    for p in sorted(glob.glob(str(CE1 / f"score_{sets}_*.parquet"))):
        country = Path(p).name.split("_")[2].split(".")[0]
        parts.append(pl.read_parquet(p).select(pl.lit(split).alias("split"), pl.lit(country).alias("country"), "s1", "cand",
                                               pl.col("ce").alias("logit")))
s1 = pl.concat(parts)
s1.write_parquet(cdir / "scores_ce.parquet")
# ce2: bge-reranker-v2-m3 (Kaggle T4 x2)
m2 = cdir / "model_ce2"
m2.mkdir(exist_ok=True)
rep2 = json.load(open(a.ce2_report)) if os.path.exists(a.ce2_report) else {}
rep2["trained_by"] = "Kaggle GPU kernel (T4 x2), work/kaggle/ce_v2*/ce_kaggle.py (recipe = ber/crossenc.train); weights in the kernel output"
json.dump(rep2, open(m2 / "crossenc_report.json", "w"), indent=1)
kg = pl.read_parquet(a.ce2_scores)
s2 = kg.filter(~pl.col("set").str.starts_with("hb_")).select(
    pl.col("set").str.split("_").list.get(0).alias("split"), pl.col("set").str.split("_").list.get(1).alias("country"),
    "s1", "cand", pl.col("ce2").alias("logit"))
s2.write_parquet(cdir / "scores_ce2.parquet")
print("scores_ce", s1.group_by("split", "country").len().sort("split", "country").rows())
print("scores_ce2", s2.group_by("split", "country").len().sort("split", "country").rows())
print("work_v5 ready:", sorted(os.listdir(W5)), sorted(os.listdir(W5 / "model")), sorted(os.listdir(cdir)))
