# -*- coding: utf-8 -*-
"""ce_score.py -- score pair sets with the fine-tuned cross-encoder (/vol/exp/ce/model); CUDA if available else CPU.
Only pairs whose stage-1 p lies in the band the model was trained on (train_report.json) are scored.
    python infra/ce_score.py oof_India oof_US hb_India hb_US [--shard k --nshards n]
Writes /vol/exp/ce/score_<set>[.part<k>].parquet (s1, cand, ce = logit); an existing output is skipped (resume).
Pairs are length-sorted for fast batching; shards split by a hash of s1.
"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

D, M = "/vol/exp/ce", "/vol/exp/ce/model"
MAXLEN = 128
ap = argparse.ArgumentParser()
ap.add_argument("sets", nargs="+")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(int(os.environ.get("CE_THREADS", "8")))
BS = 2048 if dev == "cuda" else 512
T0 = time.time()
AMP = None  # set below
rep = json.load(open(f"{M}/train_report.json"))
lo, hi = rep["band"]
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()



def _amp_dtype(dev):
    """bf16 autocast only where the hardware has it (CUDA, or CPU with AVX512-BF16 / AMX); else fp32."""
    if dev == "cuda":
        return torch.bfloat16
    ok = torch._C._cpu._is_avx512_bf16_supported() or torch._C._cpu._is_amx_tile_supported()
    return torch.bfloat16 if ok else None


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *x, flush=True)


AMP = _amp_dtype(dev)
log(f"autocast {AMP}")
log(f"device {dev}, band [{lo}, {hi}], layers {model.config.num_hidden_layers}")
for name in a.sets:
    out_path = f"{D}/score_{name}" + (f".part{a.shard}" if a.nshards > 1 else "") + ".parquet"
    if os.path.exists(out_path):
        log(f"{name}: exists, skipped")
        continue
    t = time.time()
    df = pl.read_parquet(f"{D}/{name}.parquet", columns=["s1", "cand", "p", "ta", "tb"]).filter(pl.col("p").is_between(lo, hi))
    if a.nshards > 1:
        df = df.filter((pl.col("s1").hash(seed=5) % a.nshards) == a.shard)
    df = df.with_row_index("_i").with_columns((pl.col("ta").str.len_chars() + pl.col("tb").str.len_chars()).alias("_l")).sort("_l")
    ta, tb = df["ta"].to_list(), df["tb"].to_list()
    out = np.empty(df.height, dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, df.height, BS):
            e = tok(ta[i:i + BS], tb[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
            e = {k: v.to(dev, non_blocking=True) for k, v in e.items()}
            with torch.autocast(dev, dtype=AMP or torch.bfloat16, enabled=AMP is not None):
                out[i:i + BS] = model(**e).logits.float().squeeze(-1).cpu().numpy()
            if (i // BS) % 200 == 0:
                log(f"{name}: {i + BS:,}/{df.height:,} ({(i + BS) / max(time.time() - t, 1e-9):,.0f}/s)")
    res = df.select("_i", "s1", "cand").with_columns(pl.Series("ce", out)).sort("_i").drop("_i")
    res.write_parquet(out_path + ".tmp")
    os.replace(out_path + ".tmp", out_path)
    log(f"{name}: {res.height:,} pairs in {time.time() - t:.0f}s ({res.height / max(time.time() - t, 1e-9):,.0f}/s)")
log("done")
