# -*- coding: utf-8 -*-
"""ce_train.py -- fine-tune a cross-encoder (intfloat/multilingual-e5-small, MIT; optionally truncated to its first
--layers transformer layers) on the ce_data.py training pairs: "S1 name | address" [SEP] "candidate name | address",
one logit, BCE loss. Only TRAIN S1 of hash buckets [300, 450) are used (never the OOF / holdout / test populations).
Only pairs whose stage-1 p lies in the uncertain band [--band-lo, --band-hi] are used (the band the model scores).
Runs on CUDA when available, else CPU (bf16 autocast on both).
    python infra/ce_train.py [--layers 6] [--epochs 2] [--bs 128] [--lr 5e-5] [--band-lo 0.01 --band-hi 0.99]
Writes /vol/exp/ce/model/ (weights + tokenizer + train_report.json).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import polars as pl
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "intfloat/multilingual-e5-small"
D, OUT = "/vol/exp/ce", "/vol/exp/ce/model"
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--max-len", type=int, default=128)
ap.add_argument("--layers", type=int, default=0, help="keep the first N transformer layers (0 = all)")
ap.add_argument("--band-lo", type=float, default=0.01)
ap.add_argument("--band-hi", type=float, default=0.99)
ap.add_argument("--val-mod", type=int, default=20, help="1/val-mod of the S1 (hash) are held out for validation")
a = ap.parse_args()
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(int(os.environ.get("CE_THREADS", "8")))
T0 = time.time()
AMP = None  # set below



def _amp_dtype(dev):
    """bf16 autocast only where the hardware has it (CUDA, or CPU with AVX512-BF16 / AMX); else fp32."""
    if dev == "cuda":
        return torch.bfloat16
    ok = torch._C._cpu._is_avx512_bf16_supported() or torch._C._cpu._is_amx_tile_supported()
    return torch.bfloat16 if ok else None


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *x, flush=True)


df = pl.concat([pl.read_parquet(f"{D}/cetrain_{c}.parquet", columns=["s1", "ta", "tb", "label", "p"]) for c in ("India", "US")])
df = df.filter(pl.col("p").is_between(a.band_lo, a.band_hi)).sort(["s1", "tb"])
isval = (pl.col("s1").hash(seed=11) % a.val_mod) == 0
val, tr = df.filter(isval), df.filter(~isval).sample(fraction=1.0, shuffle=True, seed=0)
AMP = _amp_dtype(dev)
log(f"autocast {AMP}")
log(f"device {dev}; band [{a.band_lo}, {a.band_hi}]: train {tr.height:,} pairs (pos {int(tr['label'].sum()):,}), "
    f"val {val.height:,} (pos {int(val['label'].sum()):,})")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
if a.layers:
    model.bert.encoder.layer = model.bert.encoder.layer[:a.layers]
    model.config.num_hidden_layers = a.layers
model = model.to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = int(math.ceil(tr.height / a.bs) * a.epochs)
sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
lossf = torch.nn.BCEWithLogitsLoss()
TA, TB, Y = tr["ta"].to_list(), tr["tb"].to_list(), tr["label"].cast(pl.Float32).to_numpy().copy()


def enc(ta, tb):
    e = tok(ta, tb, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in e.items()}


@torch.no_grad()
def predict(ta, tb, bs=512):
    model.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)], kind="stable")
    out = np.empty(len(ta), dtype=np.float32)
    for i in range(0, len(ta), bs):
        idx = order[i:i + bs]
        with torch.autocast(dev, dtype=AMP or torch.bfloat16, enabled=AMP is not None):
            out[idx] = model(**enc([ta[j] for j in idx], [tb[j] for j in idx])).logits.float().squeeze(-1).cpu().numpy()
    model.train()
    return out


model.train()
n = len(TA)
t = time.time()
for step in range(steps):
    i = (step * a.bs) % n
    idx = slice(i, min(i + a.bs, n))
    y = torch.from_numpy(Y[idx]).to(dev)
    with torch.autocast(dev, dtype=AMP or torch.bfloat16, enabled=AMP is not None):
        logits = model(**enc(TA[idx], TB[idx])).logits.float().squeeze(-1)
    loss = lossf(logits, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    if step % 200 == 0 or step == steps - 1:
        log(f"step {step}/{steps} loss {loss.item():.4f} {(step + 1) * a.bs / (time.time() - t):,.0f} pairs/s")
    if step and step % 2000 == 0:          # checkpoint (a restart can at least score with it)
        os.makedirs(OUT, exist_ok=True)
        model.save_pretrained(OUT)
        tok.save_pretrained(OUT)
z = predict(val["ta"].to_list(), val["tb"].to_list())
yv, p1 = val["label"].to_numpy(), val["p"].to_numpy()
pv = 1 / (1 + np.exp(-z))
rep = {"model": MODEL, "layers": a.layers or 12, "params": n_params, "device": dev, "band": [a.band_lo, a.band_hi],
       "train_pairs": tr.height, "steps": steps, "epochs": a.epochs, "bs": a.bs, "lr": a.lr, "max_len": a.max_len,
       "val_pairs": val.height, "val_auc_ce": float(roc_auc_score(yv, z)),
       "val_logloss_ce": float(log_loss(yv, np.clip(pv, 1e-7, 1 - 1e-7))),
       "val_auc_stage1": float(roc_auc_score(yv, p1)), "val_logloss_stage1": float(log_loss(yv, np.clip(p1, 1e-7, 1 - 1e-7))),
       "seconds": round(time.time() - T0)}
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
tok.save_pretrained(OUT)
json.dump(rep, open(f"{OUT}/train_report.json", "w"), indent=1)
log("done", rep)
