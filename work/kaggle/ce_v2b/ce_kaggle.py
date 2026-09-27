# -*- coding: utf-8 -*-
"""ce_kaggle.py -- Master Bolt cross-encoder v2 on Kaggle (GPU T4 x2).
SECOND EPOCH: continues the fine-tuned BAAI/bge-reranker-v2-m3 of kernel mbolt-ce-v2 (epoch 1) on the same pairs, with a
new shuffle and new synthetic negatives (lr 1e-5, fresh warm-up/decay). Record-pair classifier on
kg_train.parquet ("name | address" of the S1 record vs the candidate record, label = same business), with synthetic
test-like hard negatives (the candidate's house number shifted by 3-100 or one digit changed by +-3..9), then scores
kg_eval.parquet. Outputs (/kaggle/working): kg_scores.parquet (set, s1, cand, ce2 logit), report.json, model/ (fp16).
Only the provided challenge data + an open-weights model; no external lookup.
"""
import glob
import json
import math
import os
import random
import re
import time
import zlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = [d for d in glob.glob("/kaggle/input/**/config.json", recursive=True) if "/model/" in d][0].rsplit("/", 1)[0]
EPOCHS, BS, LR, MAXLEN = 1.0, 64, 1e-5, 128
AUG_FRAC = 0.25            # share of positives with a shared house number that get a shifted-number negative twin
VAL_MOD = 40               # 1/40 of the train S1 (hash) held out
TRAIN_BUDGET_S = 7.5 * 3600
OUT = "/kaggle/working"
T0 = time.time()
torch.manual_seed(2)
random.seed(2)
np.random.seed(2)


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:6.0f}s]", *x, flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


NUM = re.compile(r"(?<![0-9A-Za-z])(\d{1,5})(?![0-9A-Za-z])")


def shift_hn(ta, tb, rng):
    """Candidate text with its house number (a number also present in the S1 address) shifted like the test
    distractors: +-3..100, or one digit changed by >= 3. None when not applicable."""
    a_addr = ta.split(" | ", 1)[1] if " | " in ta else ""
    if " | " not in tb:
        return None
    name, addr = tb.split(" | ", 1)
    nums_a = set(NUM.findall(a_addr))
    m = next((m for m in NUM.finditer(addr) if m.group(1) in nums_a), None)
    if m is None:
        return None
    n = m.group(1)
    if rng.random() < 0.5:
        new = str(max(1, int(n) + rng.choice([-1, 1]) * rng.randint(3, 100)))
    else:
        i = rng.randrange(len(n))
        d = int(n[i])
        ch = [x for x in range(10) if abs(x - d) >= 3 and not (i == 0 and x == 0 and len(n) > 1)]
        if not ch:
            return None
        new = n[:i] + str(rng.choice(ch)) + n[i + 1:]
    if new == n:
        return None
    return name + " | " + addr[:m.start(1)] + new + addr[m.end(1):]


# ------------------------------------------------------------------------------------------------ data
tr = pd.read_parquet(find("kg_train.parquet"))
isval = np.array([zlib.crc32(s.encode()) % VAL_MOD == 0 for s in tr["s1"]])
val, tr = tr[isval].reset_index(drop=True), tr[~isval].reset_index(drop=True)
rng = random.Random(20)
aug = []
for ta, tb in zip(tr.loc[tr["label"] == 1, "ta"], tr.loc[tr["label"] == 1, "tb"]):
    if rng.random() < AUG_FRAC:
        nb = shift_hn(ta, tb, rng)
        if nb is not None:
            aug.append((ta, nb))
TA = tr["ta"].tolist() + [a for a, _ in aug]
TB = tr["tb"].tolist() + [b for _, b in aug]
Y = np.concatenate([tr["label"].to_numpy(np.float32), np.zeros(len(aug), np.float32)])
log(f"train {len(tr):,} pairs (pos {int(tr['label'].sum()):,}) + {len(aug):,} synthetic hard negatives; val {len(val):,}")
# synthetic validation twins (sanity: the model must score the shifted twin below the true pair)
vrng = random.Random(1)
vtw = [(a, b, shift_hn(a, b, vrng)) for a, b in zip(val.loc[val["label"] == 1, "ta"], val.loc[val["label"] == 1, "tb"])]
vtw = [t for t in vtw if t[2] is not None][:5000]

# ------------------------------------------------------------------------------------------------ model
dev = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, torch_dtype=torch.float32).to(dev)
n_params = sum(p.numel() for p in model.parameters())
model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)   # 250k-token table: frozen (memory/speed)
trainable = [p for p in model.parameters() if p.requires_grad]
ngpu = torch.cuda.device_count()
net = torch.nn.DataParallel(model) if ngpu > 1 else model
log(f"model {MODEL}: {n_params / 1e6:.1f}M params ({sum(p.numel() for p in trainable) / 1e6:.1f}M trainable); "
    f"GPUs {ngpu} x {torch.cuda.get_device_name(0)}")
opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
n = len(TA)
steps = int(math.ceil(n / BS) * EPOCHS)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()


def enc(ta, tb):
    e = tok(ta, tb, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in e.items() if k in ("input_ids", "attention_mask")}


@torch.no_grad()
def predict(ta, tb, bs=512):
    net.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)], kind="stable")
    out = np.empty(len(ta), dtype=np.float32)
    t = time.time()
    for k, i in enumerate(range(0, len(ta), bs)):
        idx = order[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = net(**enc([ta[j] for j in idx], [tb[j] for j in idx])).logits.float().squeeze(-1).cpu().numpy()
        if k % 500 == 0:
            log(f"  predict {i + len(idx):,}/{len(ta):,} ({(i + len(idx)) / max(time.time() - t, 1e-9):,.0f}/s)")
    net.train()
    return out


# length-bucketed batches: shuffle, sort within chunks of 50 batches, shuffle the batches
perm = np.random.permutation(n)
lens = np.array([len(a) + len(b) for a, b in zip(TA, TB)])
batches = []
for c in range(0, n, BS * 50):
    ch = perm[c:c + BS * 50]
    ch = ch[np.argsort(lens[ch], kind="stable")]
    batches += [ch[i:i + BS] for i in range(0, len(ch), BS)]
random.Random(22).shuffle(batches)
batches = (batches * int(math.ceil(EPOCHS)))[:steps]

net.train()
t = time.time()
done = 0
for step, idx in enumerate(batches):
    y = torch.from_numpy(Y[idx]).to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = net(**enc([TA[j] for j in idx], [TB[j] for j in idx])).logits.float().squeeze(-1)
    loss = lossf(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    opt.zero_grad(set_to_none=True)
    done += len(idx)
    if step % 200 == 0 or step == len(batches) - 1:
        log(f"step {step}/{len(batches)} loss {loss.item():.4f} {done / (time.time() - t):,.0f} pairs/s")
    if time.time() - T0 > TRAIN_BUDGET_S:
        log(f"time budget reached at step {step}; stopping training")
        break

# ------------------------------------------------------------------------------------------------ validation
zv = predict(val["ta"].tolist(), val["tb"].tolist())
yv, p1 = val["label"].to_numpy(), val["p"].to_numpy()
rep = {"model": MODEL, "params": n_params, "train_pairs": len(tr), "synthetic_negatives": len(aug), "steps_done": step + 1,
       "steps_planned": len(batches), "bs": BS, "lr": LR, "max_len": MAXLEN, "val_pairs": len(val),
       "val_auc_ce": float(roc_auc_score(yv, zv)),
       "val_logloss_ce": float(log_loss(yv, np.clip(1 / (1 + np.exp(-zv)), 1e-7, 1 - 1e-7))),
       "val_auc_stage1": float(roc_auc_score(yv, p1)), "val_logloss_stage1": float(log_loss(yv, np.clip(p1, 1e-7, 1 - 1e-7)))}
if vtw:
    za = predict([a for a, _, _ in vtw], [b for _, b, _ in vtw])
    zb = predict([a for a, _, _ in vtw], [c for _, _, c in vtw])
    rep["twin_true_gt_shifted"] = float((za > zb).mean())
log("validation", rep)
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)

# ------------------------------------------------------------------------------------------------ scoring
ev = pd.read_parquet(find("kg_eval.parquet"))
log(f"scoring {len(ev):,} eval pairs")
ev["ce2"] = predict(ev["ta"].tolist(), ev["tb"].tolist(), bs=1024)
ev[["set", "s1", "cand", "ce2"]].to_parquet(f"{OUT}/kg_scores.parquet", index=False)
log("scores written:", ev.groupby("set")["ce2"].describe().to_string())
rep["seconds"] = round(time.time() - T0)
json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1)
model.half().save_pretrained(f"{OUT}/model")
tok.save_pretrained(f"{OUT}/model")
log("done")
