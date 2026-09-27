# -*- coding: utf-8 -*-
"""ce_bench.py -- CPU throughput of the cross-encoder candidates on realistic record pairs (S1 text vs S2 text).
Inference: fp32 / bf16 autocast / int8 dynamic quantisation; training step: fp32 / bf16; 12 vs 6 layers.
    python infra/ce_bench.py
"""
import copy
import os
import time

import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(int(os.environ.get("CE_THREADS", "8")))
MODEL = "intfloat/multilingual-e5-small"
N, BS = 2048, 256


def txt(p):
    d = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, n_rows=200_000, missing_utf8_is_empty_string=True)
    return (d["business_name"] + " | " + d["business_address"]).to_list()


a = txt("/vol/dataset/train/train_source1.tsv")
b = txt("/vol/dataset/train/train_source2.tsv")
ta, tb = a[:N], b[:N]
tok = AutoTokenizer.from_pretrained(MODEL)
e = tok(ta, tb, truncation=True, max_length=128)
lens = sorted(len(x) for x in e["input_ids"])
print(f"torch {torch.__version__} threads {torch.get_num_threads()}; pair tokens median {lens[N // 2]} p95 {lens[int(N * .95)]} "
      f"max {lens[-1]}; cpu bf16 support: {torch.backends.mkldnn.is_available()}", flush=True)
t = time.time()
for i in range(0, N, BS):
    tok(ta[i:i + BS], tb[i:i + BS], truncation=True, max_length=128, padding=True, return_tensors="pt")
print(f"tokenize: {N / (time.time() - t):,.0f} pairs/s", flush=True)
base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
order = sorted(range(N), key=lambda i: len(ta[i]) + len(tb[i]))
sa, sb = [ta[i] for i in order], [tb[i] for i in order]


def infer(model, mode, bs=512):
    model.eval()
    t = time.time()
    with torch.inference_mode():
        for i in range(0, N, bs):
            x = tok(sa[i:i + bs], sb[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="pt")
            if mode == "bf16":
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    model(**x)
            else:
                model(**x)
    return N / (time.time() - t)


def train(model, mode, steps=4):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    y = torch.zeros(BS)
    t = time.time()
    for s in range(steps):
        x = tok(ta[s * BS:(s + 1) * BS], tb[s * BS:(s + 1) * BS], truncation=True, max_length=128, padding=True, return_tensors="pt")
        if mode == "bf16":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(**x).logits.float().squeeze(-1)
        else:
            out = model(**x).logits.squeeze(-1)
        torch.nn.functional.binary_cross_entropy_with_logits(out, y).backward()
        opt.step()
        opt.zero_grad()
    return steps * BS / (time.time() - t)


for layers in (12, 6):
    m = copy.deepcopy(base)
    if layers < 12:
        m.bert.encoder.layer = m.bert.encoder.layer[:layers]
        m.config.num_hidden_layers = layers
    r = {"fp32": infer(m, "fp32"), "bf16": infer(m, "bf16")}
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {torch.nn.Linear}, dtype=torch.qint8)
    r["int8"] = infer(q, "fp32")
    r["train_fp32"] = train(copy.deepcopy(m), "fp32")
    r["train_bf16"] = train(copy.deepcopy(m), "bf16")
    print(f"layers {layers}: " + ", ".join(f"{k} {v:,.0f} pairs/s" for k, v in r.items()), flush=True)
