# -*- coding: utf-8 -*-
"""crossenc.py -- transformer cross-encoders as stage-2 features (Master Bolt ER).

A cross-encoder reads the raw text of both records of a candidate pair,
    "S1 business_name | business_address"  [SEP]  "candidate business_name | business_address",
and outputs one logit (same business or not). It is fine-tuned from an open-weights multilingual model
(MIT / Apache-2.0, <= 8B parameters; see models_manifest.json) on TRAIN pairs only and applied to the pairs whose
stage-1 probability lies in an uncertain band (default [0.01, 0.99]: ~16% of the pairs with p >= 1e-3 on train,
~92% of the expected errors). Its logit and within-S1 context become stage-2 features (ber/stage2.py).

Honest populations (hash buckets metric.fold_of(s1, 1000, seed + 7919), the pipeline's sample hash):
    [0, 1000 * train_s1_frac)  stage-1 OOF sample = stage-2 training population (CE scores use the OOF p band)
    CrossEncModel.train_buckets  CE training S1 (disjoint from the sample; e.g. [300, 1000))
so no model ever scores a pair of an S1 it was trained on.

Training recipe (both submission cross-encoders): AdamW, linear warm-up/decay, BCE on the logit, gradient clipping 1.0,
length-bucketed batches, mixed precision when the hardware supports it (CUDA fp16/bf16, CPU bf16 on AVX512-BF16/AMX),
optional layer truncation (keep the first N transformer layers), optional frozen word-embedding table, and optional
synthetic test-like hard negatives: a true pair whose candidate house number (a number shared with the S1 address) is
shifted by 3-100 or has one digit changed by >= 3 is added as a NON-match (the label-free test-shift diagnosis found
exactly these distractors ~1.9x more often on test than on train).

The backbones are the only download (one time, pinned revision); afterwards every run can be made provably
offline with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 (the fine-tuned models are saved under <work>/crossenc/).
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import zlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

FEATURE_SUFFIXES = ("", "_rank", "_gap_top", "_gap_other")


@dataclass
class CrossEncModel:
    name: str = "ce"                              # feature prefix (ce, ce_rank, ce_gap_top, ce_gap_other)
    model: str = "intfloat/multilingual-e5-small"  # Hugging Face id of the open-weights backbone
    revision: Optional[str] = None                # pinned Hub commit sha of the backbone (models_manifest.json)
    layers: int = 0                               # keep the first N transformer layers (0 = all)
    train_buckets: List[int] = field(default_factory=lambda: [300, 450])   # [lo, hi) of the 1000 S1 hash buckets
    epochs: float = 1.5
    bs: int = 128
    lr: float = 5e-5
    max_len: int = 128
    warmup: float = 0.05
    freeze_embeddings: bool = False               # freeze the word-embedding table (large-vocabulary models)
    aug_frac: float = 0.0                         # share of positives given a shifted-house-number negative twin
    continue_epochs: float = 0.0                  # optional 2nd phase from the phase-1 weights (fresh schedule)
    continue_lr: float = 1e-5
    val_mod: int = 20                             # 1/val_mod of the CE training S1 (crc32) held out for the report
    seed: int = 0


def as_model(m) -> CrossEncModel:
    return m if isinstance(m, CrossEncModel) else CrossEncModel(**m)


def feature_names(models: Sequence) -> List[str]:
    return [as_model(m).name + s for m in models for s in FEATURE_SUFFIXES]


# ------------------------------------------------------------------------------------------------ texts / pairs
def record_texts(norm_path: str) -> pl.DataFrame:
    """entity_id, t = 'business_name | business_address' (raw strings) from a prepared (normalized) frame."""
    d = pl.read_parquet(norm_path, columns=["entity_id", "business_name", "business_address"])
    return d.select("entity_id", (pl.col("business_name").fill_null("") + " | "
                                  + pl.col("business_address").fill_null("")).alias("t"))


def attach_texts(pairs: pl.DataFrame, s1_norm: str, pool_norm: str) -> pl.DataFrame:
    """pairs (s1, cand, ...) + ta (S1 text) + tb (candidate text); row order kept."""
    s1t, poolt = record_texts(s1_norm), record_texts(pool_norm)
    return (pairs.join(s1t.rename({"entity_id": "s1", "t": "ta"}), on="s1", how="left", maintain_order="left")
                 .join(poolt.rename({"entity_id": "cand", "t": "tb"}), on="cand", how="left", maintain_order="left")
                 .with_columns(pl.col("ta").fill_null(""), pl.col("tb").fill_null("")))


_NUM = re.compile(r"(?<![0-9A-Za-z])(\d{1,5})(?![0-9A-Za-z])")


def shift_house_number(ta: str, tb: str, rng: random.Random) -> Optional[str]:
    """Candidate text with its house number (a number also in the S1 address) shifted like the test distractors
    (+-3..100, or one digit changed by >= 3); None when the pair has no shared number."""
    if " | " not in tb:
        return None
    a_addr = ta.split(" | ", 1)[1] if " | " in ta else ""
    name, addr = tb.split(" | ", 1)
    nums_a = set(_NUM.findall(a_addr))
    m = next((m for m in _NUM.finditer(addr) if m.group(1) in nums_a), None)
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


# ------------------------------------------------------------------------------------------------ torch helpers
def _torch():
    import torch                                  # heavy optional dependency: imported lazily
    return torch


def _device_and_amp(torch):
    """(device, autocast dtype or None, use GradScaler)."""
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return "cuda", (torch.bfloat16 if bf16 else torch.float16), not bf16
    ok = torch._C._cpu._is_avx512_bf16_supported() or torch._C._cpu._is_amx_tile_supported()
    torch.set_num_threads(int(os.environ.get("CE_THREADS", os.cpu_count() or 8)))
    return "cpu", (torch.bfloat16 if ok else None), False


def _load(model_id: str, layers: int = 0, revision: Optional[str] = None):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, revision=revision)
    if layers:
        enc = model.base_model.encoder
        enc.layer = enc.layer[:layers]
        model.config.num_hidden_layers = layers
    return tok, model


def _encode(tok, ta, tb, max_len, dev):
    e = tok(list(ta), list(tb), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    return {k: v.to(dev, non_blocking=True) for k, v in e.items() if k in ("input_ids", "attention_mask", "token_type_ids")}


def _predict(torch, net, tok, ta, tb, max_len, dev, amp, bs, log=None):
    net.eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(ta, tb)], kind="stable")
    out = np.empty(len(ta), dtype=np.float32)
    t = time.time()
    with torch.inference_mode():
        for k, i in enumerate(range(0, len(ta), bs)):
            idx = order[i:i + bs]
            with torch.autocast(dev, dtype=amp or torch.float32, enabled=amp is not None):
                z = net(**_encode(tok, [ta[j] for j in idx], [tb[j] for j in idx], max_len, dev)).logits
            out[idx] = z.float().squeeze(-1).cpu().numpy()
            if log and k % 500 == 0:
                log(f"crossenc: scored {i + len(idx):,}/{len(ta):,} ({(i + len(idx)) / max(time.time() - t, 1e-9):,.0f}/s)")
    return out


# ------------------------------------------------------------------------------------------------ train / score
def train(pairs: pl.DataFrame, mcfg, out_dir: str, log=print) -> Dict:
    """Fine-tune one cross-encoder. pairs: s1, ta, tb, label [, p] (CE training S1 only). Saves the model to out_dir
    and returns the report (validation AUC / logloss vs the stage-1 p on the same held-out pairs)."""
    torch = _torch()
    from sklearn.metrics import log_loss, roc_auc_score
    from transformers import get_linear_schedule_with_warmup
    m = as_model(mcfg)
    random.seed(m.seed)
    np.random.seed(m.seed)
    torch.manual_seed(m.seed)
    dev, amp, use_scaler = _device_and_amp(torch)
    pairs = pairs.sort(["s1", "tb"])
    isval = np.array([zlib.crc32(s.encode()) % m.val_mod == 0 for s in pairs["s1"].to_list()])
    val, tr = pairs.filter(pl.Series(isval)), pairs.filter(pl.Series(~isval))
    tok, model = _load(m.model, m.layers, m.revision)
    if m.freeze_embeddings:
        model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
    model = model.to(dev)
    net = torch.nn.DataParallel(model) if dev == "cuda" and torch.cuda.device_count() > 1 else model
    params = [p for p in model.parameters() if p.requires_grad]
    lossf = torch.nn.BCEWithLogitsLoss()

    def phase(seed: int, epochs: float, lr: float, tag: str):
        """One training phase: synthetic negatives drawn with `seed`, length-bucketed batches, AdamW with a fresh
        linear warm-up / decay schedule over `epochs`."""
        TA, TB = tr["ta"].to_list(), tr["tb"].to_list()
        Y = tr["label"].cast(pl.Float32).to_numpy().copy()
        n_aug = 0
        if m.aug_frac > 0:
            rng = random.Random(seed)
            extra = []
            for j in np.flatnonzero(Y == 1):
                if rng.random() < m.aug_frac:
                    nb = shift_house_number(TA[j], TB[j], rng)
                    if nb is not None:
                        extra.append((TA[j], nb))
            n_aug = len(extra)
            TA += [x for x, _ in extra]
            TB += [y for _, y in extra]
            Y = np.concatenate([Y, np.zeros(n_aug, np.float32)])
        n = len(TA)
        steps = int(math.ceil(n / m.bs) * epochs)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(m.warmup * steps), steps)
        scaler = torch.cuda.amp.GradScaler() if use_scaler else None
        log(f"crossenc {m.name} [{tag}]: {m.model} layers {m.layers or 'all'}, device {dev}, amp {amp}; train {len(tr):,} "
            f"pairs (+{n_aug:,} synthetic negatives), val {val.height:,}, {steps:,} steps, lr {lr}")
        # length-bucketed batches (shuffle, sort inside chunks of 50 batches, shuffle the batches), repeated per epoch
        lens = np.array([len(a) + len(b) for a, b in zip(TA, TB)])
        batches: List[np.ndarray] = []
        ep = 0
        while len(batches) < steps:
            perm = np.random.RandomState(seed + ep).permutation(n)
            chunk = []
            for c in range(0, n, m.bs * 50):
                ch = perm[c:c + m.bs * 50]
                ch = ch[np.argsort(lens[ch], kind="stable")]
                chunk += [ch[i:i + m.bs] for i in range(0, len(ch), m.bs)]
            random.Random(seed + 1000 + ep).shuffle(chunk)
            batches += chunk
            ep += 1
        net.train()
        t = time.time()
        seen = 0
        for step, idx in enumerate(batches[:steps]):
            y = torch.from_numpy(Y[idx]).to(dev)
            with torch.autocast(dev, dtype=amp or torch.float32, enabled=amp is not None):
                z = net(**_encode(tok, [TA[j] for j in idx], [TB[j] for j in idx], m.max_len, dev)).logits.float().squeeze(-1)
            loss = lossf(z, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            seen += len(idx)
            if step % 500 == 0 or step == steps - 1:
                log(f"crossenc {m.name} [{tag}]: step {step}/{steps} loss {loss.item():.4f} {seen / (time.time() - t):,.0f} pairs/s")
        return n_aug, steps

    t = time.time()
    n_aug, steps = phase(m.seed, m.epochs, m.lr, "phase 1")
    if m.continue_epochs > 0:                     # second phase from the phase-1 weights: new shuffle / negatives
        n_aug2, steps2 = phase(m.seed + 2, m.continue_epochs, m.continue_lr, "phase 2")
        n_aug, steps = n_aug + n_aug2, steps + steps2
    rep = {"config": asdict(m), "device": dev, "amp": str(amp), "train_pairs": len(tr), "synthetic_negatives": n_aug,
           "steps": steps, "params": int(sum(p.numel() for p in model.parameters())), "val_pairs": val.height}
    if val.height and 0 < int(val["label"].sum()) < val.height:
        zv = _predict(torch, net, tok, val["ta"].to_list(), val["tb"].to_list(), m.max_len, dev, amp, 512)
        yv = val["label"].to_numpy()
        rep.update(val_auc=float(roc_auc_score(yv, zv)),
                   val_logloss=float(log_loss(yv, np.clip(1 / (1 + np.exp(-zv)), 1e-7, 1 - 1e-7))))
        if "p" in val.columns:
            p1 = val["p"].to_numpy()
            rep.update(val_auc_stage1=float(roc_auc_score(yv, p1)),
                       val_logloss_stage1=float(log_loss(yv, np.clip(p1, 1e-7, 1 - 1e-7))))
    rep["seconds"] = round(time.time() - t)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "crossenc_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=str)
    log(f"crossenc {m.name}: trained; {({k: v for k, v in rep.items() if k.startswith('val_')})}")
    return rep


def score(model_dir: str, pairs: pl.DataFrame, max_len: int = 128, log=None) -> np.ndarray:
    """Logits for pairs (ta, tb), in row order."""
    torch = _torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev, amp, _ = _device_and_amp(torch)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev)
    if dev == "cuda" and amp is torch.float16:
        model = model.half()
    net = torch.nn.DataParallel(model) if dev == "cuda" and torch.cuda.device_count() > 1 else model
    return _predict(torch, net, tok, pairs["ta"].to_list(), pairs["tb"].to_list(), max_len, dev, amp,
                    1024 if dev == "cuda" else 512, log)


# ------------------------------------------------------------------------------------------------ stage-2 features
def add_features(frame: pl.DataFrame, scores: pl.DataFrame, name: str) -> pl.DataFrame:
    """frame (s1, cand, ...) + <name> (logit; null outside the band), <name>_rank (1 = best within the S1),
    <name>_gap_top (S1 max - logit), <name>_gap_other (margin to the best OTHER scored pair of the S1)."""
    f32 = pl.Float32
    c = pl.col(name)
    x = frame.join(scores.select("s1", "cand", pl.col("logit").cast(f32).alias(name)), on=["s1", "cand"], how="left",
                   maintain_order="left")
    mx = c.max().over("s1")
    second = c.drop_nulls().top_k(2).min().over("s1")
    return x.with_columns(
        c.rank("ordinal", descending=True).over("s1").cast(f32).alias(f"{name}_rank"),
        (mx - c).cast(f32).alias(f"{name}_gap_top"),
        pl.when(c >= mx).then(c - second).otherwise(c - mx).cast(f32).alias(f"{name}_gap_other"))
