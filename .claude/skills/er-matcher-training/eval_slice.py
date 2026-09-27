# -*- coding: utf-8 -*-
"""eval_slice.py -- features -> LightGBM OOF -> decisions -> macro F0.5 on one labelled dev slice.

    python eval_slice.py --s1 work/dev/IN_KA/s1_norm.parquet --pool work/dev/IN_KA/pool_norm.parquet \
        --cands work/dev/IN_KA/cands.parquet --gt work/dev/IN_KA/links.parquet --cap 30 --out work/dev/IN_KA/eval.json

Inputs: normalized s1/pool frames (er-text-normalization), blocking candidates with `label` and `prio`
(er-blocking-candidates), GT long form (s1, mid) for the slice. Every S1 of --s1 is in the metric denominator.
Measured on the Karnataka slice (58,993 S1, cap 30): OOF AUC 0.99984, best macro F0.5 0.9772 (soft exclusivity +
expected-F, empty_bias 2), oracle 0.9931, 673 s total on the laptop.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import polars as pl

SK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, folder):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(SK, folder, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("s1", "pool", "cands", "gt", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args(argv)
    _load("normalize", "er-text-normalization")
    M = _load("metric", "er-f05-decisions")
    D = _load("decide", "er-f05-decisions")
    B = _load("blocking", "er-blocking-candidates")
    F = _load("features", "er-pair-features")
    MD = _load("model", "er-matcher-training")
    t0 = time.time()
    s1, pool, gt = pl.read_parquet(a.s1), pl.read_parquet(a.pool), pl.read_parquet(a.gt)
    cands = B.cap_candidates(pl.read_parquet(a.cands), a.cap)
    feats = F.compute_features(cands, s1, pool, F.FeatureConfig(chunk_pairs=600_000))
    bundle, oof = MD.train_matcher(feats, MD.ModelConfig(n_folds=a.folds))
    ids = s1["entity_id"]
    lam = (gt.height - int(oof["label"].sum())) / len(ids)
    rows = []
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        rows.append((f"threshold t={t}", M.macro_f05(D.select_links(oof, "threshold", exclusivity="none", t=t), gt, ids)))
    for ex in ("none", "soft"):
        for b in (1.0, 1.5, 2.0, 3.0):
            links = D.select_links(oof, "expected_f", exclusivity=ex, lam_missing=lam, empty_bias=b)
            rows.append((f"{ex}+expected_f bias={b}", M.macro_f05(links, gt, ids)))
    oracle = oof.filter(pl.col("label") == 1).select("s1", pl.col("cand").alias("mid"))
    res = {"cap": a.cap, "pairs": feats.height, "pair_recall": int(oof["label"].sum()) / gt.height,
           "lam_missing": lam, "oof_auc": bundle["report"]["oof_auc"], "decisions": rows,
           "oracle_f05": M.macro_f05(oracle, gt, ids), "top_gain": bundle["report"]["top_gain"][:20],
           "seconds": round(time.time() - t0)}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    for r in sorted(rows, key=lambda r: -r[1])[:5]:
        print(f"{r[1]:.5f}  {r[0]}")
    print(f"oracle {res['oracle_f05']:.5f}  AUC {res['oof_auc']:.5f}  pairs {feats.height:,}  {res['seconds']}s")


if __name__ == "__main__":
    main()
