# -*- coding: utf-8 -*-
"""slice_eval.py -- features / OOF / leave-one-country-out on labelled dev slices, using the assembled ber package.

    python infra/slice_eval.py feats --dev work/dev --name US_TX --cap 30
    python infra/slice_eval.py oof   --dev work/dev --name US_TX --cap 30
    python infra/slice_eval.py loco  --dev work/dev --train US_TX --eval IN_KA --cap 30

Inputs per slice (work/dev/<name>/): <name>_{s1,pool}_norm.parquet (slices_from_norm.py), <name>_links.parquet,
<name>_cands.parquet (measure_blocking.py, labelled, with `prio`).
oof : GroupKFold-by-S1 OOF -> isotonic calibration -> decision grid (threshold; expected-F x exclusivity none/soft
      x empty_bias) on OOF macro F0.5 over ALL slice S1 (singletons included). Saves model_cap<k>/ + oof + eval json.
loco: the France proxy. Model, calibrator AND decision parameters come from the TRAIN slice only; the eval slice
      is scored exactly like an unseen country at test time. Reported next to the eval slice's own OOF score.
"""
import argparse
import json
import os
import sys
import time

import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(HERE, "..", "code", "business_entity_resolution", "src"),
             os.path.join(HERE, "..", "..", "code", "business_entity_resolution", "src")):
    if os.path.isdir(os.path.join(cand, "ber")):
        sys.path.insert(0, os.path.abspath(cand))
        break

from ber import blocking as B      # noqa: E402
from ber import decide as D        # noqa: E402
from ber import features as F      # noqa: E402
from ber import metric as M        # noqa: E402
from ber import model as MD        # noqa: E402

THRESHOLDS = (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
BIASES = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def _p(dev, name, suffix):
    return os.path.join(dev, name, f"{name}_{suffix}")


def feats_path(dev, name, cap):
    return _p(dev, name, f"feats_cap{cap}.parquet")


def build_feats(dev, name, cap):
    out = feats_path(dev, name, cap)
    if os.path.exists(out):
        return pl.read_parquet(out)
    t = time.time()
    s1 = pl.read_parquet(_p(dev, name, "s1_norm.parquet"))
    pool = pl.read_parquet(_p(dev, name, "pool_norm.parquet"))
    cands = B.cap_candidates(pl.read_parquet(_p(dev, name, "cands.parquet")), cap)
    fe = F.compute_features(cands, s1, pool, F.FeatureConfig(chunk_pairs=1_000_000))
    fe.write_parquet(out)
    print(f"[{name}] features cap {cap}: {fe.height:,} pairs x {len(F.FEATURE_COLUMNS)} in {time.time() - t:.0f}s", flush=True)
    return fe


def _gt_ids(dev, name):
    gt = pl.read_parquet(_p(dev, name, "links.parquet")).select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8))
    ids = pl.read_parquet(_p(dev, name, "s1_norm.parquet"), columns=["entity_id"])["entity_id"]
    return gt, ids


def decision_grid(scored, gt, ids):
    """scored: s1, cand, p (+label). lam = true links lost by blocking, per S1."""
    lam = max(0.0, (gt.height - int(scored["label"].sum())) / max(1, len(ids)))
    rows = []
    for t in THRESHOLDS:
        rows.append({"method": "threshold", "exclusivity": "none", "t": t,
                     "f05": M.macro_f05(D.select_links(scored, "threshold", exclusivity="none", t=t), gt, ids)})
    for ex in ("none", "soft"):
        for b in BIASES:
            links = D.select_links(scored, "expected_f", exclusivity=ex, lam_missing=lam, empty_bias=b)
            rows.append({"method": "expected_f", "exclusivity": ex, "empty_bias": b, "lam_missing": lam,
                         "f05": M.macro_f05(links, gt, ids)})
    return rows, lam


def apply_decision(scored, best, lam=None):
    kw = {k: best[k] for k in ("t", "empty_bias") if k in best}
    if best["method"] == "expected_f":
        kw["lam_missing"] = best["lam_missing"] if lam is None else lam
    return D.select_links(scored, best["method"], exclusivity=best["exclusivity"], **kw)


def cmd_oof(dev, name, cap, folds):
    t = time.time()
    fe = build_feats(dev, name, cap)
    gt, ids = _gt_ids(dev, name)
    bundle, oof = MD.train_matcher(fe, MD.ModelConfig(n_folds=folds))
    mdir = os.path.join(dev, name, f"model_cap{cap}")
    MD.save_bundle(bundle, mdir)
    oof.write_parquet(_p(dev, name, f"oof_cap{cap}.parquet"))
    rows, lam = decision_grid(oof, gt, ids)
    best = max(rows, key=lambda r: r["f05"])
    oracle = oof.filter(pl.col("label") == 1).select("s1", pl.col("cand").alias("mid"))
    res = {"slice": name, "cap": cap, "n_s1": len(ids), "pairs": fe.height, "true_links": gt.height,
           "pair_recall": int(oof["label"].sum()) / max(1, gt.height), "lam_missing": lam,
           "oof_auc": bundle["report"]["oof_auc"], "best": best, "grid": rows,
           "oracle_f05": M.macro_f05(oracle, gt, ids),
           "links_per_s1_pred": apply_decision(oof, best).height / max(1, len(ids)),
           "links_per_s1_true": gt.height / max(1, len(ids)),
           "top_gain": bundle["report"]["top_gain"][:20], "seconds": round(time.time() - t)}
    with open(_p(dev, name, f"eval_cap{cap}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    for r in sorted(rows, key=lambda r: -r["f05"])[:6]:
        print(f"  {r['f05']:.5f}  {r}")
    print(f"[{name}] OOF macro F0.5 {best['f05']:.5f} | oracle {res['oracle_f05']:.5f} | AUC {res['oof_auc']:.5f} | "
          f"pair recall {res['pair_recall']:.4f} | pred links/S1 {res['links_per_s1_pred']:.2f} vs true "
          f"{res['links_per_s1_true']:.2f} | {res['seconds']}s", flush=True)
    return res


def cmd_loco(dev, a, b, cap):
    t = time.time()
    ev_a = json.load(open(_p(dev, a, f"eval_cap{cap}.json"), encoding="utf-8"))
    ev_b = json.load(open(_p(dev, b, f"eval_cap{cap}.json"), encoding="utf-8"))
    bundle = MD.load_bundle(os.path.join(dev, a, f"model_cap{cap}"))
    fb = build_feats(dev, b, cap)
    gt, ids = _gt_ids(dev, b)
    scored = MD.predict_matcher(bundle, fb).join(fb.select("s1", "cand", "label"), on=["s1", "cand"])
    auc = MD._auc(scored["label"].to_numpy(), scored["p"].to_numpy())
    best_a = ev_a["best"]
    # lam_missing is a blocking statistic of the TARGET data; at test time it is unknown -> use A's value
    f_transfer = M.macro_f05(apply_decision(scored, best_a), gt, ids)
    rows, _ = decision_grid(scored, gt, ids)            # oracle-tuned decision on B with A's model (diagnostic)
    best_b_params = max(rows, key=lambda r: r["f05"])
    res = {"train": a, "eval": b, "cap": cap, "auc": auc, "f05_transfer": f_transfer,
           "decision_from_train": best_a, "f05_best_decision_on_eval": best_b_params,
           "eval_own_oof_f05": ev_b["best"]["f05"], "eval_own_oof_auc": ev_b["oof_auc"],
           "gap_vs_own_oof": ev_b["best"]["f05"] - f_transfer,
           "links_per_s1_pred": apply_decision(scored, best_a).height / max(1, len(ids)),
           "links_per_s1_true": gt.height / max(1, len(ids)), "seconds": round(time.time() - t)}
    with open(os.path.join(dev, f"loco_{a}_to_{b}_cap{cap}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"[LOCO {a}->{b}] macro F0.5 {f_transfer:.5f} (decision from {a}) | best decision on {b}: "
          f"{best_b_params['f05']:.5f} | {b} own OOF {ev_b['best']['f05']:.5f} | gap {res['gap_vs_own_oof']:.5f} | "
          f"AUC {auc:.5f} | links/S1 {res['links_per_s1_pred']:.2f} vs true {res['links_per_s1_true']:.2f}", flush=True)
    return res


def cmd_links(dev, name, cap):
    """Best-decision OOF links of a slice -> <name>_links_best_cap<k>.parquet (input for error_report.py)."""
    ev = json.load(open(_p(dev, name, f"eval_cap{cap}.json"), encoding="utf-8"))
    oof = pl.read_parquet(_p(dev, name, f"oof_cap{cap}.parquet"))
    links = apply_decision(oof, ev["best"])
    out = _p(dev, name, f"links_best_cap{cap}.parquet")
    links.write_parquet(out)
    gt, ids = _gt_ids(dev, name)
    print(f"[{name}] {links.height:,} links -> {out}; macro F0.5 {M.macro_f05(links, gt, ids):.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "oof", "loco", "links"])
    ap.add_argument("--dev", default="work/dev")
    ap.add_argument("--name")
    ap.add_argument("--train")
    ap.add_argument("--eval")
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "feats":
        build_feats(a.dev, a.name, a.cap)
    elif a.cmd == "oof":
        cmd_oof(a.dev, a.name, a.cap, a.folds)
    elif a.cmd == "links":
        cmd_links(a.dev, a.name, a.cap)
    else:
        cmd_loco(a.dev, a.train, a.eval, a.cap)


if __name__ == "__main__":
    main()
