# -*- coding: utf-8 -*-
"""diag_stage2_b_train.py -- stage-2 LightGBM on the stage-1 OOF (frozen folds) + decision grid, ablations and the
honest-holdout (HB) transfer test. Needs the outputs of diag_stage2_a_prep.py in /vol/diag/stage2/.

    python infra/diag_stage2_b_train.py stage1 full [--final] [--grid full|small] [--floor 1e-3]
    variants: stage1 (no model: grid on p1), full, no_sib, no_comp, comp_oo, no_orig, within_only
Writes /vol/diag/stage2/train_<variant>.json (+ oof_p2_<variant>.parquet, model_s2_<variant>.txt with --final).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import numpy as np            # noqa: E402
import polars as pl           # noqa: E402
import psutil                 # noqa: E402

from ber.features import FEATURE_COLUMNS as FC  # noqa: E402
import diag_stage2_lib as L   # noqa: E402

OUT = "/vol/diag/stage2"
LAM = 0.05303060038344832
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
          "verbose": -1, "num_threads": 0, "seed": 0, "deterministic": True, "force_row_wise": True}
ORIG = list(FC)
NEW_W = L.W_FEATS + L.D_FEATS
VARIANTS = {
    "full": ORIG + NEW_W + L.c_feats() + L.S_FEATS,
    "no_sib": ORIG + NEW_W + L.c_feats(),
    "no_comp": ORIG + NEW_W + L.S_FEATS,
    "comp_oo": ORIG + NEW_W + L.c_feats("_oo") + L.S_FEATS,
    "no_orig": NEW_W + L.c_feats() + L.S_FEATS,
    "within_only": ORIG + NEW_W,
}
EB_FULL, EB_SMALL = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0], [1.5, 2.0, 3.0]
THR = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
T0 = time.time()


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def auc(y, p):
    from sklearn.metrics import roc_auc_score, log_loss
    return float(roc_auc_score(y, p)), float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def grid(scored, gt, ev, eb_grid, thr=True, excl=("none", "soft")):
    rows = []
    for ex in excl:
        for eb in eb_grid:
            t = time.time()
            s = L.score(L.select_fast(scored, ex, LAM, eb), gt, ev)
            rows.append({"method": "expected_f", "exclusivity": ex, "empty_bias": eb, **s, "sec": round(time.time() - t, 1)})
            log(f"    {ex:5s} eb {eb:<4} -> {s}")
    if thr:
        for t_ in THR:
            s = L.score(L.select_threshold(scored, t_), gt, ev)
            rows.append({"method": "threshold", "exclusivity": "none", "t": t_, **s})
    return rows


def links_for(scored, best):
    if best["method"] == "threshold":
        return L.select_threshold(scored, best["t"])
    return L.select_fast(scored, best["exclusivity"], LAM, best["empty_bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--final", action="store_true", help="train the final stage-2 model and score HB")
    ap.add_argument("--grid", default="full")
    ap.add_argument("--floor", type=float, default=1e-3)
    a = ap.parse_args()
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    gt = pl.read_parquet(f"{OUT}/gt.parquet")
    ev0 = meta.filter(pl.col("grp") == 0).select("s1", "country")
    ev1 = meta.filter(pl.col("grp") == 1).select("s1", "country")
    gt0, gt1 = gt.join(ev0, on="s1", how="semi"), gt.join(ev1, on="s1", how="semi")
    pf = pl.scan_parquet(f"{OUT}/p_full.parquet")
    all0 = pf.filter(pl.col("grp") == 0).select("s1", "cand", "p1").collect()
    fr = pl.read_parquet(f"{OUT}/s2_oof.parquet").filter(pl.col("p1") >= a.floor)
    log(f"OOF edges {all0.height:,}; stage-2 rows (p1 >= {a.floor}) {fr.height:,}, positives {int(fr['label'].sum()):,}")
    y = fr["label"].to_numpy()
    folds = fr["fold"].to_numpy()
    ebg = EB_FULL if a.grid == "full" else EB_SMALL
    for v in a.variants:
        res = {"variant": v, "floor": a.floor, "rows": fr.height}
        t = time.time()
        if v == "stage1":
            scored = all0.select("s1", "cand", pl.col("p1").alias("p"))
            res["auc_logloss_rescored_rows"] = auc(y, fr["p1"].to_numpy())
        else:
            cols = VARIANTS[v]
            res["n_features"] = len(cols)
            raw, iters, gain = L.fit_oof(fr, cols, folds, PARAMS)
            res["t_train_s"] = round(time.time() - t, 1)
            res["best_iters"] = iters
            kx, ky = L.isotonic(raw, y)
            p2 = np.interp(raw, kx, ky)
            res["auc_logloss_rescored_rows"] = auc(y, p2)
            res["auc_logloss_stage1_same_rows"] = auc(y, fr["p1"].to_numpy())
            res["top_gain"] = sorted(((k, round(g)) for k, g in gain.items()), key=lambda kv: -kv[1])[:30]
            grp_gain = {"orig72": 0.0, "within": 0.0, "decision": 0.0, "competition": 0.0, "sibling": 0.0}
            for k, g in gain.items():
                grp_gain["orig72" if k in ORIG else "within" if k in L.W_FEATS else "decision" if k in L.D_FEATS
                         else "sibling" if k.startswith("sib_") else "competition"] += g
            tot = sum(grp_gain.values()) or 1.0
            res["gain_share_by_group"] = {k: round(g / tot, 4) for k, g in grp_gain.items()}
            rescored = fr.select("s1", "cand").with_columns(pl.Series("p2", p2.astype(np.float32)))
            rescored.with_columns(pl.Series("raw", raw.astype(np.float32))).write_parquet(f"{OUT}/oof_p2_{v}.parquet")
            scored = L.merge_p(all0, rescored)
            log(f"{v}: trained in {res['t_train_s']}s, iters {iters}, AUC/logloss s2 {res['auc_logloss_rescored_rows']}"
                f" vs s1 {res['auc_logloss_stage1_same_rows']}")
            log(f"{v}: gain share {res['gain_share_by_group']}")
            log(f"{v}: top gain {res['top_gain'][:15]}")
        rows = grid(scored, gt0, ev0, ebg if v in ("stage1", "full") else EB_SMALL,
                    thr=v in ("stage1", "full"), excl=("none", "soft") if v in ("stage1", "full") else ("soft",))
        best = max(rows, key=lambda r: r["ALL"])
        res["grid"], res["best"] = rows, best
        bl = links_for(scored, best)
        res["ladder"] = L.ladder(scored, bl, gt0, ev0)
        res["links_per_s1"] = bl.height / ev0.height
        log(f"{v}: BEST {best}")
        log(f"{v}: ladder {res['ladder']}")
        if a.final and v != "stage1":
            t = time.time()
            n_final = int(round(1.1 * float(np.mean(iters))))
            b = L.fit_final(fr, cols, PARAMS, n_final)
            res["t_final_train_s"] = round(time.time() - t, 1)
            b.save_model(f"{OUT}/model_s2_{v}.txt")
            json.dump({"x": kx.tolist(), "y": ky.tolist(), "cols": cols, "floor": a.floor, "rounds": n_final},
                      open(f"{OUT}/model_s2_{v}_calib.json", "w"))
            hb = pl.read_parquet(f"{OUT}/s2_hb.parquet").filter(pl.col("p1") >= a.floor)
            t = time.time()
            raw_hb = b.predict(hb.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy())
            res["t_predict_hb_s"], res["hb_rows"] = round(time.time() - t, 1), hb.height
            p2_hb = np.interp(raw_hb, kx, ky)
            yh = hb["label"].to_numpy()
            res["hb_auc_logloss_s2"], res["hb_auc_logloss_s1"] = auc(yh, p2_hb), auc(yh, hb["p1"].to_numpy())
            all1 = pf.filter(pl.col("grp") == 1).select("s1", "cand", "p1").collect()
            sc1 = all1.select("s1", "cand", pl.col("p1").alias("p"))
            sc2 = L.merge_p(all1, hb.select("s1", "cand").with_columns(pl.Series("p2", p2_hb.astype(np.float32))))
            osum = L.external_osum(pl.read_parquet(f"{OUT}/osum_full.parquet"), all1)
            hbres = {}
            for name, sc in (("stage1", sc1), ("stage2", sc2)):
                for ex in ("none", "soft", "soft_ext"):
                    for eb in EB_SMALL:
                        s = L.score(L.select_fast(sc, ex, LAM, eb, osum_ext=osum if ex == "soft_ext" else None), gt1, ev1)
                        hbres[f"{name}|{ex}|{eb}"] = s
                        log(f"  HB {name} {ex} eb {eb}: {s}")
            s1b = L.select_fast(sc1, "soft", LAM, 2.0)
            hbres["stage1_ladder_soft_eb2"] = L.ladder(sc1, s1b, gt1, ev1)
            bh = links_for(sc2, best)
            hbres["stage2_oofbest"] = L.score(bh, gt1, ev1)
            hbres["stage2_ladder_oofbest"] = L.ladder(sc2, bh, gt1, ev1)
            res["hb"] = hbres
            log(f"{v}: HB stage-2 with OOF-best decision {hbres['stage2_oofbest']}")
        res["seconds"] = round(time.time() - t, 1)
        res["peak_rss_gb"] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
        with open(f"{OUT}/train_{v}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
