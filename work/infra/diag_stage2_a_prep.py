# -*- coding: utf-8 -*-
"""diag_stage2_a_prep.py -- build the stage-2 frames (run inside the Modal container from /root/proj).

1. stage-1 p for EVERY train pair: OOF p for the 15% model-training sample (hash buckets < 150 of
   fold_of(s1, 1000, seed 7919) = run_pipeline._train_sample), final-model p (model_final_0 + isotonic calibrator)
   for all other S1. Buckets 150-299 ("HB", another 15%) are excluded from BOTH the stage-1 model and the prio model
   (prio_exclude_frac 0.3) -> an honest holdout scored like test (final model, full competition).
2. stage-2 frames for the rows with p1 >= FLOOR of the OOF population (grp 0) and of HB (grp 1):
   72 stage-1 features + within-S1 + stage-1-decision + competition (full train frame, and OOF-only for grp 0)
   + sibling features.
Outputs (/vol/diag/stage2/): p_full.parquet (s1, cand codes, grp, label, p1, p1raw), s2_oof.parquet, s2_hb.parquet,
names.parquet, meta.parquet, gt.parquet, prep_stats.json
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import numpy as np            # noqa: E402
import polars as pl           # noqa: E402
import psutil                 # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ber import metric as M   # noqa: E402
from ber import decide as D   # noqa: E402
from ber.features import FEATURE_COLUMNS as FC  # noqa: E402
import diag_stage2_lib as L   # noqa: E402

W3 = "/vol/work_v3"
OUT = "/vol/diag/stage2"
FLOOR = float(os.environ.get("S2_FLOOR", "1e-4"))
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
LAM, EB = 0.05303060038344832, 2.0
BREAKS = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0])   # bucket = upper bound
T0 = time.time()


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def countries():
    return sorted(os.path.basename(p)[len("train_"):-len(".parquet")] for p in glob.glob(f"{W3}/feats/train_*.parquet"))


def main():
    os.makedirs(OUT, exist_ok=True)
    stats = {"floor": FLOOR}
    cs = countries()
    log("countries", cs)
    # ---------------------------------------------------------------- S1 meta (fold, bucket, grp)
    meta = pl.concat([pl.read_parquet(f"{W3}/norm/train_{c}_s1.parquet", columns=["entity_id", "country"]) for c in cs])
    meta = meta.with_columns(L.id_code("entity_id").alias("s1"),
                             pl.Series("fold", M.fold_of(meta["entity_id"], 5, 0)),
                             pl.Series("bucket", M.fold_of(meta["entity_id"], 1000, 7919)))
    meta = meta.with_columns(pl.when(pl.col("bucket") < 150).then(0).when(pl.col("bucket") < 300).then(1)
                             .otherwise(2).cast(pl.Int8).alias("grp"))
    assert meta["s1"].n_unique() == meta.height
    meta.select("s1", "entity_id", "country", "fold", "grp").write_parquet(f"{OUT}/meta.parquet")
    stats["n_s1"] = meta.group_by("grp").len().sort("grp").rows()
    log("meta", stats["n_s1"])
    gt = pl.read_parquet("/vol/work/cache/train_gt_long.parquet").select(
        L.id_code("s1").alias("s1"), L.id_code("mid").alias("mid"))
    gt.write_parquet(f"{OUT}/gt.parquet")
    # ---------------------------------------------------------------- OOF
    oof = pl.read_parquet(f"{W3}/model/oof.parquet").select(
        L.id_code("s1").alias("s1"), L.id_code("cand").alias("cand"), pl.col("label").cast(pl.Int8),
        pl.col("p").cast(pl.Float32).alias("p1"), pl.col("p_raw").cast(pl.Float32).alias("p1raw"))
    oof_s1 = oof.select("s1").unique()
    g0 = meta.filter(pl.col("grp") == 0).select("s1")
    stats["oof_s1"] = oof_s1.height
    stats["oof_s1_not_grp0"] = oof_s1.join(g0, on="s1", how="anti").height
    log(f"OOF {oof.height:,} pairs, {oof_s1.height:,} S1; not in grp0: {stats['oof_s1_not_grp0']}")
    oof_keep = oof.filter(pl.col("p1") >= FLOOR).select("s1", "cand")
    # ---------------------------------------------------------------- stream stage-1 features, predict non-OOF rows
    import lightgbm as lgb
    bundle = json.load(open(f"{W3}/model/bundle.json", encoding="utf-8"))
    cx, cy = (np.asarray(v, dtype=np.float64) for v in bundle["calibrator"])
    cols = bundle["feature_columns"]
    if cols != list(FC):
        log("WARNING: bundle feature columns differ from ber.features.FEATURE_COLUMNS; using the bundle's")
    booster = lgb.Booster(model_file=f"{W3}/model/model_final_0.txt")
    grp_map = meta.select("s1", "grp")
    pparts, fparts = [], []
    n_pred, t_pred = 0, 0.0
    resume = os.path.exists(f"{OUT}/p_full.parquet") and os.path.exists(f"{OUT}/feats_keep.parquet")
    for c in ([] if resume else cs):
        pf = pq.ParquetFile(f"{W3}/feats/train_{c}.parquet")
        log(f"stream train_{c}: {pf.metadata.num_rows:,} rows")
        for rb in pf.iter_batches(batch_size=2_000_000, columns=["s1", "cand", "label"] + cols):
            b = pl.from_arrow(rb).with_columns(L.id_code("s1").alias("s1"), L.id_code("cand").alias("cand"))
            b = b.join(grp_map, on="s1", how="left")
            assert b["grp"].null_count() == 0
            nb = b.filter(pl.col("grp") != 0)
            t = time.time()
            raw = booster.predict(nb.select([pl.col(x).cast(pl.Float32) for x in cols]).to_numpy(), num_threads=THREADS)
            t_pred += time.time() - t
            n_pred += nb.height
            p = np.interp(raw, cx, cy)
            nb = nb.with_columns(pl.Series("p1", p.astype(np.float32)), pl.Series("p1raw", raw.astype(np.float32)))
            pparts.append(nb.select("s1", "cand", "grp", pl.col("label").cast(pl.Int8), "p1", "p1raw"))
            fparts.append(b.filter(pl.col("grp") == 0).join(oof_keep, on=["s1", "cand"], how="semi")
                          .select(["s1", "cand"] + cols))
            fparts.append(nb.filter((pl.col("grp") == 1) & (pl.col("p1") >= FLOOR)).select(["s1", "cand"] + cols))
    stats["predict_rows"], stats["predict_s"] = n_pred, round(t_pred, 1)
    log(f"predicted {n_pred:,} rows in {t_pred:.0f}s ({n_pred / max(t_pred, 1e-9):,.0f} rows/s, {THREADS} threads)")
    if resume:
        log("resume: loading p_full.parquet + feats_keep.parquet")
        pfull, feats = pl.read_parquet(f"{OUT}/p_full.parquet"), pl.read_parquet(f"{OUT}/feats_keep.parquet")
    else:
        pfull = pl.concat(pparts + [oof.with_columns(pl.lit(0, pl.Int8).alias("grp"))
                                    .select("s1", "cand", "grp", "label", "p1", "p1raw")])
        del pparts
        feats = pl.concat(fparts)
        del fparts
        pfull.write_parquet(f"{OUT}/p_full.parquet")
        feats.write_parquet(f"{OUT}/feats_keep.parquet")
    log(f"p_full {pfull.height:,} rows; feature rows kept {feats.height:,}")
    stats["p_full_rows"] = pfull.height
    # ---------------------------------------------------------------- sanity: stage-1 decision on OOF
    ev0 = meta.filter(pl.col("grp") == 0).select("s1", "country")
    gt0 = gt.join(ev0, on="s1", how="semi")
    t = time.time()
    links = D.select_links(oof.select("s1", "cand", pl.col("p1").alias("p")), "expected_f", exclusivity="soft",
                           lam_missing=LAM, empty_bias=EB)
    stats["stage1_oof_pipeline"] = L.score(links, gt0, ev0)
    stats["t_select_links_full_s"] = round(time.time() - t, 1)
    t = time.time()
    links_f = L.select_fast(oof.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, EB)
    stats["stage1_oof_fast"] = L.score(links_f, gt0, ev0)
    stats["t_select_fast_s"] = round(time.time() - t, 1)
    log("stage-1 OOF", stats["stage1_oof_pipeline"], stats["t_select_links_full_s"], "s | fast",
        stats["stage1_oof_fast"], stats["t_select_fast_s"], "s")
    # p1 buckets: pairs, positives, positives selected by stage 1
    sel = links.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("_sel"))
    bk = (oof.with_columns(pl.Series("b", BREAKS[np.searchsorted(BREAKS, oof["p1"].to_numpy(), side="left")]))
             .join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("_sel").fill_null(0))
             .group_by("b").agg(pl.len().alias("pairs"), pl.col("label").sum().alias("pos"),
                                (pl.col("label") * pl.col("_sel")).sum().alias("pos_sel"),
                                ((1 - pl.col("label")) * pl.col("_sel")).sum().alias("neg_sel"))
             .sort("b"))
    stats["p1_buckets_oof"] = bk.rows()
    print(bk)
    # ---------------------------------------------------------------- names for candidates of the rescored rows
    need = pfull.filter((pl.col("grp") <= 1) & (pl.col("p1") >= FLOOR)).select("cand").unique()
    names = pl.concat([pl.read_parquet(f"{W3}/norm/train_{c}_pool.parquet",
                                       columns=["entity_id", "name_core", "addr_norm", "name_key"]) for c in cs])
    names = (names.select(L.id_code("entity_id").alias("cand"), pl.col("name_core").fill_null("").alias("nc"),
                          pl.col("addr_norm").fill_null("").alias("ad"), pl.col("name_key").fill_null("").alias("nk"))
                  .join(need, on="cand", how="semi"))
    names.write_parquet(f"{OUT}/names.parquet")
    log(f"names {names.height:,} (needed {need.height:,})")
    # ---------------------------------------------------------------- competition aggregates over the FULL frame
    t = time.time()
    agg_full = L.competition_aggregates(pfull.select("s1", "cand", "p1"), need)
    stats["t_comp_agg_s"] = round(time.time() - t, 1)
    agg_full.select("cand", "_osum").write_parquet(f"{OUT}/osum_full.parquet")
    log(f"competition aggregates over {pfull.height:,} rows: {agg_full.height:,} records, {time.time() - t:.0f}s")
    # ---------------------------------------------------------------- stage-2 frames
    for grp, tag in ((0, "oof"), (1, "hb")):
        tt = time.time()
        e = pfull.filter(pl.col("grp") == grp).select("s1", "cand", "label", "p1", "p1raw")
        e = L.within_s1_feats(e, "p1", "p1raw")
        e = L.decision_feats(e, LAM, EB, "p1")
        t_w = time.time() - tt
        Fr = e.filter(pl.col("p1") >= FLOOR)
        tt2 = time.time()
        Fr = L.competition_feats(Fr, agg=agg_full)
        if grp == 0:   # competition limited to the 15% OOF frame (what an OOF-only stage 2 would see)
            Fr = L.competition_feats(Fr, e.select("s1", "cand", "p1"), suffix="_oo")
        t_c = time.time() - tt2
        tt3 = time.time()
        Fr = L.sibling_feats(Fr, names, "p1")
        t_s = time.time() - tt3
        Fr = Fr.join(feats, on=["s1", "cand"], how="left")
        miss = Fr[cols[0]].null_count()
        Fr = Fr.join(meta.select("s1", "fold", "country"), on="s1", how="left")
        Fr.write_parquet(f"{OUT}/s2_{tag}.parquet")
        stats[f"{tag}_rows_all"] = e.height
        stats[f"{tag}_rows_rescored"] = Fr.height
        stats[f"{tag}_pos_all"] = int(e["label"].sum())
        stats[f"{tag}_pos_rescored"] = int(Fr["label"].sum())
        stats[f"{tag}_missing_feat_rows"] = int(miss)
        stats[f"{tag}_t_within_s"], stats[f"{tag}_t_comp_s"], stats[f"{tag}_t_sib_s"] = \
            round(t_w, 1), round(t_c, 1), round(t_s, 1)
        log(f"s2_{tag}: {Fr.height:,} rescored rows of {e.height:,} ({stats[f'{tag}_pos_rescored']:,} of "
            f"{stats[f'{tag}_pos_all']:,} positives), missing 72-feat rows {miss}; within {t_w:.0f}s comp {t_c:.0f}s "
            f"sib {t_s:.0f}s")
        if grp == 0:
            neg = Fr.filter((pl.col("label") == 0) & (pl.col("p1") >= 0.3))
            pos = Fr.filter(pl.col("label") == 1)
            stats["limitation"] = {
                "mean_n_other_full": float(Fr["c_n_other"].mean()), "mean_n_other_oofonly": float(Fr["c_n_other_oo"].mean()),
                "neg_p03_n": neg.height,
                "neg_p03_share_other_max_ge_0.5_full": float((neg["c_other_max"] >= 0.5).mean()),
                "neg_p03_share_other_max_ge_0.5_oofonly": float((neg["c_other_max_oo"] >= 0.5).mean()),
                "pos_share_is_best_full": float(pos["c_is_best"].mean()),
                "pos_share_is_best_oofonly": float(pos["c_is_best_oo"].mean())}
            log("limitation", stats["limitation"])
        del e, Fr
    stats["seconds"] = round(time.time() - T0, 1)
    stats["peak_rss_gb"] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    with open(f"{OUT}/prep_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, default=str)
    log("done", json.dumps({k: v for k, v in stats.items() if k != "p1_buckets_oof"}, default=str))


if __name__ == "__main__":
    main()
