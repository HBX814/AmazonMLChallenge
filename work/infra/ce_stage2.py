# -*- coding: utf-8 -*-
"""ce_stage2.py -- does the cross-encoder score improve stage 2?  Same frames, two stage-2 models:
    base = the v4 stage-2 features (the shipped model, refit here on the same frame for a like-for-like test)
    +ce  = base + cross-encoder features (ce logit, rank / gaps within the S1)
Measured on (1) the stage-2 OOF of the training sample (hash buckets < 150) and (2) the honest holdout HB
(buckets [150, 300): no stage-1 / stage-2 / cross-encoder model ever saw these S1), with the pipeline's decision
(expected-F, soft exclusivity, stage-1 lam_missing, empty_bias grid).
Inputs: /vol/work_v4 (v4 pipeline), /vol/exp/ce/{pfull_train_<c>, score_oof_<c>, score_hb_<c>}.parquet
Writes /vol/exp/ce/s2/{frame_<c>.parquet, model_base/, model_ce/, report.json}.
    python infra/ce_stage2.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402
from ber import model as MD         # noqa: E402
from ber import stage2 as S2        # noqa: E402

W, CE = Path("/vol/work_v4"), Path("/vol/exp/ce")
OUT = CE / "s2"
OUT.mkdir(parents=True, exist_ok=True)
CE_COLS = ["ce", "ce_rank", "ce_gap_top", "ce_gap_other"]
BIASES = [1.0, 1.5, 2.0, 3.0]
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def bucket(s):
    return pl.Series(M.fold_of(s, n_folds=1000, seed=7919))


def ce_feats(fr: pl.DataFrame, sc: pl.DataFrame) -> pl.DataFrame:
    """Join the cross-encoder logit and its within-S1 context (over the re-scored rows of the S1)."""
    f32 = pl.Float32
    x = fr.join(sc.select("s1", "cand", pl.col("ce").cast(f32)), on=["s1", "cand"], how="left", maintain_order="left")
    mx = pl.col("ce").max().over("s1")
    second = pl.col("ce").drop_nulls().top_k(2).min().over("s1")
    return x.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("s1").cast(f32).alias("ce_rank"),
        (mx - pl.col("ce")).cast(f32).alias("ce_gap_top"),
        pl.when(pl.col("ce") >= mx).then(pl.col("ce") - second).otherwise(pl.col("ce") - mx).cast(f32).alias("ce_gap_other"))


b1 = MD.load_bundle(str(W / "model"))
oof = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", "label", "p", "p_raw")
dec1 = json.load(open(W / "model" / "decision.json", encoding="utf-8"))["best"]
lam, eb = float(dec1["lam_missing"]), float(dec1.get("empty_bias", 2.0))
gt_all = bio.scan_split(W, "train")["gt"].collect()
cfg_base = S2.Stage2Config(orig_features=list(b1["feature_columns"]), floor=1e-3, lam_missing=lam, empty_bias=eb, guard="G1")
cfg_ce = S2.Stage2Config(orig_features=list(b1["feature_columns"]) + CE_COLS, floor=1e-3, lam_missing=lam,
                         empty_bias=eb, guard="G1")
log(f"stage-1 decision {dec1}")

frames, scored_all, pops = [], [], []
for c in ("India", "US"):
    fpath = OUT / f"frame_{c}.parquet"
    pf = pl.read_parquet(CE / f"pfull_train_{c}.parquet").select("s1", "cand", "p", "p_raw", "label")
    bk = bucket(pf["s1"])
    samp = pf.filter(bk < 150).select("s1").unique()
    own = oof.join(samp, on="s1", how="semi").select(pf.columns)
    scored = pl.concat([own, pf.filter(bk >= 150)], how="vertical_relaxed")
    del pf, own
    pop = scored.select("s1").unique().sort("s1")
    pop = pop.filter(bucket(pop["s1"]) < 300)
    if fpath.exists():
        fr = pl.read_parquet(fpath)
    else:
        t = time.time()
        fr = S2.build_frame(scored, str(W / "feats" / f"train_{c}.parquet"), str(W / "norm" / f"train_{c}_pool.parquet"),
                            cfg_base, population=pop)
        fr.write_parquet(fpath)
        log(f"frame {c}: {fr.height:,} rows x {fr.width} in {time.time() - t:.0f}s")
    if "--frames-only" in sys.argv:
        continue
    sc = pl.concat([pl.read_parquet(p) for k in ("oof", "hb")
                    for p in sorted(glob.glob(str(CE / f"score_{k}_{c}*.parquet")))])
    fr = ce_feats(fr, sc)
    log(f"{c}: ce coverage {fr['ce'].is_not_null().mean():.4f} of {fr.height:,} frame rows")
    frames.append(fr.with_columns(pl.lit(c).alias("country")))
    scored_all.append(scored.join(pop, on="s1", how="semi").select("s1", "cand", "label", "p").with_columns(pl.lit(c).alias("country")))
    pops.append(pop.with_columns(pl.lit(c).alias("country")))
    del scored

if "--frames-only" in sys.argv:
    log("frames written")
    sys.exit(0)
frame = pl.concat(frames, how="vertical_relaxed")
scored = pl.concat(scored_all, how="vertical_relaxed")
meta = pl.concat(pops)
del frames, scored_all
fb = bucket(frame["s1"])
tr_fr, hb_fr = frame.filter(fb < 150), frame.filter(fb >= 150)
mb = bucket(meta["s1"])
meta_tr, meta_hb = meta.filter(mb < 150), meta.filter(mb >= 150)
log(f"train frame {tr_fr.height:,} rows ({meta_tr.height:,} S1); holdout frame {hb_fr.height:,} rows ({meta_hb.height:,} S1)")


def evaluate(tag, sc_pop, meta_pop, res):
    """sc_pop: s1, cand, p (stage-1) of the population; res: s1, cand, p (stage 2) -> F0.5 per empty_bias."""
    merged = S2.merge(sc_pop.select("s1", "cand", "p"), res.select("s1", "cand", "p")) if res is not None else sc_pop
    gt = gt_all.join(meta_pop.select("s1"), on="s1", how="semi")
    out = {}
    for b in BIASES:
        links = D.select_links(merged, "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=b)
        row = {"all": M.macro_f05(links, gt, meta_pop["s1"])}
        for c in ("India", "US"):
            ids = meta_pop.filter(pl.col("country") == c)["s1"]
            row[c] = M.macro_f05(links.join(ids.to_frame(), on="s1", how="semi"), gt.join(ids.to_frame(), on="s1", how="semi"), ids)
        out[b] = row
        log(f"{tag} eb {b}: " + ", ".join(f"{k} {v:.5f}" for k, v in row.items()))
    return out


report = {"lam_missing": lam, "n_train_s1": meta_tr.height, "n_hb_s1": meta_hb.height}
sc_tr = scored.join(meta_tr.select("s1"), on="s1", how="semi")
sc_hb = scored.join(meta_hb.select("s1"), on="s1", how="semi")
report["stage1"] = {"oof": evaluate("stage1 OOF", sc_tr, meta_tr, None), "hb": evaluate("stage1 HB", sc_hb, meta_hb, None)}
for name, cfg in (("base", cfg_base), ("ce", cfg_ce)):
    t = time.time()
    b2, oof2 = S2.fit(tr_fr, cfg)
    S2.save_bundle(b2, str(OUT / f"model_{name}"))
    S2.merge(sc_tr.select("s1", "cand", "label", "p", "country"), oof2.select("s1", "cand", "p")).write_parquet(
        OUT / f"oof_{name}.parquet")                   # final OOF p (for the adaptation source priors)
    ph = S2.predict(b2, hb_fr)
    report[name] = {"oof": evaluate(f"{name} OOF", sc_tr, meta_tr, oof2), "hb": evaluate(f"{name} HB", sc_hb, meta_hb, ph),
                    "fit_report": b2["report"]}
    log(f"{name}: done in {time.time() - t:.0f}s; top gain {b2['report']['top_gain'][:12]}")
    json.dump(report, open(OUT / "report.json", "w"), indent=1, default=str)
log("done")
