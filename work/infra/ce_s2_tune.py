# -*- coding: utf-8 -*-
"""ce_s2_tune.py -- stage-2 + cross-encoder variants on the cached ce_stage2 frames (same OOF / HB populations, same
folds, same decision). Variants (--variants, comma list):
    ce4      the 4 CE columns of ce_stage2 (reference: OOF 0.98752 / HB 0.98755)
    ce8      + ce_minus_logit, ce_sig_sum (CE expected matches in the S1), ce_sig_other, ce_npos
    ce8_lr05 ce8 with lr 0.05, 255 leaves, min_data_in_leaf 50
Reads /vol/exp/ce/s2/frame_<c>.parquet + /vol/exp/ce/score_{oof,hb}_<c>*.parquet (or --scores-tag <t> for
score_<t>_{oof,hb}_...). Writes /vol/exp/ce/s2/tune_<variant>.json and model_<variant>/, oof_<variant>.parquet.
    python infra/ce_s2_tune.py --variants ce4,ce8,ce8_lr05
"""
import argparse
import glob
import json
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

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="ce4,ce8,ce8_lr05")
ap.add_argument("--scores-tag", default="", help="score file prefix tag ('' = CE v1 scores)")
a = ap.parse_args()
W, CE = Path("/vol/work_v4"), Path("/vol/exp/ce")
OUT = CE / "s2"
BIASES = [1.0, 1.5, 2.0]
CE4 = ["ce", "ce_rank", "ce_gap_top", "ce_gap_other"]
CE8 = CE4 + ["ce_minus_logit", "ce_sig_sum", "ce_sig_other", "ce_npos"]
T0 = time.time()


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


def bucket(s):
    return pl.Series(M.fold_of(s, n_folds=1000, seed=7919))


def ce_feats(fr, sc):
    f32 = pl.Float32
    x = fr.join(sc.select("s1", "cand", pl.col("ce").cast(f32)), on=["s1", "cand"], how="left", maintain_order="left")
    mx = pl.col("ce").max().over("s1")
    second = pl.col("ce").drop_nulls().top_k(2).min().over("s1")
    sig = 1.0 / (1.0 + (-pl.col("ce")).exp())
    x = x.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("s1").cast(f32).alias("ce_rank"),
        (mx - pl.col("ce")).cast(f32).alias("ce_gap_top"),
        pl.when(pl.col("ce") >= mx).then(pl.col("ce") - second).otherwise(pl.col("ce") - mx).cast(f32).alias("ce_gap_other"),
        (pl.col("ce") - pl.col("p1_logit")).cast(f32).alias("ce_minus_logit"),
        sig.alias("_sig"))
    return x.with_columns(
        pl.col("_sig").sum().over("s1").cast(f32).alias("ce_sig_sum"),
        (pl.col("_sig").sum().over("s1") - pl.col("_sig").fill_null(0.0)).cast(f32).alias("ce_sig_other"),
        (pl.col("ce") > 0).sum().over("s1").cast(f32).alias("ce_npos")).drop("_sig")


b1 = MD.load_bundle(str(W / "model"))
oof = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", "label", "p", "p_raw")
dec1 = json.load(open(W / "model" / "decision.json", encoding="utf-8"))["best"]
lam, eb1 = float(dec1["lam_missing"]), float(dec1.get("empty_bias", 2.0))
gt_all = bio.scan_split(W, "train")["gt"].collect()
pre = f"score_{a.scores_tag}_" if a.scores_tag else "score_"
frames, scored_all, pops = [], [], []
for c in ("India", "US"):
    pf = pl.read_parquet(CE / f"pfull_train_{c}.parquet").select("s1", "cand", "p", "p_raw", "label")
    bk = bucket(pf["s1"])
    samp = pf.filter(bk < 150).select("s1").unique()
    own = oof.join(samp, on="s1", how="semi").select(pf.columns)
    scored = pl.concat([own, pf.filter((bk >= 150) & (bk < 300))], how="vertical_relaxed")
    del pf, own
    pop = scored.select("s1").unique().sort("s1")
    fr = pl.read_parquet(OUT / f"frame_{c}.parquet")
    sc = pl.concat([pl.read_parquet(p) for k in ("oof", "hb") for p in sorted(glob.glob(str(CE / f"{pre}{k}_{c}*.parquet")))])
    fr = ce_feats(fr, sc)
    log(f"{c}: ce coverage {fr['ce'].is_not_null().mean():.4f} of {fr.height:,} rows")
    frames.append(fr.with_columns(pl.lit(c).alias("country")))
    scored_all.append(scored.select("s1", "cand", "label", "p").with_columns(pl.lit(c).alias("country")))
    pops.append(pop.with_columns(pl.lit(c).alias("country")))
frame = pl.concat(frames, how="vertical_relaxed")
scored = pl.concat(scored_all, how="vertical_relaxed")
meta = pl.concat(pops)
del frames, scored_all
fb, mb = bucket(frame["s1"]), bucket(meta["s1"])
tr_fr, hb_fr = frame.filter(fb < 150), frame.filter(fb >= 150)
meta_tr, meta_hb = meta.filter(mb < 150), meta.filter(mb >= 150)
sc_tr = scored.join(meta_tr.select("s1"), on="s1", how="semi")
sc_hb = scored.join(meta_hb.select("s1"), on="s1", how="semi")
del frame, scored


def evaluate(tag, sc_pop, meta_pop, res):
    merged = S2.merge(sc_pop.select("s1", "cand", "p"), res.select("s1", "cand", "p"))
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


for v in a.variants.split(","):
    cols = CE4 if v == "ce4" else CE8
    params = dict(S2.S2_PARAMS)
    if v.endswith("_lr05"):
        params.update(learning_rate=0.05, num_leaves=255, min_data_in_leaf=50)
    cfg = S2.Stage2Config(orig_features=list(b1["feature_columns"]) + cols, floor=1e-3, lam_missing=lam, empty_bias=eb1,
                          guard="G1", params=params)
    t = time.time()
    b2, oof2 = S2.fit(tr_fr, cfg)
    S2.save_bundle(b2, str(OUT / f"model_{v}{'_' + a.scores_tag if a.scores_tag else ''}"))
    S2.merge(sc_tr.select("s1", "cand", "label", "p", "country"), oof2.select("s1", "cand", "p")).write_parquet(
        OUT / f"oof_{v}{'_' + a.scores_tag if a.scores_tag else ''}.parquet")
    rep = {"variant": v, "scores_tag": a.scores_tag, "oof": evaluate(f"{v} OOF", sc_tr, meta_tr, oof2),
           "hb": evaluate(f"{v} HB", sc_hb, meta_hb, S2.predict(b2, hb_fr)), "lam_missing": lam,
           "fit_report": b2["report"]}
    json.dump(rep, open(OUT / f"tune_{v}{'_' + a.scores_tag if a.scores_tag else ''}.json", "w"), indent=1, default=str)
    log(f"{v}: done in {time.time() - t:.0f}s; top gain {b2['report']['top_gain'][:10]}")
log("done")
