# -*- coding: utf-8 -*-
"""ce_s2_v2.py -- stage 2 with cross-encoder features from CE v1 (e5-small, /vol/exp/ce) and/or CE v2 (Kaggle
bge-reranker-v2-m3, /vol/exp/ce2/kg_scores.parquet), same frames / populations / folds / decision as ce_stage2.py.
Variants: v1 (reference), v2, v1v2. Writes /vol/exp/ce2/s2/{tune_<v>.json, model_<v>/, oof_<v>.parquet, best.json}.
best.json = the variant with the best mean of (best-eb OOF, best-eb HB) among those that beat v1 on BOTH.
    python infra/ce_s2_v2.py [--variants v1,v2,v1v2]
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import crossenc as CX      # noqa: E402
from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402
from ber import model as MD         # noqa: E402
from ber import stage2 as S2        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="v1,v2,v1v2")
ap.add_argument("--ce2-file", default="kg_scores.parquet")
ap.add_argument("--out", default="s2")
a = ap.parse_args()
W, CE1, CE2 = Path("/vol/work_v4"), Path("/vol/exp/ce"), Path("/vol/exp/ce2")
OUT = CE2 / a.out
OUT.mkdir(parents=True, exist_ok=True)
BIASES = [1.0, 1.5, 2.0]
SRC = {"v1": ["ce"], "v2": ["ce2"], "v3": ["ce3"], "v1v2": ["ce", "ce2"], "v2v3": ["ce2", "ce3"],
       "v1v2v3": ["ce", "ce2", "ce3"]}
KG = {"ce2": a.ce2_file, "ce3": "kg_scores_e5l.parquet"}
T0 = time.time()
NEED = sorted({n for v in a.variants.split(",") for n in SRC[v]})


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


def bucket(s):
    return pl.Series(M.fold_of(s, n_folds=1000, seed=7919))


def scores(name, sets):
    """s1, cand, logit of a CE for the given sets (e.g. ['oof_India', 'hb_India'])."""
    if name == "ce":
        return pl.concat([pl.read_parquet(p).select("s1", "cand", pl.col("ce").alias("logit"))
                          for k in sets for p in sorted(glob.glob(str(CE1 / f"score_{k}*.parquet")))])
    kg = pl.read_parquet(CE2 / KG[name])
    return kg.filter(pl.col("set").is_in(sets)).select("s1", "cand", pl.col("ce2").alias("logit"))


b1 = MD.load_bundle(str(W / "model"))
oof = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", "label", "p", "p_raw")
dec1 = json.load(open(W / "model" / "decision.json", encoding="utf-8"))["best"]
lam, eb1 = float(dec1["lam_missing"]), float(dec1.get("empty_bias", 2.0))
gt_all = bio.scan_split(W, "train")["gt"].collect()
frames, scored_all, pops = [], [], []
for c in ("India", "US"):
    pf = pl.read_parquet(CE1 / f"pfull_train_{c}.parquet").select("s1", "cand", "p", "p_raw", "label")
    bk = bucket(pf["s1"])
    own = oof.join(pf.filter(bk < 150).select("s1").unique(), on="s1", how="semi").select(pf.columns)
    scored = pl.concat([own, pf.filter((bk >= 150) & (bk < 300))], how="vertical_relaxed")
    del pf, own
    fr = pl.read_parquet(CE1 / "s2" / f"frame_{c}.parquet")
    for name in NEED:
        fr = CX.add_features(fr, scores(name, [f"oof_{c}", f"hb_{c}"]), name)
    log(f"{c}: coverage " + ", ".join(f"{n} {fr[n].is_not_null().mean():.4f}" for n in NEED) + f" of {fr.height:,}")
    frames.append(fr.with_columns(pl.lit(c).alias("country")))
    scored_all.append(scored.select("s1", "cand", "label", "p").with_columns(pl.lit(c).alias("country")))
    pops.append(scored.select("s1").unique().with_columns(pl.lit(c).alias("country")))
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
            row[c] = M.macro_f05(links.join(ids.to_frame(), on="s1", how="semi"),
                                 gt.join(ids.to_frame(), on="s1", how="semi"), ids)
        out[b] = row
        log(f"{tag} eb {b}: " + ", ".join(f"{k} {v:.5f}" for k, v in row.items()))
    return out


res = {}
for v in a.variants.split(","):
    cols = CX.feature_names([{"name": n} for n in SRC[v]])
    cfg = S2.Stage2Config(orig_features=list(b1["feature_columns"]) + cols, floor=1e-3, lam_missing=lam, empty_bias=eb1,
                          guard="G1")
    t = time.time()
    b2, oof2 = S2.fit(tr_fr, cfg)
    S2.save_bundle(b2, str(OUT / f"model_{v}"))
    S2.merge(sc_tr.select("s1", "cand", "label", "p", "country"), oof2.select("s1", "cand", "p")).write_parquet(OUT / f"oof_{v}.parquet")
    r = {"variant": v, "ce": SRC[v], "oof": evaluate(f"{v} OOF", sc_tr, meta_tr, oof2),
         "hb": evaluate(f"{v} HB", sc_hb, meta_hb, S2.predict(b2, hb_fr)), "lam_missing": lam, "fit_report": b2["report"]}
    r["oof_best"] = max(x["all"] for x in r["oof"].values())
    r["hb_best"] = max(x["all"] for x in r["hb"].values())
    r["eb_best_oof"] = max(r["oof"].items(), key=lambda kv: kv[1]["all"])[0]
    res[v] = r
    json.dump(r, open(OUT / f"tune_{v}.json", "w"), indent=1, default=str)
    log(f"{v}: OOF {r['oof_best']:.5f} HB {r['hb_best']:.5f} (eb {r['eb_best_oof']}); {time.time() - t:.0f}s; "
        f"top gain {b2['report']['top_gain'][:8]}")
ref = res.get("v1")
ok = {v: r for v, r in res.items() if v != "v1" and (ref is None or (r["oof_best"] > ref["oof_best"] and r["hb_best"] > ref["hb_best"]))}
best = max(ok.values(), key=lambda r: r["oof_best"] + r["hb_best"]) if ok else ref
json.dump({"variant": best["variant"], "ce": best["ce"], "eb": best["eb_best_oof"], "lam_missing": lam,
           "oof": best["oof_best"], "hb": best["hb_best"],
           "all": {v: [r["oof_best"], r["hb_best"]] for v, r in res.items()}}, open(OUT / "best.json", "w"), indent=1)
log(f"best: {best['variant']} OOF {best['oof_best']:.5f} HB {best['hb_best']:.5f}")
log("done")
