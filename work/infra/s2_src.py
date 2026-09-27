# -*- coding: utf-8 -*-
"""s2_src.py -- stage 2 (CE v1 + v2 features, = the v5 model) with and without per-source copy-budget features.
Same frames / populations (OOF buckets < 150, HB 150-299) / frozen folds / decision grid as ce_s2_v2.py.
Why: copies per S1 are capped and peaked per source (S2 0..5, S3 0..6; P(one more S2 copy | m found) = 0.87, 0.59,
0.42, 0.30, 0.17, 0 for m = 0..5), and the largest model loss is empty-address records whose name core is shared by
>= 2 S1 (9,550 missed true links). No stage-1/2 feature knows how many same-source copies each claimant already has.
Features per stage-2 row (s1, cand; src = S2 | S3 of cand), from stage-1 p of ALL train pairs (pfull, every S1):
  sb_own09 / sb_own05  other candidates of this S1 from the same source with p1 >= 0.9 / >= 0.5
  sb_oth09             candidates of this S1 from the other source with p1 >= 0.9
  sb_ownsum            sum p1 (>= 0.05) of the other same-source candidates of this S1
  sb_riv_p, sb_riv09   strongest OTHER S1 claiming this cand (p1 >= 0.05): its p1, its same-source >= 0.9 count
  sb_h_own, sb_h_riv   hazard P(n_src >= m+1 | n_src >= m) at those counts (train GT of S1 buckets >= 300 only)
  sb_h_share           sb_h_own / (sb_h_own + sb_h_riv)
    python infra/s2_src.py
"""
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

W, CE1, CE2 = Path("/vol/work_v4"), Path("/vol/exp/ce"), Path("/vol/exp/ce2")
OUT = CE2 / "s2src"
OUT.mkdir(parents=True, exist_ok=True)
BIASES = [1.0, 1.5, 2.0]
SB = ["sb_own09", "sb_own05", "sb_oth09", "sb_ownsum", "sb_riv_p", "sb_riv09", "sb_h_own", "sb_h_riv", "sb_h_share"]
T0 = time.time()


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


def bucket(s):
    return pl.Series(M.fold_of(s, n_folds=1000, seed=7919))


def ienc(df, col, out):
    """S2-166376419 -> 2*1e10 + 166376419 (same code as stage2._enc); integer columns pass through."""
    c = pl.col(col)
    e = c.cast(pl.Int64) if df.schema[col].is_integer() else c.str.slice(1, 1).cast(pl.Int64) * 10_000_000_000 + c.str.slice(3).cast(pl.Int64)
    return e.alias(out)


def scores(name, sets):
    if name == "ce":
        return pl.concat([pl.read_parquet(p).select("s1", "cand", pl.col("ce").alias("logit"))
                          for k in sets for p in sorted(glob.glob(str(CE1 / f"score_{k}*.parquet")))])
    kg = pl.read_parquet(CE2 / "kg_scores.parquet")
    return kg.filter(pl.col("set").is_in(sets)).select("s1", "cand", pl.col("ce2").alias("logit"))


# ------------------------------------------------------------------------------------------------ hazard (train GT)
gt_all = bio.scan_split(W, "train")["gt"].collect()
s1_all = pl.concat([pl.read_parquet(W / "norm" / f"train_{c}_s1.parquet", columns=["entity_id"]) for c in ("India", "US")])["entity_id"]
fit_s1 = s1_all.filter(bucket(s1_all) >= 300).to_frame("s1")
cnt = (gt_all.join(fit_s1, on="s1", how="semi").with_columns(pl.col("mid").str.slice(1, 1).cast(pl.Int64).alias("src"))
             .group_by("s1", "src").len())
haz = []
for src in (2, 3):
    n = fit_s1.join(cnt.filter(pl.col("src") == src).select("s1", "len"), on="s1", how="left")["len"].fill_null(0)
    for m in range(0, 12):
        ge, ge1 = int((n >= m).sum()), int((n >= m + 1).sum())
        haz.append({"src": src, "m": m, "h": (ge1 / ge) if ge else 0.0})
HAZ = pl.DataFrame(haz, schema={"src": pl.Int64, "m": pl.Int64, "h": pl.Float32})
log("hazard S2:", [round(r["h"], 3) for r in haz if r["src"] == 2][:7], "S3:", [round(r["h"], 3) for r in haz if r["src"] == 3][:8])


def budget_feats(pf):
    """pf: s1, cand, p (stage-1 p of ALL pairs of one country) -> _s1i, _ci + SB columns for every pair."""
    e = pf.select(ienc(pf, "s1", "_s1i"), ienc(pf, "cand", "_ci"), pl.col("p").cast(pl.Float32))
    e = e.filter(pl.col("p") >= 1e-3).with_columns((pl.col("_ci") // 10_000_000_000).alias("src"))
    agg = e.group_by("_s1i", "src").agg((pl.col("p") >= 0.9).sum().cast(pl.Int64).alias("n09"),
                                        (pl.col("p") >= 0.5).sum().cast(pl.Int64).alias("n05"),
                                        pl.col("p").filter(pl.col("p") >= 0.05).sum().alias("psum"))
    oth = agg.select("_s1i", (5 - pl.col("src")).alias("src"), pl.col("n09").alias("oth09"))
    top = (e.filter(pl.col("p") >= 0.05).sort(["_ci", "p", "_s1i"], descending=[False, True, False])
             .group_by("_ci", maintain_order=True).head(2)
             .with_columns(pl.int_range(pl.len()).over("_ci").alias("r")))
    t1 = top.filter(pl.col("r") == 0).select("_ci", pl.col("_s1i").alias("a1"), pl.col("p").alias("pa1"))
    t2 = top.filter(pl.col("r") == 1).select("_ci", pl.col("_s1i").alias("a2"), pl.col("p").alias("pa2"))
    x = (e.join(agg, on=["_s1i", "src"], how="left").join(oth, on=["_s1i", "src"], how="left")
          .join(t1, on="_ci", how="left").join(t2, on="_ci", how="left")
          .with_columns(
              (pl.col("n09") - (pl.col("p") >= 0.9).cast(pl.Int64)).alias("sb_own09"),
              (pl.col("n05") - (pl.col("p") >= 0.5).cast(pl.Int64)).alias("sb_own05"),
              pl.col("oth09").fill_null(0).alias("sb_oth09"),
              (pl.col("psum") - pl.when(pl.col("p") >= 0.05).then(pl.col("p")).otherwise(0.0)).alias("sb_ownsum"),
              pl.when(pl.col("a1") == pl.col("_s1i")).then(pl.col("a2")).otherwise(pl.col("a1")).alias("riv"),
              pl.when(pl.col("a1") == pl.col("_s1i")).then(pl.col("pa2")).otherwise(pl.col("pa1")).alias("sb_riv_p")))
    x = (x.join(agg.select(pl.col("_s1i").alias("riv"), "src", pl.col("n09").alias("rn09")), on=["riv", "src"], how="left")
          .with_columns(pl.when(pl.col("riv").is_null()).then(None)
                          .otherwise(pl.col("rn09").fill_null(0) - (pl.col("sb_riv_p") >= 0.9).cast(pl.Int64)).alias("sb_riv09")))
    x = (x.join(HAZ.rename({"m": "sb_own09", "h": "sb_h_own"}), on=["src", "sb_own09"], how="left")
          .join(HAZ.rename({"m": "sb_riv09", "h": "sb_h_riv"}), on=["src", "sb_riv09"], how="left")
          .with_columns((pl.col("sb_h_own") / (pl.col("sb_h_own") + pl.col("sb_h_riv"))).alias("sb_h_share")))
    return x.select("_s1i", "_ci", *[pl.col(c).cast(pl.Float32) for c in SB])


# ------------------------------------------------------------------------------------------------ frames (as ce_s2_v2)
b1 = MD.load_bundle(str(W / "model"))
oof = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", "label", "p", "p_raw")
dec1 = json.load(open(W / "model" / "decision.json", encoding="utf-8"))["best"]
lam, eb1 = float(dec1["lam_missing"]), float(dec1.get("empty_bias", 2.0))
frames, scored_all, pops = [], [], []
for c in ("India", "US"):
    pf = pl.read_parquet(CE1 / f"pfull_train_{c}.parquet").select("s1", "cand", "p", "p_raw", "label")
    sbf = budget_feats(pf)
    bk = bucket(pf["s1"])
    own = oof.join(pf.filter(bk < 150).select("s1").unique(), on="s1", how="semi").select(pf.columns)
    scored = pl.concat([own, pf.filter((bk >= 150) & (bk < 300))], how="vertical_relaxed")
    del pf, own
    fr = pl.read_parquet(CE1 / "s2" / f"frame_{c}.parquet")
    for name in ("ce", "ce2"):
        fr = CX.add_features(fr, scores(name, [f"oof_{c}", f"hb_{c}"]), name)
    fr = (fr.with_columns(ienc(fr, "s1", "_s1i"), ienc(fr, "cand", "_ci"))
            .join(sbf, on=["_s1i", "_ci"], how="left").drop("_s1i", "_ci"))
    del sbf
    log(f"{c}: frame {fr.height:,} rows; sb coverage {fr['sb_own09'].is_not_null().mean():.4f}; "
        f"rival share {fr['sb_riv_p'].is_not_null().mean():.4f}; mean own09 {fr['sb_own09'].mean():.3f}")
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
ce_cols = CX.feature_names([{"name": "ce"}, {"name": "ce2"}])
for v, extra in (("v1v2", []), ("v1v2_sb", SB)):
    cfg = S2.Stage2Config(orig_features=list(b1["feature_columns"]) + ce_cols + extra, floor=1e-3, lam_missing=lam,
                          empty_bias=eb1, guard="G1")
    t = time.time()
    b2, oof2 = S2.fit(tr_fr, cfg)
    S2.save_bundle(b2, str(OUT / f"model_{v}"))
    S2.merge(sc_tr.select("s1", "cand", "label", "p", "country"), oof2.select("s1", "cand", "p")).write_parquet(OUT / f"oof_{v}.parquet")
    r = {"variant": v, "oof": evaluate(f"{v} OOF", sc_tr, meta_tr, oof2),
         "hb": evaluate(f"{v} HB", sc_hb, meta_hb, S2.predict(b2, hb_fr)), "lam_missing": lam, "fit_report": b2["report"]}
    r["oof_best"] = max(x["all"] for x in r["oof"].values())
    r["hb_best"] = max(x["all"] for x in r["hb"].values())
    r["eb_best_oof"] = max(r["oof"].items(), key=lambda kv: kv[1]["all"])[0]
    res[v] = r
    json.dump(r, open(OUT / f"tune_{v}.json", "w"), indent=1, default=str)
    log(f"{v}: OOF {r['oof_best']:.5f} HB {r['hb_best']:.5f} (eb {r['eb_best_oof']}); {time.time() - t:.0f}s; "
        f"top gain {b2['report']['top_gain'][:12]}")
d_oof = res["v1v2_sb"]["oof_best"] - res["v1v2"]["oof_best"]
d_hb = res["v1v2_sb"]["hb_best"] - res["v1v2"]["hb_best"]
log(f"source-budget delta: OOF {d_oof:+.5f}  HB {d_hb:+.5f}")
json.dump({"delta_oof": d_oof, "delta_hb": d_hb, "all": {v: [r["oof_best"], r["hb_best"]] for v, r in res.items()}},
          open(OUT / "summary.json", "w"), indent=1)
log("done")
