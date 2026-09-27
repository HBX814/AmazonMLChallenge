# -*- coding: utf-8 -*-
"""diag_stage2_h_profile.py -- are the links stage 2 ADDS / REMOVES on test the same kind as on train?
Profile of a changed link = house-number relation (exact / base / mismatch / missing) x empty address x stage-1 p band.
Precision per profile is measured on the labelled OOF; the test profile mix then gives a reweighted estimate of the
precision of the links stage 2 adds / removes on test (break-even for adding a link at F~0.98 is q* = F/1.25 ~ 0.78).
Writes /vol/diag/stage2/profile.json
"""
import json
import sys

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import polars as pl           # noqa: E402

import diag_stage2_lib as L   # noqa: E402

OUT, W3 = "/vol/diag/stage2", "/vol/work_v3"
LAM = 0.05303060038344832
FEATS = ["hn_both", "hn_exact", "hn_base", "a_empty_any"]


def profile(df):
    hn = (pl.when(pl.col("hn_exact") > 0).then(pl.lit("exact")).when(pl.col("hn_base") > 0).then(pl.lit("base"))
            .when(pl.col("hn_both") > 0).then(pl.lit("mismatch")).otherwise(pl.lit("missing")))
    band = (pl.when(pl.col("p1") < 0.1).then(pl.lit("a<.1")).when(pl.col("p1") < 0.3).then(pl.lit("b.1-.3"))
              .when(pl.col("p1") < 0.5).then(pl.lit("c.3-.5")).when(pl.col("p1") < 0.7).then(pl.lit("d.5-.7"))
              .otherwise(pl.lit("e>=.7")))
    return df.with_columns(pl.concat_str([hn, pl.col("a_empty_any").cast(pl.Int8).cast(pl.Utf8), band], separator="|")
                           .alias("prof"))


def diff(l1, l2):
    k = ["s1", "mid"]
    return l2.join(l1, on=k, how="anti"), l1.join(l2, on=k, how="anti")


def main():
    res = {}
    # ------------------------------------------------ OOF (labels)
    pf = pl.scan_parquet(f"{OUT}/p_full.parquet")
    alle = pf.filter(pl.col("grp") == 0).select("s1", "cand", "p1").collect()
    s2 = pl.read_parquet(f"{OUT}/s2_oof.parquet", columns=["s1", "cand", "label", "p1"] + FEATS)
    resc = pl.read_parquet(f"{OUT}/oof_p2_full.parquet").select("s1", "cand", "p2")
    l1 = L.select_fast(alle.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, 2.0)
    l2 = L.select_fast(L.merge_p(alle, resc), "soft", LAM, 1.5)
    add, rem = diff(l1, l2)
    tab = {}
    for name, d in (("added", add), ("removed", rem)):
        x = profile(d.join(s2.rename({"cand": "mid"}), on=["s1", "mid"], how="left"))
        t = x.group_by("prof").agg(pl.len().alias("n"), pl.col("label").mean().alias("prec")).sort("n", descending=True)
        tab[name] = t
        res[f"OOF_{name}"] = {"n": d.height, "precision": float(x["label"].mean()), "by_profile": t.rows()}
        print(f"OOF {name}: n {d.height:,} precision {float(x['label'].mean()):.4f}")
        print(t.head(15))
    # ------------------------------------------------ test (no labels): reweight OOF precision by the test profile mix
    import glob
    import os
    for path in sorted(glob.glob(f"{OUT}/test_*_s2_links.parquet")):
        c = os.path.basename(path)[len("test_"):-len("_s2_links.parquet")]
        t2 = pl.read_parquet(path)
        t1 = pl.read_parquet(f"{W3}/pred/test_{c}_links.parquet")
        a, r = diff(t1, t2)
        ch = pl.concat([a, r]).rename({"mid": "cand"})
        f = (pl.scan_parquet(f"{W3}/feats/test_{c}.parquet").select(["s1", "cand"] + FEATS)
               .join(ch.lazy(), on=["s1", "cand"], how="semi").collect())
        p = pl.scan_parquet(f"{W3}/pred/test_{c}_scored.parquet").join(ch.lazy(), on=["s1", "cand"], how="semi") \
              .select("s1", "cand", pl.col("p").alias("p1")).collect()
        f = f.join(p, on=["s1", "cand"], how="left").rename({"cand": "mid"})
        out = {}
        for name, d in (("added", a), ("removed", r)):
            x = profile(d.join(f, on=["s1", "mid"], how="left"))
            m = x.group_by("prof").len().join(tab[name].select("prof", "prec"), on="prof", how="left")
            cov = float(m.filter(pl.col("prec").is_not_null())["len"].sum() / max(m["len"].sum(), 1))
            est = float((m["len"] * m["prec"].fill_null(0.0)).sum() / max(m.filter(pl.col("prec").is_not_null())["len"].sum(), 1))
            share_mm = float((x["prof"].str.starts_with("mismatch")).mean()) if x.height else 0.0
            out[name] = {"n": d.height, "reweighted_precision": round(est, 4), "profile_coverage": round(cov, 4),
                         "share_hn_mismatch": round(share_mm, 4), "top_profiles": m.sort("len", descending=True).head(8).rows()}
        oof_mm = {k: float(profile(d.join(s2.rename({"cand": "mid"}), on=["s1", "mid"], how="left"))["prof"]
                           .str.starts_with("mismatch").mean()) for k, d in (("added", add), ("removed", rem))}
        out["oof_share_hn_mismatch"] = oof_mm
        res[c] = out
        print(c, json.dumps(out, default=str))
    json.dump(res, open(f"{OUT}/profile.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
