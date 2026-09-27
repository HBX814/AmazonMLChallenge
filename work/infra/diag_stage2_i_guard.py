# -*- coding: utf-8 -*-
"""diag_stage2_i_guard.py -- guard stage 2 against the test-US hard-negative shift.
On test US 69% of the links stage 2 adds have a house-number MISMATCH (numbers on both sides, none equal, no common
base) vs 31% on OOF, and diag_struct measured ~3.4x more hard-negative house-number pairs per S1 on test US than on
OOF -> the OOF precision of those additions (0.90) is unlikely to hold there. Guards (applied to rescored rows):
    G0  p = p2                                   (plain stage 2)
    G1  mismatch rows: p = min(p1, p2)           (stage 2 may only LOWER a house-number-mismatch pair)
    G2  mismatch rows: p = p1                    (stage 2 does not touch them)
Evaluated on OOF (stage-2 OOF p) and HB (final stage-2 model); on test: label-free change counts + guarded links.
Writes /vol/diag/stage2/guard.json and test_<c>_s2g{1,2}_links.parquet
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import lightgbm as lgb        # noqa: E402
import numpy as np            # noqa: E402
import polars as pl           # noqa: E402

import diag_stage2_lib as L   # noqa: E402

OUT, W3 = "/vol/diag/stage2", "/vol/work_v3"
LAM = 0.05303060038344832
MM = (pl.col("hn_both") > 0) & (pl.col("hn_exact") == 0) & (pl.col("hn_base") == 0)


def guarded(resc, g):
    """resc: s1, cand, p1, p2, mm -> s1, cand, p2 (guarded)."""
    if g == "G0":
        e = pl.col("p2")
    elif g == "G1":
        e = pl.when(pl.col("mm")).then(pl.min_horizontal("p1", "p2")).otherwise(pl.col("p2"))
    else:
        e = pl.when(pl.col("mm")).then(pl.col("p1")).otherwise(pl.col("p2"))
    return resc.select("s1", "cand", e.alias("p2"))


def changes(l1, l2, n, mmf):
    k = ["s1", "mid"]
    a, r = l2.join(l1, on=k, how="anti"), l1.join(l2, on=k, how="anti")
    a_mm = a.join(mmf.rename({"cand": "mid"}), on=k, how="left")["mm"].fill_null(False).sum()
    return {"links_per_s1": round(l2.height / n, 4), "added_per_s1": round(a.height / n, 5),
            "removed_per_s1": round(r.height / n, 5), "added_mismatch_per_s1": round(int(a_mm) / n, 5)}


def main():
    res = {}
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    gt = pl.read_parquet(f"{OUT}/gt.parquet")
    pf = pl.scan_parquet(f"{OUT}/p_full.parquet")
    cal = json.load(open(f"{OUT}/model_s2_full_calib.json"))
    kx, ky, cols, floor = np.asarray(cal["x"]), np.asarray(cal["y"]), cal["cols"], float(cal["floor"])
    b = lgb.Booster(model_file=f"{OUT}/model_s2_full.txt")
    for grp, tag in ((0, "OOF"), (1, "HB")):
        ev = meta.filter(pl.col("grp") == grp).select("s1", "country")
        g = gt.join(ev, on="s1", how="semi")
        alle = pf.filter(pl.col("grp") == grp).select("s1", "cand", "p1").collect()
        fr = pl.read_parquet(f"{OUT}/s2_{tag.lower()}.parquet").filter(pl.col("p1") >= floor)
        if grp == 0:
            p2 = pl.read_parquet(f"{OUT}/oof_p2_full.parquet").select("s1", "cand", "p2")
            resc = fr.select("s1", "cand", "p1", MM.alias("mm")).join(p2, on=["s1", "cand"], how="left")
        else:
            p2 = np.interp(b.predict(fr.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()), kx, ky)
            resc = fr.select("s1", "cand", "p1", MM.alias("mm")).with_columns(pl.Series("p2", p2.astype(np.float32)))
        l1 = L.select_fast(alle.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, 2.0)
        R = {"stage1": L.score(l1, g, ev), "rescored_mismatch_share": round(float(resc["mm"].mean()), 4)}
        for gg in ("G0", "G1", "G2"):
            sc = L.merge_p(alle, guarded(resc, gg))
            for eb in (1.5, 2.0):
                l2 = L.select_fast(sc, "soft", LAM, eb)
                R[f"{gg}_eb{eb}"] = {**L.score(l2, g, ev), **changes(l1, l2, ev.height, resc.select("s1", "cand", "mm"))}
                print(tag, gg, eb, R[f"{gg}_eb{eb}"], flush=True)
        res[tag] = R
    for path in sorted(glob.glob(f"{OUT}/test_*_s2_rescored.parquet")):
        c = os.path.basename(path)[len("test_"):-len("_s2_rescored.parquet")]
        resc = pl.read_parquet(path)                                  # s1, cand (codes), p1, p2
        hn = (pl.scan_parquet(f"{W3}/feats/test_{c}.parquet").select("s1", "cand", "hn_both", "hn_exact", "hn_base")
                .with_columns(L.id_code("s1").alias("s1"), L.id_code("cand").alias("cand"))
                .join(resc.lazy().select("s1", "cand"), on=["s1", "cand"], how="semi").select("s1", "cand", MM.alias("mm"))
                .collect())
        resc = resc.join(hn, on=["s1", "cand"], how="left").with_columns(pl.col("mm").fill_null(False))
        alle = pl.read_parquet(f"{W3}/pred/test_{c}_scored.parquet").select(
            L.id_code("s1").alias("s1"), L.id_code("cand").alias("cand"), pl.col("p").alias("p1"))
        n = pl.scan_parquet(f"{W3}/norm/test_{c}_s1.parquet").select(pl.len()).collect().item()
        l1 = L.select_fast(alle.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, 2.0)
        R = {"rescored_mismatch_share": round(float(resc["mm"].mean()), 4)}
        s1ids = pl.read_parquet(f"{W3}/norm/test_{c}_s1.parquet", columns=["entity_id"]).select(
            L.id_code("entity_id").alias("s1"), pl.col("entity_id").alias("s1_id"))
        pids = pl.read_parquet(f"{W3}/norm/test_{c}_pool.parquet", columns=["entity_id"]).select(
            L.id_code("entity_id").alias("mid"), pl.col("entity_id").alias("mid_id"))
        for gg in ("G0", "G1", "G2"):
            l2 = L.select_fast(L.merge_p(alle, guarded(resc, gg)), "soft", LAM, 1.5)
            R[gg] = changes(l1, l2, n, resc.select("s1", "cand", "mm"))
            print(c, gg, R[gg], flush=True)
            if gg != "G0":
                out = l2.join(s1ids, on="s1", how="left").join(pids, on="mid", how="left")
                assert out["s1_id"].null_count() == 0 and out["mid_id"].null_count() == 0
                out.select(pl.col("s1_id").alias("s1"), pl.col("mid_id").alias("mid")).write_parquet(
                    f"{OUT}/test_{c}_s2{gg.lower()}_links.parquet")
        res[c] = R
    json.dump(res, open(f"{OUT}/guard.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
