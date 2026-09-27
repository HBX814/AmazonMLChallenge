# -*- coding: utf-8 -*-
"""diag_stage2_d_test.py -- apply the OOF-trained stage-2 model (model_s2_full.txt) to the TEST split exactly as the
pipeline would (per country, competition over the whole country frame), measure runtime / memory at test scale and
compare the label-free behaviour of stage 1 vs stage 2 per country (France included).

Inputs : /vol/work_v3/{feats,pred,norm,model}, /vol/diag/stage2/model_s2_full.txt (+ _calib.json)
Outputs: /vol/diag/stage2/test_<country>_s2_links.parquet (s1, mid as original id strings), test_s2_stats.json
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import lightgbm as lgb        # noqa: E402
import numpy as np            # noqa: E402
import polars as pl           # noqa: E402
import psutil                 # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import diag_stage2_lib as L   # noqa: E402

W3 = "/vol/work_v3"
OUT = "/vol/diag/stage2"
LAM = 0.05303060038344832
EB1, EB2 = 2.0, 1.5              # stage-1 pipeline decision / stage-2 OOF-best decision (both soft exclusivity)
T0 = time.time()


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def list_stats(links, n_s1):
    per = links.group_by("s1").len()
    return {"links": links.height, "links_per_s1": round(links.height / n_s1, 4),
            "empty_rate": round(1 - per.height / n_s1, 5)}


def main():
    cal = json.load(open(f"{OUT}/model_s2_full_calib.json"))
    kx, ky, cols2, floor = np.asarray(cal["x"]), np.asarray(cal["y"]), cal["cols"], float(cal["floor"])
    s2 = lgb.Booster(model_file=f"{OUT}/model_s2_full.txt")
    bundle = json.load(open(f"{W3}/model/bundle.json", encoding="utf-8"))
    cx, cy = (np.asarray(v, dtype=np.float64) for v in bundle["calibrator"])
    cols1 = bundle["feature_columns"]
    s1m = lgb.Booster(model_file=f"{W3}/model/model_final_0.txt")
    cs = sorted(os.path.basename(p)[len("test_"):-len(".parquet")] for p in glob.glob(f"{W3}/feats/test_*.parquet"))
    log("test countries", cs)
    stats = {}
    for c in cs:
        t0 = time.time()
        T = {}
        sc = pl.read_parquet(f"{W3}/pred/test_{c}_scored.parquet")          # same row order as the feats file
        sc = sc.with_columns(L.id_code("s1").alias("s1c"), L.id_code("cand").alias("candc"))
        pf = pq.ParquetFile(f"{W3}/feats/test_{c}.parquet")
        assert pf.metadata.num_rows == sc.height
        fparts, raws, off, maxdiff = [], [], 0, 0.0
        for rb in pf.iter_batches(batch_size=2_000_000, columns=["s1", "cand"] + cols1):
            b = pl.from_arrow(rb)
            s = sc.slice(off, b.height)
            assert (b["s1"] == s["s1"]).all() and (b["cand"] == s["cand"]).all(), "row order differs"
            keep = (s["p"] >= floor).to_numpy()
            bk = b.filter(pl.Series(keep))
            raw = s1m.predict(bk.select([pl.col(x).cast(pl.Float32) for x in cols1]).to_numpy())
            maxdiff = max(maxdiff, float(np.abs(np.interp(raw, cx, cy) - s["p"].filter(pl.Series(keep)).to_numpy()).max()
                                         if len(raw) else 0.0))
            fparts.append(bk.with_columns(L.id_code("s1").alias("s1"), L.id_code("cand").alias("cand"))
                          .select(["s1", "cand"] + [x for x in cols1 if x in cols2]))
            raws.append(pl.DataFrame({"s1": s["s1c"].filter(pl.Series(keep)), "cand": s["candc"].filter(pl.Series(keep)),
                                      "p1raw": raw.astype(np.float32)}))
            off += b.height
        feats = pl.concat(fparts)
        del fparts
        T["stream_and_raw_s"] = round(time.time() - t0, 1)
        e = (sc.select(pl.col("s1c").alias("s1"), pl.col("candc").alias("cand"), pl.col("p").alias("p1"))
               .join(pl.concat(raws), on=["s1", "cand"], how="left")
               .with_columns(pl.coalesce("p1raw", "p1").alias("p1raw")))
        log(f"{c}: {e.height:,} edges, {feats.height:,} rescored rows; recomputed-p max diff {maxdiff:.2e}")
        t = time.time()
        e = L.decision_feats(L.within_s1_feats(e, "p1", "p1raw"), LAM, EB1, "p1")
        T["within_decision_s"] = round(time.time() - t, 1)
        Fr = e.filter(pl.col("p1") >= floor)
        t = time.time()
        Fr = L.competition_feats(Fr, agg=L.competition_aggregates(e.select("s1", "cand", "p1"), Fr))
        T["competition_s"] = round(time.time() - t, 1)
        pool = pl.read_parquet(f"{W3}/norm/test_{c}_pool.parquet", columns=["entity_id", "name_core", "addr_norm", "name_key"])
        names = (pool.select(L.id_code("entity_id").alias("cand"), pl.col("name_core").fill_null("").alias("nc"),
                             pl.col("addr_norm").fill_null("").alias("ad"), pl.col("name_key").fill_null("").alias("nk"))
                     .join(Fr.select("cand").unique(), on="cand", how="semi"))
        t = time.time()
        Fr = L.sibling_feats(Fr, names, "p1")
        T["sibling_s"] = round(time.time() - t, 1)
        Fr = Fr.join(feats, on=["s1", "cand"], how="left")
        miss = int(Fr[cols2[0]].null_count())
        t = time.time()
        raw2 = s2.predict(Fr.select([pl.col(x).cast(pl.Float32) for x in cols2]).to_numpy())
        T["s2_predict_s"] = round(time.time() - t, 1)
        p2 = np.interp(raw2, kx, ky).astype(np.float32)
        resc = Fr.select("s1", "cand", "p1").with_columns(pl.Series("p2", p2))
        merged = L.merge_p(e, resc)
        t = time.time()
        l1 = L.select_fast(e.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, EB1)
        l2 = L.select_fast(merged, "soft", LAM, EB2)
        T["decide_s"] = round(time.time() - t, 1)
        n_s1 = pl.scan_parquet(f"{W3}/norm/test_{c}_s1.parquet").select(pl.len()).collect().item()
        v3 = pl.read_parquet(f"{W3}/pred/test_{c}_links.parquet")
        a1 = l1.with_columns(pl.lit(1).alias("_a"))
        both = l2.join(a1, on=["s1", "mid"], how="left")
        added = int(both["_a"].null_count())
        removed = l1.height - (l2.height - added)
        ch1 = l1.group_by("s1").agg(pl.col("mid").sort().alias("m1"))
        ch2 = l2.group_by("s1").agg(pl.col("mid").sort().alias("m2"))
        chg = ch1.join(ch2, on="s1", how="full", coalesce=True)
        n_changed = int(chg.filter(pl.col("m1").is_null() | pl.col("m2").is_null() | (pl.col("m1") != pl.col("m2"))).height)
        # decisions near the boundary: how p moved on the rescored rows
        mv = resc.with_columns((pl.col("p2") - pl.col("p1")).alias("d"))
        st = {"n_s1": n_s1, "edges": e.height, "rescored_rows": Fr.height, "rescored_share": round(Fr.height / e.height, 4),
              "missing_feat_rows": miss, "recomputed_p_maxdiff": maxdiff,
              "v3_submitted": list_stats(v3, n_s1), "stage1_rebuilt": list_stats(l1, n_s1), "stage2": list_stats(l2, n_s1),
              "links_added": added, "links_removed": removed, "s1_changed": n_changed,
              "s1_changed_rate": round(n_changed / n_s1, 5),
              "p_mid_band_share_s1": round(float(((resc["p1"] > 0.1) & (resc["p1"] < 0.9)).mean()), 5),
              "p_mid_band_share_s2": round(float(((resc["p2"] > 0.1) & (resc["p2"] < 0.9)).mean()), 5),
              "mean_abs_dp": round(float(mv["d"].abs().mean()), 5),
              "timing_s": T, "total_s": round(time.time() - t0, 1),
              "rss_gb": round(psutil.Process().memory_info().rss / 2 ** 30, 2)}
        stats[c] = st
        log(f"{c}: {json.dumps(st)}")
        # original id strings for the links
        s1ids = pl.read_parquet(f"{W3}/norm/test_{c}_s1.parquet", columns=["entity_id"]).select(
            L.id_code("entity_id").alias("s1"), pl.col("entity_id").alias("s1_id"))
        pids = pool.select(L.id_code("entity_id").alias("mid"), pl.col("entity_id").alias("mid_id"))
        out = l2.join(s1ids, on="s1", how="left").join(pids, on="mid", how="left")
        assert out["s1_id"].null_count() == 0 and out["mid_id"].null_count() == 0
        out.select(pl.col("s1_id").alias("s1"), pl.col("mid_id").alias("mid")).write_parquet(f"{OUT}/test_{c}_s2_links.parquet")
        resc.write_parquet(f"{OUT}/test_{c}_s2_rescored.parquet")
        json.dump(stats, open(f"{OUT}/test_s2_stats.json", "w"), indent=1)
        del e, Fr, feats, merged, sc
    log("done")


if __name__ == "__main__":
    main()
