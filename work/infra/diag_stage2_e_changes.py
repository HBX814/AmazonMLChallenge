# -*- coding: utf-8 -*-
"""diag_stage2_e_changes.py -- the same label-free change statistics that diag_stage2_d_test.py prints for test
(links/S1, empty rate, share of S1 whose list changed, links added / removed), measured on the labelled OOF and HB
populations, plus the precision of the added / removed links -> tells whether the test-time change pattern
(France / India / US) looks like the one that gains F0.5 on train. Writes /vol/diag/stage2/changes.json
"""
import json
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import lightgbm as lgb        # noqa: E402
import numpy as np            # noqa: E402
import polars as pl           # noqa: E402

import diag_stage2_lib as L   # noqa: E402
from ber import metric as M   # noqa: E402

OUT = "/vol/diag/stage2"
LAM = 0.05303060038344832


def changes(l1, l2, labels, gt, ev):
    n = ev.height
    a1 = l1.with_columns(pl.lit(1).alias("_a"))
    a2 = l2.with_columns(pl.lit(1).alias("_b"))
    added = l2.join(a1, on=["s1", "mid"], how="anti")
    removed = l1.join(a2, on=["s1", "mid"], how="anti")
    lab = labels.select("s1", pl.col("cand").alias("mid"), "label")
    ap = added.join(lab, on=["s1", "mid"], how="left")["label"].fill_null(0).mean()
    rp = removed.join(lab, on=["s1", "mid"], how="left")["label"].fill_null(0).mean()
    ch = pl.concat([added.select("s1"), removed.select("s1")]).unique()
    f1 = M.per_s1_scores(l1, gt, ev.select("s1")).select("s1", pl.col("f").alias("f1"))
    f2 = M.per_s1_scores(l2, gt, ev.select("s1")).select("s1", pl.col("f").alias("f2"))
    d = f1.join(f2, on="s1").join(ch.select(pl.col("s1").cast(pl.Utf8)), on="s1", how="semi")
    by = []
    for c, g in ev.join(ch, on="s1", how="semi").group_by("country"):
        by.append((c[0], g.height / ev.filter(pl.col("country") == c[0]).height))
    return {"links_per_s1_s1": round(l1.height / n, 4), "links_per_s1_s2": round(l2.height / n, 4),
            "empty_rate_s1": round(1 - l1["s1"].n_unique() / n, 5), "empty_rate_s2": round(1 - l2["s1"].n_unique() / n, 5),
            "links_added": added.height, "links_removed": removed.height,
            "added_per_s1": round(added.height / n, 5), "removed_per_s1": round(removed.height / n, 5),
            "added_precision": round(float(ap), 4), "removed_precision": round(float(rp), 4),
            "s1_changed_rate": round(ch.height / n, 5), "s1_changed_rate_by_country": sorted(by),
            "changed_s1_f_before": round(float(d["f1"].mean()), 4), "changed_s1_f_after": round(float(d["f2"].mean()), 4),
            "changed_s1_improved": int((d["f2"] > d["f1"]).sum()), "changed_s1_worse": int((d["f2"] < d["f1"]).sum())}


def main():
    t0 = time.time()
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    gt = pl.read_parquet(f"{OUT}/gt.parquet")
    pf = pl.scan_parquet(f"{OUT}/p_full.parquet")
    cal = json.load(open(f"{OUT}/model_s2_full_calib.json"))
    kx, ky, cols, floor = np.asarray(cal["x"]), np.asarray(cal["y"]), cal["cols"], float(cal["floor"])
    res = {}
    for grp, tag in ((0, "OOF"), (1, "HB")):
        ev = meta.filter(pl.col("grp") == grp).select("s1", "country")
        g = gt.join(ev, on="s1", how="semi")
        fr = pf.filter(pl.col("grp") == grp).select("s1", "cand", "label", "p1").collect()
        if grp == 0:
            resc = pl.read_parquet(f"{OUT}/oof_p2_full.parquet").select("s1", "cand", "p2")
        else:
            hb = pl.read_parquet(f"{OUT}/s2_hb.parquet").filter(pl.col("p1") >= floor)
            b = lgb.Booster(model_file=f"{OUT}/model_s2_full.txt")
            p2 = np.interp(b.predict(hb.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()), kx, ky)
            resc = hb.select("s1", "cand").with_columns(pl.Series("p2", p2.astype(np.float32)))
        l1 = L.select_fast(fr.select("s1", "cand", pl.col("p1").alias("p")), "soft", LAM, 2.0)
        l2 = L.select_fast(L.merge_p(fr, resc), "soft", LAM, 1.5)
        res[tag] = changes(l1, l2, fr.select("s1", "cand", "label"), g, ev)
        res[tag]["f_s1"], res[tag]["f_s2"] = L.score(l1, g, ev), L.score(l2, g, ev)
        print(tag, json.dumps(res[tag]), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/changes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
