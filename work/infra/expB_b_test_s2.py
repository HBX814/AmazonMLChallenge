# -*- coding: utf-8 -*-
"""expB_b_test_s2.py -- apply the ber/stage2.py model (trained by expB_a_stage2.py) to TEST per country, the way the
pipeline would: stage-1 v3 scored pairs (whole country frame = all claimants), stage-1 raw score re-predicted from the
v3 bundle for the re-scored rows, stage-2 predict, guard G1, merge.
Outputs /vol/exp/expB/test_<c>_s2.parquet (s1, cand, p1, p_g0, p_g1 as original ids) + b_results.json
(timing / memory, links per S1, agreement with the prototype's G1 test links /vol/diag/stage2/test_<c>_s2g1_links.parquet)
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")

import polars as pl           # noqa: E402
import psutil                 # noqa: E402

from ber import decide as D   # noqa: E402
from ber import model as MD   # noqa: E402
from ber import stage2 as S2  # noqa: E402

W3, DG, OUT = "/vol/work_v3", "/vol/diag/stage2", "/vol/exp/expB"
LAM = 0.05303060038344832
T0 = time.time()


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def main():
    A = json.load(open(f"{OUT}/a_results.json"))
    eb2 = max((1.0, 1.5, 2.0), key=lambda eb: A["mine_OOF_G1"][f"eb{eb}"]["ALL"])
    b1 = MD.load_bundle(f"{W3}/model")
    b2 = S2.load_bundle(f"{OUT}/stage2_model")
    cfg = S2.stage2_config_from_bundle(b2)
    cs = sorted(os.path.basename(p)[len("test_"):-len("_scored.parquet")] for p in glob.glob(f"{W3}/pred/test_*_scored.parquet"))
    R = {"eb_stage2": eb2, "countries": cs}
    log("countries", cs, "stage-2 empty_bias", eb2)
    for c in cs:
        t0 = time.time()
        sc = pl.read_parquet(f"{W3}/pred/test_{c}_scored.parquet").select("s1", "cand", "p")
        fr = S2.build_frame(sc, f"{W3}/feats/test_{c}.parquet", f"{W3}/norm/test_{c}_pool.parquet", cfg, stage1_bundle=b1)
        t_frame = time.time() - t0
        t = time.time()
        res = S2.predict(b2, fr, guard="G1")
        t_pred = time.time() - t
        out = (S2.merge(sc, res.select("s1", "cand", "p")).rename({"p": "p_g1"})
                 .join(res.select("s1", "cand", "p2"), on=["s1", "cand"], how="left", maintain_order="left")
                 .with_columns(sc["p"].cast(pl.Float32).alias("p1"))
                 .select("s1", "cand", "p1", pl.coalesce("p2", "p1").alias("p_g0"), "p_g1"))
        out.write_parquet(f"{OUT}/test_{c}_s2.parquet")
        t = time.time()
        lk = D.select_links(out.select("s1", "cand", pl.col("p_g1").alias("p")), "expected_f", exclusivity="soft",
                            lam_missing=LAM, empty_bias=eb2)
        t_dec = time.time() - t
        n_s1 = pl.scan_parquet(f"{W3}/norm/test_{c}_s1.parquet").select(pl.len()).collect().item()
        v3 = pl.read_parquet(f"{W3}/pred/test_{c}_links.parquet")
        proto = pl.read_parquet(f"{DG}/test_{c}_s2g1_links.parquet")
        k = ["s1", "mid"]
        R[c] = {"n_s1": n_s1, "pairs": sc.height, "rescored": fr.height,
                "links_per_s1_v3": round(v3.height / n_s1, 4), "links_per_s1_s2g1": round(lk.height / n_s1, 4),
                "links_per_s1_proto_s2g1": round(proto.height / n_s1, 4),
                "agree_with_proto": lk.join(proto, on=k, how="inner").height,
                "only_mine": lk.join(proto, on=k, how="anti").height, "only_proto": proto.join(lk, on=k, how="anti").height,
                "added_vs_v3_per_s1": round(lk.join(v3, on=k, how="anti").height / n_s1, 5),
                "removed_vs_v3_per_s1": round(v3.join(lk, on=k, how="anti").height / n_s1, 5),
                "sum_p_per_s1_s1": round(float(out["p1"].sum()) / n_s1, 4),
                "sum_p_per_s1_s2g1": round(float(out["p_g1"].sum()) / n_s1, 4),
                "t_frame_s": round(t_frame, 1), "t_predict_s": round(t_pred, 1), "t_decide_s": round(t_dec, 1),
                "t_total_s": round(time.time() - t0, 1), "rss_gb": round(psutil.Process().memory_info().rss / 2 ** 30, 2)}
        log(c, json.dumps(R[c]))
        lk.write_parquet(f"{OUT}/test_{c}_s2g1_links.parquet")
        json.dump(R, open(f"{OUT}/b_results.json", "w"), indent=1)
        del sc, fr, res, out, lk
    R["seconds"] = round(time.time() - T0, 1)
    json.dump(R, open(f"{OUT}/b_results.json", "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
