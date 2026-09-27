# -*- coding: utf-8 -*-
"""expB_a_stage2.py -- validate ber/stage2.py against the prototype (work/infra/diag_stage2_*.py) and train it.

1. build_frame per train country (population = OOF S1 (grp 0) + HB S1 (grp 1), competition = every train S1) from
   /vol/diag/stage2/p_full.parquet (stage-1 p of ALL train pairs), /vol/work_v3/feats, /vol/work_v3/norm
2. feature equality vs the prototype frames s2_oof.parquet / s2_hb.parquet (rows p1 >= 1e-3)
3. prototype model (model_s2_full.txt) through stage2.predict on the new HB frame -> must give HB 0.98452 / 0.98397
   (G0 / G1, soft exclusivity, empty_bias 1.5); prototype OOF p2 through stage2.merge/apply_guard -> 0.98411 / 0.98361
4. stage2.fit on the OOF frame (frozen folds) -> OOF grid + final model -> HB
Outputs /vol/exp/expB/: stage2_model/ (model_s2.txt + stage2.json), scored_oof.parquet, scored_hb.parquet
(s1, cand codes, label, p1, p_g0, p_g1), a_results.json
"""
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

from ber import decide as D   # noqa: E402
from ber import stage2 as S2  # noqa: E402
import diag_stage2_lib as L   # noqa: E402

W3, DG, OUT = "/vol/work_v3", "/vol/diag/stage2", "/vol/exp/expB"
LAM = 0.05303060038344832
T0 = time.time()
R = {}


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def dump():
    json.dump(R, open(f"{OUT}/a_results.json", "w"), indent=1, default=str)


def dec(df):
    return df.with_columns([pl.format("S{}-{}", pl.col(c) // 10_000_000_000, pl.col(c) % 10_000_000_000).alias(c)
                            for c in ("s1", "cand")])


def enc(df):
    return df.with_columns([(pl.col(c).str.slice(1, 1).cast(pl.Int64) * 10_000_000_000
                             + pl.col(c).str.slice(3).cast(pl.Int64)).alias(c) for c in ("s1", "cand")])


def decide(scored, eb, fast=True):
    if fast:
        return L.select_fast(scored, "soft", LAM, eb)
    return D.select_links(scored.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM,
                          empty_bias=eb)


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle1 = json.load(open(f"{W3}/model/bundle.json", encoding="utf-8"))
    orig = bundle1["feature_columns"]
    cfg = S2.Stage2Config(orig_features=orig, floor=1e-3, lam_missing=LAM, empty_bias=2.0, guard="G1")
    meta = pl.read_parquet(f"{DG}/meta.parquet")            # s1 (code), entity_id, country, fold, grp
    gt = pl.read_parquet(f"{DG}/gt.parquet")                # codes
    pf = pl.read_parquet(f"{DG}/p_full.parquet")            # s1, cand, grp, label, p1, p1raw (codes)
    log(f"p_full {pf.height:,}")
    frames = []
    for c in sorted(meta["country"].unique().to_list()):
        s1c = meta.filter(pl.col("country") == c).select("s1")
        sc = (pf.join(s1c, on="s1", how="semi")
                .select("s1", "cand", pl.col("p1").alias("p"), pl.col("p1raw").alias("p_raw"), "label"))
        pop = meta.filter((pl.col("country") == c) & (pl.col("grp") <= 1)).select("s1")
        t = time.time()
        fr = S2.build_frame(sc, f"{W3}/feats/train_{c}.parquet", f"{W3}/norm/train_{c}_pool.parquet", cfg,
                            population=pop)
        R[f"build_frame_{c}_s"] = round(time.time() - t, 1)
        R[f"build_frame_{c}_rows"] = fr.height
        log(f"{c}: frame {fr.height:,} x {fr.width} in {R[f'build_frame_{c}_s']}s")
        frames.append(fr.join(meta.select("s1", "grp", "country"), on="s1", how="left"))
        del sc
    del pf
    fr = pl.concat(frames)
    del frames
    R["peak_rss_after_frames_gb"] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    dump()
    # ---------------------------------------------------------------- 2. feature equality vs the prototype
    cols = S2.feature_columns(cfg)
    for grp, tag in ((0, "oof"), (1, "hb")):
        proto = pl.read_parquet(f"{DG}/s2_{tag}.parquet").filter(pl.col("p1") >= 1e-3)
        mine = fr.filter(pl.col("grp") == grp)
        j = mine.join(proto.select(["s1", "cand"] + cols), on=["s1", "cand"], how="inner", suffix="_pr")
        diff = {}
        for col in cols:
            a, b = j[col].cast(pl.Float64).to_numpy(), j[f"{col}_pr"].cast(pl.Float64).to_numpy()
            d = np.abs(a - b)
            n_bad = int((d > 1e-5).sum())
            if n_bad:
                diff[col] = {"rows_diff": n_bad, "max_abs": float(d.max())}
        R[f"equality_{tag}"] = {"mine": mine.height, "proto": proto.height, "matched": j.height, "diff_cols": diff}
        log(f"equality {tag}: mine {mine.height:,} proto {proto.height:,} matched {j.height:,}; differing {diff}")
        del proto, j
    dump()
    # ---------------------------------------------------------------- populations / edges
    pf = pl.scan_parquet(f"{DG}/p_full.parquet")
    ev = {g: meta.filter(pl.col("grp") == g).select("s1", "country") for g in (0, 1)}
    gts = {g: gt.join(ev[g], on="s1", how="semi") for g in (0, 1)}
    edges = {g: pf.filter(pl.col("grp") == g).select("s1", "cand", "label", "p1").collect() for g in (0, 1)}

    def evaluate(g, resc, tag, ebs=(1.5,), fast=True):
        """resc: s1, cand, p (re-scored rows) -> merged into the population edges -> decisions."""
        sc = S2.merge(edges[g].select("s1", "cand", pl.col("p1").alias("p")), resc.select("s1", "cand", "p"))
        out = {}
        for eb in ebs:
            lk = decide(sc, eb, fast)
            out[f"eb{eb}"] = {**L.score(lk, gts[g], ev[g]), "links_per_s1": round(lk.height / ev[g].height, 4)}
        log(tag, out)
        return out, sc
    # stage-1 reference
    for g, tag in ((0, "OOF"), (1, "HB")):
        sc1 = edges[g].select("s1", "cand", pl.col("p1").alias("p"))
        R[f"stage1_{tag}"] = {**L.score(decide(sc1, 2.0), gts[g], ev[g])}
        R[f"stage1_{tag}_select_links"] = {**L.score(decide(sc1, 2.0, fast=False), gts[g], ev[g])}
        log(f"stage1 {tag}", R[f"stage1_{tag}"], R[f"stage1_{tag}_select_links"])
    # ---------------------------------------------------------------- 3. prototype model through stage2.predict
    cal = json.load(open(f"{DG}/model_s2_full_calib.json"))
    proto_b = {"booster": lgb.Booster(model_file=f"{DG}/model_s2_full.txt"), "feature_columns": cal["cols"],
               "calibrator": [cal["x"], cal["y"]], "guard": "G1"}
    hb = fr.filter(pl.col("grp") == 1)
    for gd in ("G0", "G1"):
        R[f"proto_model_HB_{gd}"], _ = evaluate(1, S2.predict(proto_b, hb, guard=gd), f"proto model HB {gd}",
                                                ebs=(1.5, 2.0))
    oofp2 = pl.read_parquet(f"{DG}/oof_p2_full.parquet").select("s1", "cand", "p2")
    oo = fr.filter(pl.col("grp") == 0).select(["s1", "cand", "p1"] + S2.GUARD_COLS).join(oofp2, on=["s1", "cand"], how="left")
    R["proto_oof_p2_missing"] = int(oo["p2"].null_count())
    for gd in ("G0", "G1"):
        R[f"proto_oofp2_OOF_{gd}"], _ = evaluate(0, S2.apply_guard(oo.drop_nulls("p2"), gd), f"proto OOF p2 {gd}",
                                                 ebs=(1.5, 2.0))
    dump()
    # ---------------------------------------------------------------- 4. fit with ber.stage2
    oof_fr = dec(fr.filter(pl.col("grp") == 0).drop("grp", "country"))
    t = time.time()
    bundle, oof = S2.fit(oof_fr, cfg)
    R["fit_s"] = round(time.time() - t, 1)
    R["fit_report"] = bundle["report"]
    S2.save_bundle(bundle, f"{OUT}/stage2_model")
    log(f"fit {R['fit_s']}s; report {bundle['report']['auc_ll_stage2']} vs {bundle['report']['auc_ll_stage1_same_rows']}; "
        f"iters {bundle['best_iters']}")
    oof_c = enc(oof)
    res_oof = {}
    for gd in ("G0", "G1"):
        rs = S2.apply_guard(oof_c.select("s1", "cand", "p1", "p2").join(
            fr.filter(pl.col("grp") == 0).select(["s1", "cand"] + S2.GUARD_COLS), on=["s1", "cand"]), gd)
        res_oof[gd], _ = evaluate(0, rs, f"my fit OOF {gd}", ebs=(1.0, 1.5, 2.0))
        R[f"mine_OOF_{gd}"] = res_oof[gd]
    b2 = S2.load_bundle(f"{OUT}/stage2_model")
    hb_res = {}
    for gd in ("G0", "G1"):
        hb_res[gd] = S2.predict(b2, hb, guard=gd)
        R[f"mine_HB_{gd}"], _ = evaluate(1, hb_res[gd], f"my fit HB {gd}", ebs=(1.0, 1.5, 2.0))
    R["mine_HB_G1_select_links"], _ = evaluate(1, hb_res["G1"], "my fit HB G1 (select_links)", ebs=(1.5,), fast=False)
    dump()
    # ---------------------------------------------------------------- scored frames for the adaptation job
    for g, tag, rsc in ((0, "oof", oof_c.select("s1", "cand", "p1", "p2").join(
            fr.filter(pl.col("grp") == 0).select(["s1", "cand"] + S2.GUARD_COLS), on=["s1", "cand"])),
                        (1, "hb", hb_res["G1"].select("s1", "cand", "p1", "p2").join(
            hb.select(["s1", "cand"] + S2.GUARD_COLS), on=["s1", "cand"]))):
        base = edges[g].select("s1", "cand", "label", pl.col("p1").alias("p"))
        g0 = S2.merge(base, S2.apply_guard(rsc, "G0").select("s1", "cand", "p")).rename({"p": "p_g0"})
        g1 = S2.merge(base, S2.apply_guard(rsc, "G1").select("s1", "cand", "p"))
        out = (base.rename({"p": "p1"}).with_columns(g0["p_g0"], g1["p"].alias("p_g1"))
                   .join(meta.select("s1", "country"), on="s1", how="left"))
        out.write_parquet(f"{OUT}/scored_{tag}.parquet")
        log(f"scored_{tag}: {out.height:,} rows")
    R["seconds"] = round(time.time() - T0, 1)
    R["peak_rss_gb"] = round(psutil.Process().memory_info().rss / 2 ** 30, 2)
    dump()
    log("done")


if __name__ == "__main__":
    main()
