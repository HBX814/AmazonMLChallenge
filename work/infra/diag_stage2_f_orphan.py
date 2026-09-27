# -*- coding: utf-8 -*-
"""diag_stage2_f_orphan.py -- is stage 2 robust to the TEST density shift (test pools ~23% denser per S1; hypothesis:
orphan records whose own S1 is absent from the S1 file)? Emulation: remove ~19% of all train S1 from the competition
frame (22.4% of the S1 outside the evaluated population, stable hash), so their records lose their owner's claim.
Only the stage-2 competition features see the change (stage-1 p is kept -> stage-1's own base-score competition still
sees every S1; the emulation is partial for stage 1).

2x2 on OOF (5-fold, frozen folds):  stage-2 trained on FULL-frame competition vs on REDUCED-frame competition,
                                     each evaluated with FULL and REDUCED competition features
and the same with the final models on HB. Writes /vol/diag/stage2/orphan.json
"""
import json
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import numpy as np            # noqa: E402
import polars as pl           # noqa: E402

from ber import metric as M   # noqa: E402
from ber import model as MD   # noqa: E402
import diag_stage2_lib as L   # noqa: E402
from diag_stage2_b_train import PARAMS, VARIANTS  # noqa: E402

OUT = "/vol/diag/stage2"
LAM = 0.05303060038344832
FLOOR = 1e-3
DROP = 224                     # per-mille of the S1 outside the evaluated population -> ~19% of all train S1
T0 = time.time()
CC = L.c_feats()


def log(*a):
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s]", *a, flush=True)


def recompute_comp(frame, pfull, keep_s1):
    red = pfull.join(keep_s1, on="s1", how="semi").select("s1", "cand", "p1")
    agg = L.competition_aggregates(red, frame)
    return L.competition_feats(frame.drop(CC), agg=agg).sort(["s1", "cand"]), red


def cross_oof(tr_frame, pr_frames, cols, folds):
    """fold models fit on tr_frame, predict every frame in pr_frames (same rows/order) -> list of raw OOF arrays."""
    cfg = MD.ModelConfig(params=dict(PARAMS), feature_columns=list(cols), verbose=False)
    outs = [np.zeros(tr_frame.height) for _ in pr_frames]
    iters = []
    for k in range(5):
        m = folds == k
        b, it = MD._fit(tr_frame.filter(pl.Series(~m)), list(cols), cfg, seed=k)
        iters.append(int(it))
        for o, f in zip(outs, pr_frames):
            o[m] = b.predict(f.filter(pl.Series(m)).select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy(),
                             num_iteration=it)
        log(f"  fold {k} best_iter {it}")
    return outs, iters


def evaluate(all_edges, frame, raw, calib, gt, ev, ext=None):
    p2 = np.interp(raw, *calib)
    sc = L.merge_p(all_edges, frame.select("s1", "cand").with_columns(pl.Series("p2", p2.astype(np.float32))))
    l2 = L.select_fast(sc, "soft", LAM, 1.5)
    out = {"soft_eb1.5": L.score(l2, gt, ev), "links_per_s1": round(l2.height / ev.height, 4)}
    if ext is not None:
        out["soft_ext_eb1.5"] = L.score(L.select_fast(sc, "soft_ext", LAM, 1.5, osum_ext=ext), gt, ev)
    return out, l2


def main():
    meta = pl.read_parquet(f"{OUT}/meta.parquet")
    meta = meta.with_columns(pl.Series("h", M.fold_of(meta["entity_id"], 1000, 4242)))
    gt = pl.read_parquet(f"{OUT}/gt.parquet")
    pfull = pl.read_parquet(f"{OUT}/p_full.parquet", columns=["s1", "cand", "grp", "p1"])
    cols = VARIANTS["full"]
    res = {"drop_per_mille_outside_population": DROP}
    stage1 = {}
    for grp, tag in ((0, "OOF"), (1, "HB")):
        ev = meta.filter(pl.col("grp") == grp).select("s1", "country")
        keep = meta.filter((pl.col("grp") == grp) | (pl.col("h") >= DROP)).select("s1")
        res[f"{tag}_dropped_share_of_all_s1"] = round(1 - keep.height / meta.height, 4)
        g = gt.join(ev, on="s1", how="semi")
        alle = pfull.filter(pl.col("grp") == grp).select("s1", "cand", "p1")
        fr_full = pl.read_parquet(f"{OUT}/s2_{tag.lower()}.parquet").filter(pl.col("p1") >= FLOOR).sort(["s1", "cand"])
        fr_red, red = recompute_comp(fr_full, pfull, keep)
        fr_red = fr_red.select(fr_full.columns)
        assert (fr_red["s1"] == fr_full["s1"]).all() and (fr_red["cand"] == fr_full["cand"]).all()
        osum_full = L.competition_aggregates(pfull.select("s1", "cand", "p1"), fr_full).select("cand", "_osum")
        osum_red = L.competition_aggregates(red, fr_full).select("cand", "_osum")
        ext_full, ext_red = L.external_osum(osum_full, alle), L.external_osum(osum_red, alle)
        sc1 = alle.select("s1", "cand", pl.col("p1").alias("p"))
        stage1[tag] = {"soft_eb2": L.score(L.select_fast(sc1, "soft", LAM, 2.0), g, ev),
                       "soft_ext_full_eb2": L.score(L.select_fast(sc1, "soft_ext", LAM, 2.0, osum_ext=ext_full), g, ev),
                       "soft_ext_red_eb2": L.score(L.select_fast(sc1, "soft_ext", LAM, 2.0, osum_ext=ext_red), g, ev)}
        log(tag, "stage1", stage1[tag])
        log(tag, "mean c_n_other full / reduced", float(fr_full["c_n_other"].mean()), float(fr_red["c_n_other"].mean()))
        y = fr_full["label"].to_numpy()
        if grp == 0:
            folds = fr_full["fold"].to_numpy()
            (raw_ff, raw_fr), it_f = cross_oof(fr_full, [fr_full, fr_red], cols, folds)
            (raw_rf, raw_rr), it_r = cross_oof(fr_red, [fr_full, fr_red], cols, folds)
            cal_f, cal_r = L.isotonic(raw_ff, y), L.isotonic(raw_rr, y)
            R = {}
            for name, raw, cal, ext in (("trainFULL_evalFULL", raw_ff, cal_f, ext_full),
                                        ("trainFULL_evalRED", raw_fr, cal_f, ext_red),
                                        ("trainRED_evalFULL", raw_rf, cal_r, ext_full),
                                        ("trainRED_evalRED", raw_rr, cal_r, ext_red)):
                R[name], _ = evaluate(alle, fr_full, raw, cal, g, ev, ext)
                log(tag, name, R[name])
            res["OOF"] = R
            # final models for HB
            b_f = L.fit_final(fr_full, cols, PARAMS, int(round(1.1 * np.mean(it_f))))
            b_r = L.fit_final(fr_red, cols, PARAMS, int(round(1.1 * np.mean(it_r))))
        else:
            R = {}
            for name, b, cal in (("trainFULL", b_f, cal_f), ("trainRED", b_r, cal_r)):
                for ev_name, f, ext in (("evalFULL", fr_full, ext_full), ("evalRED", fr_red, ext_red)):
                    raw = b.predict(f.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy())
                    R[f"{name}_{ev_name}"], l2 = evaluate(alle, fr_full, raw, cal, g, ev, ext)
                    log(tag, f"{name}_{ev_name}", R[f"{name}_{ev_name}"])
            res["HB"] = R
            b_r.save_model(f"{OUT}/model_s2_full_dens.txt")
            json.dump({"x": cal_r[0].tolist(), "y": cal_r[1].tolist(), "cols": cols, "floor": FLOOR},
                      open(f"{OUT}/model_s2_full_dens_calib.json", "w"))
    res["stage1"] = stage1
    res["seconds"] = round(time.time() - T0, 1)
    json.dump(res, open(f"{OUT}/orphan.json", "w"), indent=1, default=str)
    log("done")


if __name__ == "__main__":
    main()
