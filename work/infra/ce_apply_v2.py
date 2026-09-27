# -*- coding: utf-8 -*-
"""ce_apply_v2.py -- test predictions with the stage-2 model chosen by ce_s2_v2.py (/vol/exp/ce2/s2/best.json), mirroring
run_pipeline.stage_predict: stage-1 -> stage-2 frame + cross-encoder features (v1 and/or v2) -> stage 2 -> density
adaptation (priors refit on that model's OOF) -> expected-F soft decision (eb best on OOF) -> [targeted rule].
Variants written: /vol/output_<tag>_density (density_hard) and /vol/output_<tag>_hsr_rule (density_hs + rule).
    python infra/ce_apply_v2.py [--tag ce2]
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import polars as pl                 # noqa: E402

from ber import adapt as AD         # noqa: E402
from ber import crossenc as CX      # noqa: E402
from ber import decide as D         # noqa: E402
from ber import model as MD         # noqa: E402
from ber import outputs as O        # noqa: E402
from ber import stage2 as S2        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="ce2")
ap.add_argument("--variant", default=None, help="override best.json")
ap.add_argument("--s2-dir", default="s2")
ap.add_argument("--ce2-file", default="kg_scores.parquet")
a = ap.parse_args()
W, CE1, CE2 = Path("/vol/work_v4"), Path("/vol/exp/ce"), Path("/vol/exp/ce2")
DATA = "/root/proj/student_resource/dataset"
P_MIN, MAX_FACTOR = 1e-3, 4.0
T0 = time.time()


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


S2DIR = CE2 / a.s2_dir
best = json.load(open(S2DIR / "best.json", encoding="utf-8"))
v = a.variant or best["variant"]
tune = json.load(open(S2DIR / f"tune_{v}.json", encoding="utf-8"))
ces, lam = tune["ce"], float(tune["lam_missing"])
eb = float(tune["eb_best_oof"])
VARIANTS = {f"{a.tag}_density": (list(AD.HARD_CLASSES), False),
            f"{a.tag}_hsr_rule": (list(AD.HARD_CLASSES) + list(AD.SMALL_CLASSES), True)}
log(f"stage-2 variant {v} (CE {ces}), OOF {tune['oof_best']:.5f} HB {tune['hb_best']:.5f}; decision soft, lam {lam:.5f}, eb {eb}")


def test_scores(name, c):
    if name == "ce":
        return pl.concat([pl.read_parquet(p).select("s1", "cand", pl.col("ce").alias("logit"))
                          for p in sorted(glob.glob(str(CE1 / f"score_test_{c}*.parquet")))])
    kg = {"ce2": a.ce2_file, "ce3": "kg_scores_e5l.parquet"}[name]
    return pl.read_parquet(CE2 / kg).filter(pl.col("set") == f"test_{c}").select(
        "s1", "cand", pl.col("ce2").alias("logit"))


b1 = MD.load_bundle(str(W / "model"))
cfg2 = S2.stage2_config_from_bundle(S2.load_bundle(str(W / "model" / "stage2")))   # frame = base stage-2 features
b2 = S2.load_bundle(str(S2DIR / f"model_{v}"))
oof = pl.read_parquet(S2DIR / f"oof_{v}.parquet")
parts = [AD.tag_pairs(oof.filter(pl.col("country") == c), str(W / "norm" / f"train_{c}_s1.parquet"),
                      str(W / "norm" / f"train_{c}_pool.parquet"), p_min=P_MIN) for c in ("India", "US")]
src = AD.fit_source(pl.concat(parts, how="vertical_relaxed"), AD.AdaptConfig(method="density", p_min=P_MIN), country_col="country")
del oof, parts
res = {k: {"links": [], "scored": []} for k in VARIANTS}
for c in ("France", "India", "US"):
    t = time.time()
    fpath = W / "feats" / f"test_{c}.parquet"
    s1p, poolp = W / "norm" / f"test_{c}_s1.parquet", W / "norm" / f"test_{c}_pool.parquet"
    n_s1 = pl.scan_parquet(s1p).select(pl.len()).collect().item()
    scored = MD.predict_matcher(b1, pl.read_parquet(fpath))
    fr = S2.build_frame(scored, str(fpath), str(poolp), cfg2, stage1_bundle=b1)
    for name in ces:
        fr = CX.add_features(fr, test_scores(name, c), name)
    cov = {name: round(float(fr[name].is_not_null().mean()), 4) for name in ces}
    scored = S2.predict(b2, fr, scored=scored).select("s1", "cand", "p")
    del fr
    tagged0 = AD.tag_pairs(scored, str(s1p), str(poolp), p_min=P_MIN)
    for k, (classes, rule) in VARIANTS.items():
        tagged, est = AD.adapt(tagged0, src, AD.AdaptConfig(method="density", classes=classes, p_min=P_MIN,
                                                            max_factor=MAX_FACTOR), country=c, n_s1=n_s1)
        sv = tagged.select("s1", "cand", "p")
        links = D.select_links(sv, "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=eb)
        msg = ""
        if rule:
            n0 = links.height
            links = AD.targeted_rule(links, tagged.select("s1", "cand", "p", "hn_rel"))
            msg = f", rule dropped {n0 - links.height:,}"
        res[k]["links"].append(links)
        res[k]["scored"].append(sv)
        log(f"{k} test/{c}: {links.height:,} links, {links.height / max(n_s1, 1):.3f} links/S1, sum p/S1 "
            f"{float(sv['p'].sum()) / max(n_s1, 1):.3f}{msg}")
    log(f"test/{c}: CE coverage {cov}; {time.time() - t:.0f}s")
    del scored, tagged0
s1_ids = O.read_s1_order(f"{DATA}/test/test_source1.tsv")
for k, r in res.items():
    out = f"/vol/output_{k}"
    Path(out).mkdir(parents=True, exist_ok=True)
    log(f"write {k}:", O.write_submission(s1_ids, pl.concat(r["links"]), pl.concat(r["scored"]), out))
log("done")
