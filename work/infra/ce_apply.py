# -*- coding: utf-8 -*-
"""ce_apply.py -- test predictions with the cross-encoder stage-2 model (/vol/exp/ce/s2/model_ce), mirroring
run_pipeline.stage_predict: stage-1 matcher -> stage-2 frame (+ cross-encoder features) -> stage-2 -> density
adaptation (source priors refit on the CE stage-2 OOF) -> expected-F soft decision (empty_bias = best on OOF) ->
[targeted rule]. Two variants in one pass, like the v4 probes:
    ce_density  = adapt density_hard, no rule          (cf. submission_v4.json)
    ce_hsr_rule = adapt density_hs  + targeted rule    (cf. submission_v4_hsr.json)
    python infra/ce_apply.py
Writes /vol/output_<variant>/{matching_results.tsv, candidate_pairs.tsv}.
"""
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")
import polars as pl                 # noqa: E402

from ber import adapt as AD         # noqa: E402
from ber import decide as D         # noqa: E402
from ber import model as MD         # noqa: E402
from ber import outputs as O        # noqa: E402
from ber import stage2 as S2        # noqa: E402

W, CE = Path("/vol/work_v4"), Path("/vol/exp/ce")
DATA = "/root/proj/student_resource/dataset"
VARIANTS = {"ce_density": (list(AD.HARD_CLASSES), False),
            "ce_hsr_rule": (list(AD.HARD_CLASSES) + list(AD.SMALL_CLASSES), True)}
P_MIN, MAX_FACTOR = 1e-3, 4.0
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *a, flush=True)


def ce_feats(fr, sc):
    f32 = pl.Float32
    x = fr.join(sc.select("s1", "cand", pl.col("ce").cast(f32)), on=["s1", "cand"], how="left", maintain_order="left")
    mx = pl.col("ce").max().over("s1")
    second = pl.col("ce").drop_nulls().top_k(2).min().over("s1")
    return x.with_columns(
        pl.col("ce").rank("ordinal", descending=True).over("s1").cast(f32).alias("ce_rank"),
        (mx - pl.col("ce")).cast(f32).alias("ce_gap_top"),
        pl.when(pl.col("ce") >= mx).then(pl.col("ce") - second).otherwise(pl.col("ce") - mx).cast(f32).alias("ce_gap_other"))


b1 = MD.load_bundle(str(W / "model"))
cfg2 = S2.stage2_config_from_bundle(S2.load_bundle(str(W / "model" / "stage2")))   # frame = base stage-2 features
b2 = S2.load_bundle(str(CE / "s2" / "model_ce"))
rep = json.load(open(CE / "s2" / "report.json", encoding="utf-8"))
lam = float(rep["lam_missing"])
eb = float(max(rep["ce"]["oof"].items(), key=lambda kv: kv[1]["all"])[0])
log(f"decision: expected_f soft, lam {lam:.5f}, empty_bias {eb} (best CE stage-2 OOF)")

# adaptation source priors from the CE stage-2 OOF (as run_pipeline._fit_adapt_source)
oof = pl.read_parquet(CE / "s2" / "oof_ce.parquet")
parts = []
for c in sorted(oof["country"].unique().to_list()):
    x = oof.filter(pl.col("country") == c)
    parts.append(AD.tag_pairs(x, str(W / "norm" / f"train_{c}_s1.parquet"), str(W / "norm" / f"train_{c}_pool.parquet"), p_min=P_MIN))
src = AD.fit_source(pl.concat(parts, how="vertical_relaxed"), AD.AdaptConfig(method="density", p_min=P_MIN), country_col="country")
del oof, parts
log("adaptation source priors fitted")

res = {v: {"links": [], "scored": []} for v in VARIANTS}
for c in ("France", "India", "US"):
    t = time.time()
    fpath = W / "feats" / f"test_{c}.parquet"
    s1p, poolp = W / "norm" / f"test_{c}_s1.parquet", W / "norm" / f"test_{c}_pool.parquet"
    n_s1 = pl.scan_parquet(s1p).select(pl.len()).collect().item()
    scored = MD.predict_matcher(b1, pl.read_parquet(fpath))
    fr = S2.build_frame(scored, str(fpath), str(poolp), cfg2, stage1_bundle=b1)
    sc = pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(str(CE / f"score_test_{c}*.parquet")))])
    fr = ce_feats(fr, sc)
    cov = fr["ce"].is_not_null().mean()
    scored = S2.predict(b2, fr, scored=scored).select("s1", "cand", "p")
    del fr
    tagged0 = AD.tag_pairs(scored, str(s1p), str(poolp), p_min=P_MIN)
    for v, (classes, rule) in VARIANTS.items():
        acfg = AD.AdaptConfig(method="density", classes=classes, p_min=P_MIN, max_factor=MAX_FACTOR)
        tagged, est = AD.adapt(tagged0, src, acfg, country=c, n_s1=n_s1)
        sv = tagged.select("s1", "cand", "p")
        links = D.select_links(sv, "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=eb)
        msg = ""
        if rule:
            n0 = links.height
            links = AD.targeted_rule(links, tagged.select("s1", "cand", "p", "hn_rel"))
            msg = f", rule dropped {n0 - links.height:,}"
        res[v]["links"].append(links)
        res[v]["scored"].append(sv)
        log(f"{v} test/{c}: {links.height:,} links, {links.height / max(n_s1, 1):.3f} links/S1, sum p/S1 "
            f"{float(sv['p'].sum()) / max(n_s1, 1):.3f}{msg}")
    log(f"test/{c}: ce coverage {cov:.4f} of the stage-2 rows; {time.time() - t:.0f}s")
    del scored, tagged0

s1_ids = O.read_s1_order(f"{DATA}/test/test_source1.tsv")
for v, r in res.items():
    out = f"/vol/output_{v}"
    Path(out).mkdir(parents=True, exist_ok=True)
    log(f"write {v}:", O.write_submission(s1_ids, pl.concat(r["links"]), pl.concat(r["scored"]), out))
log("done")
