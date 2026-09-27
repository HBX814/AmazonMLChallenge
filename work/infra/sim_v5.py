# -*- coding: utf-8 -*-
"""sim_v5.py -- choose the shift-adaptation / rule / empty-bias settings for the CE stage-2 model (v1v2) on the honest
holdout HB (buckets 150-299), without and with a SIMULATED test-like hard-negative density shift.
Shift: per country, add (r - 1) x (HB pairs in the HARD house-number classes with p1 >= 1e-3) NEGATIVE pairs, drawn from
OOF negatives of the same country and classes with probability ~ stage-1 p1 (copy-like distractors, as diagnosed on
test), each attached to a random HB S1 under a fresh record id. r = measured test/train density ratio (US 1.84,
India 1.12); a stronger scenario uses r = 2.5 for both. Settings are scored with the exact decision + macro F0.5.
    python infra/sim_v5.py
Writes /vol/exp/ce2/s2/sim_v5.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
import numpy as np                  # noqa: E402
import polars as pl                 # noqa: E402

from ber import adapt as AD         # noqa: E402
from ber import crossenc as CX      # noqa: E402
from ber import decide as D         # noqa: E402
from ber import io as bio           # noqa: E402
from ber import metric as M         # noqa: E402
from ber import stage2 as S2        # noqa: E402

W, CE1, CE2 = Path("/vol/work_v4"), Path("/vol/exp/ce"), Path("/vol/exp/ce2")
S2D = CE2 / "s2"
V = "v1v2"
PMIN = 1e-3
HARD, SMALL = list(AD.HARD_CLASSES), list(AD.SMALL_CLASSES)
T0 = time.time()
R = {}


def log(*x):
    print(time.strftime("%H:%M:%S"), f"[{time.time() - T0:5.0f}s]", *x, flush=True)


def bucket(s):
    return pl.Series(M.fold_of(s, n_folds=1000, seed=7919))


tune = json.load(open(S2D / f"tune_{V}.json"))
lam = float(tune["lam_missing"])
b2 = S2.load_bundle(str(S2D / f"model_{V}"))
gt_all = bio.scan_split(W, "train")["gt"].collect()
hb_parts, oof_parts = [], []
oofv = pl.read_parquet(S2D / f"oof_{V}.parquet")                          # s1, cand, label, p (final), country
p1_oof = pl.read_parquet(W / "model" / "oof.parquet").select("s1", "cand", pl.col("p").alias("p1"))
for c in ("India", "US"):
    pf = pl.read_parquet(CE1 / f"pfull_train_{c}.parquet").select("s1", "cand", "p", "label")
    bk = bucket(pf["s1"])
    hb = pf.filter((bk >= 150) & (bk < 300)).rename({"p": "p1"})
    fr = pl.read_parquet(CE1 / "s2" / f"frame_{c}.parquet")
    fr = fr.filter(bucket(fr["s1"]) >= 150)
    fr = CX.add_features(fr, pl.concat([pl.read_parquet(p) for p in sorted(CE1.glob(f"score_hb_{c}*.parquet"))])
                         .select("s1", "cand", pl.col("ce").alias("logit")), "ce")
    kg = pl.read_parquet(CE2 / "kg_scores.parquet").filter(pl.col("set") == f"hb_{c}")
    fr = CX.add_features(fr, kg.select("s1", "cand", pl.col("ce2").alias("logit")), "ce2")
    res = S2.predict(b2, fr)                                               # s1, cand, p1, p2, p
    hb = hb.join(res.select("s1", "cand", pl.col("p").alias("pf")), on=["s1", "cand"], how="left").with_columns(
        pl.coalesce("pf", "p1").alias("p")).drop("pf").with_columns(pl.lit(c).alias("country"))
    hb_parts.append(AD.tag_pairs(hb, str(W / "norm" / f"train_{c}_s1.parquet"), str(W / "norm" / f"train_{c}_pool.parquet"),
                                 p_min=PMIN))
    o = oofv.filter(pl.col("country") == c).join(p1_oof, on=["s1", "cand"], how="left")
    oof_parts.append(AD.tag_pairs(o, str(W / "norm" / f"train_{c}_s1.parquet"), str(W / "norm" / f"train_{c}_pool.parquet"),
                                  p_min=PMIN))
    log(f"{c}: HB {hb.height:,} pairs; stage-2 rows {res.height:,}")
hb = pl.concat(hb_parts, how="vertical_relaxed")
oof = pl.concat(oof_parts, how="vertical_relaxed")
del hb_parts, oof_parts
ev = hb.select("s1", "country").unique()
gt = gt_all.join(ev.select("s1"), on="s1", how="semi")
src = AD.fit_source(oof.select("s1", "cand", "p", "hn_rel", "label", "country"), AD.AdaptConfig(method="density", p_min=PMIN),
                    country_col="country")
log(f"HB {hb.height:,} rows, {ev.height:,} S1; OOF tagged {oof.height:,}")

SETTINGS = [  # name, classes (None = no adaptation), max_factor, rule p_max (None = no rule)
    ("none", None, None, None), ("density_hard", HARD, 4.0, None), ("density_hs", HARD + SMALL, 4.0, None),
    ("hsr_rule", HARD + SMALL, 4.0, 0.99), ("hsr8_rule", HARD + SMALL, 8.0, 0.99), ("hsr_rule999", HARD + SMALL, 4.0, 0.999),
    ("hard_rule", HARD, 4.0, 0.99), ("none_rule", None, None, 0.99)]


def run(frame, tag, ebs=(1.0, 1.5, 2.0)):
    out = {}
    base = frame.select("s1", "cand", "p", "hn_rel", "country")
    for name, classes, mf, pmax in SETTINGS:
        if classes is None:
            sc = base
        else:
            parts = []
            for c in sorted(base["country"].unique().to_list()):
                x = base.filter(pl.col("country") == c)
                ad, _ = AD.adapt(x, src, AD.AdaptConfig(method="density", classes=classes, p_min=PMIN, max_factor=mf),
                                 country=c, n_s1=ev.filter(pl.col("country") == c).height)
                parts.append(ad)
            sc = pl.concat(parts, how="vertical_relaxed")
        for eb in ebs:
            lk = D.select_links(sc.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=lam, empty_bias=eb)
            if pmax is not None:
                lk = AD.targeted_rule(lk, sc.select("s1", "cand", "p", "hn_rel"), p_max=pmax)
            f = M.macro_f05(lk, gt, ev["s1"])
            out[f"{name}|eb{eb}"] = {"F": f, "links_per_s1": lk.height / ev.height}
            log(f"[{tag}] {name:12s} eb {eb}: F {f:.5f}  links/S1 {lk.height / ev.height:.4f}")
    return out


def simulate(ratio, seed=0):
    rng = np.random.default_rng(seed)
    parts, info = [hb], {}
    for c in sorted(hb["country"].unique().to_list()):
        r = ratio[c] if isinstance(ratio, dict) else ratio
        don = oof.filter((pl.col("country") == c) & (pl.col("label") == 0) & pl.col("hn_rel").is_in(HARD)
                         & (pl.col("p1") >= PMIN)).sort(["s1", "cand"])
        n_h = hb.filter((pl.col("country") == c) & pl.col("hn_rel").is_in(HARD) & (pl.col("p1") >= PMIN)).height
        n_add = int(round((r - 1.0) * n_h))
        w = don["p1"].cast(pl.Float64).to_numpy()
        idx = rng.choice(don.height, n_add, p=w / w.sum())
        s1s = hb.filter(pl.col("country") == c)["s1"].unique().sort().to_numpy()
        add = (don.select("p1", "p", "hn_rel").select(pl.all().gather(pl.Series(idx)))
                  .with_columns(pl.Series("s1", rng.choice(s1s, n_add)),
                                pl.Series("cand", [f"S2-9{c[:2]}{k}" for k in range(n_add)]),
                                pl.lit(0, pl.Int8).alias("label"), pl.lit(c).alias("country")))
        parts.append(add.select([x for x in hb.columns if x in add.columns]))
        info[c] = {"hb_hard_pairs": n_h, "added": n_add, "added_mean_p1": float(add["p1"].mean()),
                   "added_mean_p": float(add["p"].mean())}
    return pl.concat(parts, how="diagonal_relaxed"), info


R["hb_noshift"] = run(hb, "HB")
json.dump(R, open(S2D / "sim_v5.json", "w"), indent=1)
for scen, ratio in (("measured", {"US": 1.84, "India": 1.12}), ("strong", 2.5)):
    sim, info = simulate(ratio)
    R[f"sim_{scen}_info"] = info
    log(f"sim {scen}: {info}")
    R[f"sim_{scen}"] = run(sim, f"sim-{scen}")
    json.dump(R, open(S2D / "sim_v5.json", "w"), indent=1)
log("done")
