# -*- coding: utf-8 -*-
"""expB_c_adapt_train.py -- ber/adapt.py on the labelled train populations.

Frames (from expB_a_stage2.py): scored_oof / scored_hb = s1, cand, label, p1 (stage 1), p_g1 (stage 2 + guard G1).
Source priors per (country, house-number relation class) = mean p on the OOF frame (rows p >= 1e-3), one source per
model (stage 1 on p1, stage 2 on p_g1); saved to /vol/exp/expB/adapt_source_{s1,s2}.json.

(i)  neutrality: OOF (neutral by construction for source_stat='p') and HB (final-model p, no shift)
(ii) simulated density shift on HB: +0.8 x (HB HARD pairs with p1 >= 1e-3) NEGATIVE pairs per country, taken from
     OOF negatives of the same country in the HARD classes (p1 >= 1e-3), each attached to a random HB S1 of that
     country under a fresh record id (never a true match, never claimed by another S1) -> HARD pairs per S1 x 1.8.
       uniform : donors drawn uniformly  (pure label shift inside HARD: the model's P(x | y) is unchanged)
       tilted  : donors drawn with probability ~ p1 (the extra negatives look like copies: what test US shows,
                 3.4x selected HARD links for 1.8x HARD pairs)
     methods: none | em | em_hard (EM, HARD classes only) | density | density_hard | rule (diag_struct_k)
Writes /vol/exp/expB/c_results.json
"""
import json
import sys
import time

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
sys.path.insert(0, "/root/proj/infra")

import numpy as np            # noqa: E402
import polars as pl           # noqa: E402
import psutil                 # noqa: E402

from ber import adapt as A    # noqa: E402
import diag_stage2_lib as L   # noqa: E402

W3, DG, OUT = "/vol/work_v3", "/vol/diag/stage2", "/vol/exp/expB"
LAM, PMIN = 0.05303060038344832, 1e-3
T0 = time.time()
HARD = list(A.HARD_CLASSES)
METHODS = [("none", None), ("em", A.AdaptConfig(method="em")),
           ("em_hard", A.AdaptConfig(method="em", classes=HARD)),
           ("density", A.AdaptConfig(method="density")),
           ("density_hard", A.AdaptConfig(method="density", classes=HARD))]
R = {}


def log(*a):
    rss = psutil.Process().memory_info().rss / 2 ** 30
    print(f"{time.strftime('%H:%M:%S')} [{time.time() - T0:6.0f}s rss {rss:5.1f}G]", *a, flush=True)


def dump():
    json.dump(R, open(f"{OUT}/c_results.json", "w"), indent=1, default=str)


def dec(df):
    return df.with_columns([pl.format("S{}-{}", pl.col(c) // 10_000_000_000, pl.col(c) % 10_000_000_000).alias(c)
                            for c in ("s1", "cand")])


def load(tag):
    df = dec(pl.read_parquet(f"{OUT}/scored_{tag}.parquet"))
    parts = []
    for c in sorted(df["country"].unique().to_list()):
        x = df.filter(pl.col("country") == c)
        parts.append(A.tag_pairs(x, f"{W3}/norm/train_{c}_s1.parquet", f"{W3}/norm/train_{c}_pool.parquet",
                                 p_min=PMIN, p="p1"))
    return pl.concat(parts)


def hard_per_country(lk, frame, ev):
    x = lk.join(frame.select("s1", pl.col("cand").alias("mid"), "hn_rel"), on=["s1", "mid"], how="left")
    x = x.filter(pl.col("hn_rel").is_in(HARD)).join(ev, on="s1", how="left")
    n = dict(ev.group_by("country").len().rows())
    h = dict(x.group_by("country").len().rows())
    return {c: round(h.get(c, 0) / n[c], 5) for c in sorted(n)} | {"ALL": round(x.height / ev.height, 5)}


def run(frame, pcol, eb, src, ev, gt, tag):
    base = frame.select("s1", "cand", pl.col(pcol).alias("p"), "hn_rel", "country")
    res, ests, lk_none = {}, {}, None
    for name, cfg in METHODS:
        t = time.time()
        if cfg is None:
            sc = base
        else:
            parts = []
            for c in sorted(base["country"].unique().to_list()):
                ad, est = A.adapt(base.filter(pl.col("country") == c), src, cfg, country=c)
                parts.append(ad)
                ests[f"{name}|{c}"] = est.to_dicts()
            sc = pl.concat(parts)
        lk = L.select_fast(sc.select("s1", "cand", "p"), "soft", LAM, eb)
        if name == "none":
            lk_none = lk
        res[name] = {**L.score(lk, gt, ev), "links_per_s1": round(lk.height / ev.height, 4),
                     "sum_p_per_s1": round(float(sc["p"].sum()) / ev.height, 4),
                     "hard_links_per_s1": hard_per_country(lk, base, ev), "sec": round(time.time() - t, 1)}
        log(f"[{tag}] {name:13s} {res[name]}")
    lk = A.targeted_rule(lk_none, base)
    res["rule"] = {**L.score(lk, gt, ev), "links_per_s1": round(lk.height / ev.height, 4),
                   "hard_links_per_s1": hard_per_country(lk, base, ev)}
    log(f"[{tag}] {'rule':13s} {res['rule']}")
    return res, ests


def simulate(hb, oof, scenario, seed=0):
    rng = np.random.default_rng(seed)
    parts, info = [hb], {}
    for c in sorted(hb["country"].unique().to_list()):
        don = oof.filter((pl.col("country") == c) & (pl.col("label") == 0) & pl.col("hn_rel").is_in(HARD)
                         & (pl.col("p1") >= PMIN)).sort(["s1", "cand"])
        n_h = hb.filter((pl.col("country") == c) & pl.col("hn_rel").is_in(HARD) & (pl.col("p1") >= PMIN)).height
        n_add = int(round(0.8 * n_h))
        if scenario == "uniform":
            idx = rng.integers(0, don.height, n_add)
        else:
            w = don["p1"].cast(pl.Float64).to_numpy()
            idx = rng.choice(don.height, n_add, p=w / w.sum())
        s1s = hb.filter(pl.col("country") == c)["s1"].unique().sort().to_numpy()
        add = (don.select("p1", "p_g0", "p_g1", "hn_rel").select(pl.all().gather(pl.Series(idx)))
                  .with_columns(pl.Series("s1", rng.choice(s1s, n_add)),
                                pl.Series("cand", [f"S2-9{c[:2]}{k}" for k in range(n_add)]),
                                pl.lit(0, pl.Int8).alias("label"), pl.lit(c).alias("country")))
        parts.append(add.select(hb.columns))
        info[c] = {"hb_hard_pairs": n_h, "added": n_add, "donor_pool": don.height,
                   "donor_mean_p1": round(float(don["p1"].mean()), 4),
                   "added_mean_p1": round(float(add["p1"].mean()), 4)}
    return pl.concat(parts, how="vertical_relaxed"), info


def main():
    A_res = json.load(open(f"{OUT}/a_results.json"))
    eb2 = max((1.0, 1.5, 2.0), key=lambda eb: A_res["mine_OOF_G1"][f"eb{eb}"]["ALL"])
    models = {"s1": ("p1", 2.0), "s2": ("p_g1", eb2)}
    R["eb"] = {k: v[1] for k, v in models.items()}
    meta = pl.read_parquet(f"{DG}/meta.parquet").select(pl.col("entity_id").alias("s1"), "country", "grp")
    gt_all = pl.read_parquet("/vol/work/cache/train_gt_long.parquet").select("s1", "mid")
    ev = {g: meta.filter(pl.col("grp") == g).select("s1", "country") for g in (0, 1)}
    gt = {g: gt_all.join(ev[g], on="s1", how="semi") for g in (0, 1)}
    oof, hb = load("oof"), load("hb")
    log(f"tagged OOF {oof.height:,} / HB {hb.height:,}; classes {oof['hn_rel'].value_counts().sort('count', descending=True).rows()[:20]}")
    src = {}
    for m, (pcol, _) in models.items():
        src[m] = A.fit_source(oof.select("s1", "cand", pl.col(pcol).alias("p"), "hn_rel", "label", "country"),
                              A.AdaptConfig(p_min=PMIN))
        A.save_source(src[m], f"{OUT}/adapt_source_{m}.json")
        R[f"source_{m}"] = src[m]["priors"]
    dump()
    for m, (pcol, eb) in models.items():
        for g, tag, fr in ((0, "OOF", oof), (1, "HB", hb)):
            R[f"{tag}_{m}"], R[f"{tag}_{m}_est"] = run(fr, pcol, eb, src[m], ev[g], gt[g], f"{tag} {m}")
            dump()
        base = hb.select("s1", "cand", pl.col(pcol).alias("p"), "hn_rel")
        R[f"HB_{m}_profile"] = A.class_profile(base, None, ev[1].height).to_dicts()
    for scen in ("uniform", "tilted"):
        sim, info = simulate(hb, oof, scen)
        R[f"sim_{scen}_info"] = info
        log(f"sim {scen}: {info}")
        for m, (pcol, eb) in models.items():
            R[f"sim_{scen}_{m}"], R[f"sim_{scen}_{m}_est"] = run(sim, pcol, eb, src[m], ev[1], gt[1], f"sim-{scen} {m}")
            dump()
        del sim
    R["seconds"] = round(time.time() - T0, 1)
    dump()
    log("done")


if __name__ == "__main__":
    main()
