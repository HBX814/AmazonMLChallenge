# -*- coding: utf-8 -*-
"""diag_fr_c.py -- label-free shift test France vs US / India (test) vs train OOF (with labels).

C1  OOF decisions recomputed (select_links, v3 decision params) -> per-S1 true F on OOF (sanity: 0.98152)
C2  self-estimated E[F0.5] per S1 from calibrated p (Monte Carlo, soft-exclusivity posterior, Poisson(lam) missing),
    on OOF (vs the true F -> is the estimate trustworthy?) and on test France / US / India
C3  p distributions: selected links, rank 2..5 candidates, uncertain zone [0.3, 0.7)
C4  top-20 features by gain: mean / p10 / p50 / p90 + PSI vs train, for ALL pairs, SELECTED, rank-2..5 not selected
C5  risk segments (label-free definitions): rate per S1 on each test country and, on OOF, the true-match share
    of the segment -> expected FP / FN per S1 = rate x error share
Writes /vol/diag/fr/shift.md and /vol/diag/fr/shift_per_s1.parquet
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide, metric  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
LAM, EB = 0.05303060038344832, 2.0
N_S1 = 20_000
MD = []
TOP = ["prio", "comp_gap", "comp_is_best", "hn_diff", "n_tsort", "a_tset", "p_key_hit", "len_cand", "len_ratio",
       "legal_conflict", "n_passes", "idf_cand", "idf_overlap", "n_tset", "st_inter", "n_partial", "idf_contain",
       "grp_max_sim", "n_jw", "comp_best"]
EXTRA = ["hn_both", "hn_base", "hn_exact", "k_eq", "a_empty_any", "c_tset", "amb_s1_c", "comp_nclaim", "city_eq"]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def md(*lines):
    MD.extend(lines)


def table(header, rows):
    md("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
    md("")


def psi(ref, cur, bins=10):
    ref = np.asarray(ref, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=edges.size + 1) / ref.size
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=edges.size + 1) / cur.size
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


# ---------------------------------------------------------------------------------------------------- data
train_country = pl.concat([pl.scan_parquet(f"{V3}/norm/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1"))
                           .with_columns(pl.lit(c).alias("country")).collect() for c in ("US", "India")])
oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
log("oof", oof.height, oof["s1"].n_unique())
oof_ids = oof.select("s1").unique().join(train_country, on="s1", how="left")
gt = pl.scan_parquet("/vol/work/cache/train_gt_long.parquet").select("s1", "mid").join(
    oof_ids.lazy().select("s1"), on="s1", how="semi").collect()
t = time.time()
oof_links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
log("oof select_links", oof_links.height, f"{time.time() - t:.0f}s")
per = metric.per_s1_scores(oof_links, gt, oof_ids["s1"]).join(oof_ids, on="s1", how="left")
md("# Shift report France vs US / India (test) vs train OOF", "",
   f"OOF recomputed decisions: macro F0.5 {per['f'].mean():.5f} (decision.json 0.98152); "
   + ", ".join(f"{c} {per.filter(pl.col('country') == c)['f'].mean():.5f}" for c in ("US", "India")), "")


def soft_post(scored):
    """soft-exclusivity posterior over the whole frame (same as decide.exclusivity_soft)."""
    pc = pl.col("p").clip(0.0, 1.0 - 1e-6)
    return scored.with_columns((pc / (1 - pc)).alias("_o")).with_columns(
        (pl.col("_o") / (1 + pl.col("_o").sum().over("cand"))).alias("pp")).drop("_o")


def mc_expected_f(frame, n_draw=300, seed=0):
    """frame: s1, pp, sel (bool). Returns s1, ef (Monte Carlo E[F0.5] of the selected set under independent
    Bernoulli(pp) truths + Poisson(LAM) matches outside the list)."""
    rng = np.random.default_rng(seed)
    f = frame.sort("s1", "pp", descending=[False, True]).with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
    f = f.filter(pl.col("_r") < 60)
    ids = f["s1"].unique(maintain_order=True)
    idx = {s: i for i, s in enumerate(ids.to_list())}
    B = len(idx)
    row = np.array([idx[s] for s in f["s1"].to_list()])
    col = f["_r"].to_numpy()
    P = np.zeros((B, 60))
    S = np.zeros((B, 60), dtype=bool)
    P[row, col] = f["pp"].to_numpy()
    S[row, col] = f["sel"].to_numpy()
    out = np.zeros(B)
    m = S.sum(1)
    for lo in range(0, B, 4000):
        Pb, Sb, mb = P[lo:lo + 4000], S[lo:lo + 4000], m[lo:lo + 4000]
        acc = np.zeros(Pb.shape[0])
        for _ in range(n_draw // 50):
            Z = rng.random((50,) + Pb.shape) < Pb[None]
            Q = rng.poisson(LAM, (50, Pb.shape[0]))
            tp = (Z & Sb[None]).sum(2)
            T = Z.sum(2) + Q
            F = np.where(mb[None] == 0, (T == 0).astype(float),
                         np.where(tp == 0, 0.0, 1.25 * tp / np.maximum(0.25 * T + mb[None], 1e-9)))
            acc += F.sum(0)
        out[lo:lo + 4000] = acc / (n_draw // 50 * 50)
    return pl.DataFrame({"s1": ids, "ef": out})


# ---------------------------------------------------------------------------------------------------- sample frames
rng_seed = 20260926
groups = {}
# train OOF (labels) -- per country sample of S1
oofp = soft_post(oof).join(oof_links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
    .with_columns(pl.col("sel").fill_null(False))
for c in ("US", "India"):
    ids = oof_ids.filter(pl.col("country") == c)["s1"].sample(N_S1, seed=rng_seed)
    groups[("train", c)] = oofp.filter(pl.col("s1").is_in(ids.to_list()))
del oofp
for c in ("France", "US", "India"):
    sc = pl.scan_parquet(f"{V3}/pred/test_{c}_scored.parquet")
    ids = pl.scan_parquet(f"{V3}/norm/test_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).collect()["s1"].sample(N_S1, seed=rng_seed)
    part = sc.filter(pl.col("s1").is_in(ids.to_list())).collect()
    # odds sum per cand over ALL S1 of the country (soft exclusivity)
    osum = (sc.filter(pl.col("cand").is_in(part["cand"].unique().to_list()))
              .with_columns((pl.col("p").clip(0.0, 1 - 1e-6) / (1 - pl.col("p").clip(0.0, 1 - 1e-6))).alias("_o"))
              .group_by("cand").agg(pl.col("_o").sum().alias("_os")).collect())
    part = part.join(osum, on="cand", how="left").with_columns(
        (pl.col("p").clip(0.0, 1 - 1e-6) / (1 - pl.col("p").clip(0.0, 1 - 1e-6)) / (1 + pl.col("_os"))).alias("pp")).drop("_os")
    links = pl.scan_parquet(f"{V3}/pred/test_{c}_links.parquet").filter(pl.col("s1").is_in(ids.to_list())).collect()
    part = part.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("sel").fill_null(False))
    groups[("test", c)] = part
    log("sample", c, part.height, int(part["sel"].sum()))
# per-pair rank by p
for k in groups:
    groups[k] = groups[k].with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("r"))

# ---------------------------------------------------------------------------------------------------- C2 self-estimated F
md("## C2 self-estimated E[F0.5] (model's own belief, calibrated p + soft exclusivity + Poisson missing)", "")
rows = []
per_all = []
for (split, c), g in groups.items():
    ef = mc_expected_f(g.select("s1", "pp", "sel"))
    allids = g.select("s1").unique()
    ef = allids.join(ef, on="s1", how="left").with_columns(pl.col("ef").fill_null(float(np.exp(-LAM))))
    true_f = None
    if split == "train":
        tf = per.filter(pl.col("s1").is_in(allids["s1"].to_list()))
        true_f = float(tf["f"].mean())
        ef = ef.join(tf.select("s1", "f", "k", "m", "tp"), on="s1", how="left")
    nsel = g.group_by("s1").agg(pl.col("sel").sum().alias("m_sel"))
    ef = ef.join(nsel, on="s1", how="left").with_columns(pl.lit(split).alias("split"), pl.lit(c).alias("country"))
    per_all.append(ef.select("split", "country", "s1", "ef", "m_sel", *([c_ for c_ in ("f", "k", "m", "tp") if c_ in ef.columns])))
    rows.append((split, c, ef.height, float(ef["ef"].mean()), true_f if true_f is not None else "-",
                 float((ef["ef"] < 0.9).mean()), float((ef["ef"] < 0.5).mean()), float((ef["m_sel"] == 0).mean())))
table(["split", "country", "n_s1", "E[F] self", "true F (OOF)", "share E[F]<0.9", "share E[F]<0.5", "empty rate"], rows)
pl.concat(per_all, how="diagonal").write_parquet(f"{OUT}/shift_per_s1.parquet")
log("C2 done")

# ---------------------------------------------------------------------------------------------------- C3 p distributions
md("## C3 p distributions", "")
rows = []
for (split, c), g in groups.items():
    sel = g.filter(pl.col("sel"))["p"]
    r25 = g.filter((pl.col("r") >= 2) & (pl.col("r") <= 5))["p"]
    r25n = g.filter((pl.col("r") >= 2) & (pl.col("r") <= 5) & ~pl.col("sel"))["p"]
    unc = g.filter((pl.col("p") >= 0.3) & (pl.col("p") < 0.7))
    ns1 = g["s1"].n_unique()
    rows.append((split, c, float(sel.mean()), float(sel.quantile(0.05)), float((sel < 0.9).mean()), float((sel < 0.7).mean()),
                 float(r25.mean()), float(r25.median()), float((r25n > 0.1).mean()), float((r25n > 0.3).mean()),
                 unc.height / ns1, unc["s1"].n_unique() / ns1, float(g.filter(pl.col("sel")).height / ns1)))
table(["split", "country", "sel p mean", "sel p q05", "sel p<0.9", "sel p<0.7", "r2-5 p mean", "r2-5 p med",
       "r2-5 unsel p>0.1", "r2-5 unsel p>0.3", "unc pairs/S1", "S1 with unc", "links/S1"], rows)
# p histogram of non-selected candidates, rank <= 8
bins = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
rows = []
for (split, c), g in groups.items():
    ns = g.filter(~pl.col("sel") & (pl.col("r") <= 8))["p"].to_numpy()
    h = np.histogram(ns, bins=bins)[0] / g["s1"].n_unique()
    rows.append((split, c, *[float(x) for x in h]))
md("non-selected candidates of rank <= 8: count per S1 by p bin")
table(["split", "country"] + [f"[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)], rows)
# OOF: true-match share by p bin for non-selected + selected
rows = []
for c in ("US", "India"):
    g = groups[("train", c)]
    for s in (False, True):
        for i in range(len(bins) - 1):
            x = g.filter((pl.col("sel") == s) & (pl.col("p") >= bins[i]) & (pl.col("p") < bins[i + 1]))
            if x.height:
                rows.append((c, "sel" if s else "unsel", f"[{bins[i]},{bins[i + 1]})", x.height, float(x["label"].mean())))
md("OOF true-match share by p bin (calibration check)")
table(["country", "set", "p bin", "n", "true share"], rows)
log("C3 done")

# ---------------------------------------------------------------------------------------------------- C4 feature shift
md("## C4 feature shift (top-20 by gain)", "", "PSI vs the same group in TRAIN (pooled US+India for France). "
   "ALL = every candidate pair of the sampled S1, SEL = selected links, R25 = rank 2..5 not selected.", "")
feats = {}
cols = ["s1", "cand"] + TOP + EXTRA
for (split, c), g in groups.items():
    path = f"{V3}/feats/{split}_{c}.parquet"
    ids = g["s1"].unique().to_list()
    F = pl.scan_parquet(path).select(cols).filter(pl.col("s1").is_in(ids)).collect()
    feats[(split, c)] = g.join(F, on=["s1", "cand"], how="inner")
    log("feats", split, c, F.height, feats[(split, c)].height)
ref_pool = {k: pl.concat([feats[("train", "US")], feats[("train", "India")]], how="diagonal") for k in ("x",)}["x"]


def sub(df, grp):
    if grp == "ALL":
        return df
    if grp == "SEL":
        return df.filter(pl.col("sel"))
    return df.filter((pl.col("r") >= 2) & (pl.col("r") <= 5) & ~pl.col("sel"))


for grp in ("ALL", "SEL", "R25"):
    rows = []
    for f in TOP:
        ref_all = sub(ref_pool, grp)[f].to_numpy()
        r = [f]
        for (split, c) in (("train", "US"), ("train", "India"), ("test", "US"), ("test", "India"), ("test", "France")):
            v = sub(feats[(split, c)], grp)[f].to_numpy()
            ref = sub(feats[("train", c)], grp)[f].to_numpy() if c != "France" else ref_all
            r.append(f"{v.mean():.3f} [{np.quantile(v, 0.1):.2f},{np.quantile(v, 0.5):.2f},{np.quantile(v, 0.9):.2f}]")
            if split == "test":
                r.append(round(psi(ref, v), 4))
        rows.append(r)
    md(f"### {grp}")
    table(["feature", "train US mean [p10,p50,p90]", "train India", "test US", "PSI US", "test India", "PSI India",
           "test France", "PSI France"], rows)
log("C4 done")

# ---------------------------------------------------------------------------------------------------- C5 risk segments
md("## C5 risk segments (label-free definitions)", "",
   "rate = pairs per S1 in the segment; OOF true share = share of those pairs that are true matches (train labels). "
   "exp. errors/S1 = rate x (1 - true share) for selected segments (FP) or rate x true share for unselected (FN).", "")
SEG = {
    "sel & hn mismatch (both hn, base differ)": (pl.col("sel") & (pl.col("hn_both") == 1) & (pl.col("hn_base") == 0)),
    "sel & legal_conflict": (pl.col("sel") & (pl.col("legal_conflict") == 1)),
    "sel & cand addr empty": (pl.col("sel") & (pl.col("a_empty_any") == 1)),
    "sel & p<0.9": (pl.col("sel") & (pl.col("p") < 0.9)),
    "sel & c_tset<0.6": (pl.col("sel") & (pl.col("c_tset") < 0.6)),
    "unsel & k_eq & addr empty": (~pl.col("sel") & (pl.col("k_eq") == 1) & (pl.col("a_empty_any") == 1)),
    "unsel & k_eq & hn_exact": (~pl.col("sel") & (pl.col("k_eq") == 1) & (pl.col("hn_exact") == 1)),
    "unsel & hn_exact & st_inter>=2": (~pl.col("sel") & (pl.col("hn_exact") == 1) & (pl.col("st_inter") >= 2)),
    "unsel & p in [0.3,0.7)": (~pl.col("sel") & (pl.col("p") >= 0.3) & (pl.col("p") < 0.7)),
    "unsel & p in [0.05,0.3)": (~pl.col("sel") & (pl.col("p") >= 0.05) & (pl.col("p") < 0.3)),
    "unsel & k_eq & hn mismatch": (~pl.col("sel") & (pl.col("k_eq") == 1) & (pl.col("hn_both") == 1) & (pl.col("hn_base") == 0)),
}
rows = []
for name, cond in SEG.items():
    r = [name]
    shares = {}
    for c in ("US", "India"):
        g = feats[("train", c)]
        x = g.filter(cond)
        shares[c] = float(x["label"].mean()) if x.height else float("nan")
        r += [x.height / g["s1"].n_unique(), shares[c]]
    pooled = np.nanmean([shares["US"], shares["India"]])
    for c in ("US", "India", "France"):
        g = feats[("test", c)]
        x = g.filter(cond)
        rate = x.height / g["s1"].n_unique()
        err = rate * ((1 - pooled) if name.startswith("sel") else pooled)
        r += [rate, err]
    rows.append(r)
table(["segment", "OOF US rate", "OOF US true", "OOF IN rate", "OOF IN true", "test US rate", "US exp.err/S1",
       "test IN rate", "IN exp.err/S1", "test FR rate", "FR exp.err/S1"], rows)
with open(f"{OUT}/shift.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("done")
