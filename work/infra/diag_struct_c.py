# -*- coding: utf-8 -*-
"""diag_struct_c.py -- cluster reasoning on the stage-1 OOF (TRAIN labels for measurement only).
Reproduces the v3 decision on /vol/work_v3/model/oof.parquet, then asks for every OOF error whether a sibling
relation to the S1's SELECTED links (same-source identical address string, identical name, ...) would have fixed it,
and simulates simple add / remove rules (thresholds tuned on folds 0-2, reported on folds 3-4).
    python infra/diag_struct_c.py
"""
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide, metric  # noqa: E402

OUT = os.environ.get("DIAG_OUT", "/vol/diag/struct")
NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
GTP = os.environ.get("DIAG_GT", "/vol/work/cache/train_gt_long.parquet")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
os.makedirs(OUT, exist_ok=True)
W = int(os.environ.get("OMP_NUM_THREADS", "8"))
T0 = time.time()
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(70)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
s1_ids = oof.select("s1").unique()
gt_all = pl.read_parquet(GTP)
gt = gt_all.join(s1_ids, on="s1", how="semi")
log("oof", oof.height, "S1", s1_ids.height, "gt links", gt.height)
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft",
                            lam_missing=0.05303060038344832, empty_bias=2.0)
F0 = metric.macro_f05(links, gt, s1_ids)
log(f"reproduced v3 OOF macro F0.5 = {F0:.5f} (links {links.height:,})")
oof = oof.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left").with_columns(
    pl.col("sel").fill_null(False), pl.col("label").cast(pl.Boolean))
folds = metric.fold_of(s1_ids["s1"], n_folds=5, seed=0)
fold = s1_ids.with_columns(pl.Series("fold", folds))
oof = oof.join(fold, on="s1")

# candidate attributes ------------------------------------------------------------------------------
R = oof.filter((pl.col("p") >= 0.002) | pl.col("label") | pl.col("sel"))
log("reduced rows", R.height)
attrs = []
for c in ("US", "India"):
    a = (pl.scan_parquet(f"{NORM}/train_{c}_pool.parquet")
         .select(pl.col("entity_id").alias("cand"), "business_name", "business_address", "name_norm", "name_core", "addr_norm")
         .join(R.select("cand").unique().lazy(), on="cand", how="semi").collect())
    attrs.append(a.with_columns(pl.lit(c).alias("country")))
attrs = pl.concat(attrs).with_columns(
    pl.col("cand").str.slice(0, 2).alias("src"),
    pl.col("business_address").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("acf"),
    pl.col("business_name").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("ncf"))
R = R.join(attrs.select("cand", "src", "acf", "ncf", "name_norm", "name_core", "addr_norm", "country"), on="cand", how="left")
R = R.with_columns((pl.col("acf").fill_null("") == "").alias("aempty"))
claimed = R.filter(pl.col("sel")).select("cand").unique().with_columns(pl.lit(True).alias("claimed"))
# calibration of p by empty address (p>=0.002 subset)
cal = (R.with_columns(pl.col("p").cut([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]).alias("bin"))
       .group_by("aempty", "bin").agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p"), pl.col("label").mean().alias("label_rate"),
                                       pl.col("sel").mean().alias("sel_rate")).sort("aempty", "bin"))
print("calibration by empty address (candidate rows with p>=0.002 or true or selected):\n", cal)
amax = R.group_by("cand").agg(pl.col("p").max().alias("pmax_c"))
R = R.join(amax, on="cand").with_columns((pl.col("p") >= pl.col("pmax_c")).alias("is_argmax"))

U = R.filter(~pl.col("sel")).join(claimed, on="cand", how="left").with_columns(pl.col("claimed").fill_null(False))
S = R.filter(pl.col("sel")).select("s1", pl.col("cand").alias("cand_s"), pl.col("src").alias("src_s"), pl.col("acf").alias("acf_s"),
                                   pl.col("ncf").alias("ncf_s"), pl.col("name_norm").alias("nn_s"), pl.col("addr_norm").alias("an_s"),
                                   pl.col("label").alias("label_s"))
pr = U.select("s1", "cand", "src", "acf", "ncf", "name_norm", "addr_norm").join(S, on="s1")
log("U", U.height, "pairs", pr.height)
pr = pr.with_columns(pl.Series("nrat", process.cpdist(pr["name_norm"].fill_null("").to_list(), pr["nn_s"].fill_null("").to_list(),
                                                      scorer=fuzz.ratio, workers=W)))
ne = pl.col("acf").fill_null("") != ""
rel = pr.group_by("s1", "cand").agg(
    (ne & (pl.col("acf") == pl.col("acf_s")) & (pl.col("src") == pl.col("src_s"))).any().alias("addr_twin_src"),
    (ne & (pl.col("acf") == pl.col("acf_s"))).any().alias("addr_twin_any"),
    ((pl.col("addr_norm").fill_null("") != "") & (pl.col("addr_norm") == pl.col("an_s"))).any().alias("an_twin"),
    (pl.col("ncf") == pl.col("ncf_s")).any().alias("name_twin"),
    (pl.col("name_norm") == pl.col("nn_s")).any().alias("nn_twin"),
    pl.col("nrat").max().alias("max_nrat"),
    (pl.col("src") == pl.col("src_s")).sum().alias("n_sel_same_src"),
    pl.len().alias("n_sel"))
U = U.join(rel, on=["s1", "cand"], how="left").with_columns(
    [pl.col(x).fill_null(False) for x in ("addr_twin_src", "addr_twin_any", "an_twin", "name_twin", "nn_twin")] +
    [pl.col("max_nrat").fill_null(0.0), pl.col("n_sel_same_src").fill_null(0), pl.col("n_sel").fill_null(0)])
FN = U.filter(pl.col("label"))
log(f"FN_model (true candidate not selected): {FN.height:,}; empty-address share {FN['aempty'].mean():.4f}; claimed by another S1 {FN['claimed'].mean():.4f}")
print("share of FN with relation:", {k: round(float(FN[k].mean()), 4) for k in ("addr_twin_src", "addr_twin_any", "an_twin", "name_twin", "nn_twin")},
      "max_nrat>=95:", round(float((FN["max_nrat"] >= 95).mean()), 4), "S1 has no selected link:", round(float((FN["n_sel"] == 0).mean()), 4))
print("FN by aempty x name_twin:", FN.group_by("aempty", "name_twin").agg(pl.len()).sort("aempty", "name_twin").rows())


def pbin(e):
    return (pl.when(e < 0.05).then(pl.lit("a<.05")).when(e < 0.2).then(pl.lit("b.05-.2")).when(e < 0.5).then(pl.lit("c.2-.5"))
            .otherwise(pl.lit("d>=.5")))


print("\nUNSELECTED candidates (p>=0.002 or true): label rate = precision if added (unclaimed only)")
Uu = U.filter(~pl.col("claimed")).with_columns(pbin(pl.col("p")).alias("pb"))
for relc in ("addr_twin_src", "name_twin", "nn_twin", "an_twin"):
    t = (Uu.group_by(relc, "aempty", "pb").agg(pl.len().alias("n"), pl.col("label").sum().alias("n_true"), pl.col("label").mean().alias("prec"))
         .sort(relc, "aempty", "pb"))
    print(f"--- by {relc} x aempty x p-bin:\n", t)
print("label rate by n_sel_same_src==0 (copy-count prior):",
      Uu.group_by((pl.col("n_sel_same_src") == 0).alias("no_sel_same_src"), "pb").agg(pl.len(), pl.col("label").mean()).sort("no_sel_same_src", "pb").rows())

# rule simulation ---------------------------------------------------------------------------------------
base_pred = links


def score(pred, fset):
    ids = fold.filter(pl.col("fold").is_in(fset)).select("s1")
    return metric.macro_f05(pred, gt, ids)


rules = {
    "addr_twin_src": pl.col("addr_twin_src"),
    "name_twin": pl.col("name_twin"),
    "name_twin_empty": pl.col("name_twin") & pl.col("aempty"),
    "nn_twin": pl.col("nn_twin"),
    "addr_or_name_twin": pl.col("addr_twin_src") | pl.col("name_twin"),
    "nrat>=95": pl.col("max_nrat") >= 95,
    "empty_argmax": pl.col("aempty") & pl.col("is_argmax"),
    "empty_argmax_nn": pl.col("aempty") & pl.col("is_argmax") & pl.col("nn_twin"),
    "addr_twin_argmax": pl.col("addr_twin_src") & pl.col("is_argmax"),
}
TUNE, EVAL = [0, 1, 2], [3, 4]
b_t, b_e = score(base_pred, TUNE), score(base_pred, EVAL)
print(f"\nbase: tune {b_t:.5f} eval {b_e:.5f}")
for name, e in rules.items():
    best = None
    for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        add = Uu.filter(e & (pl.col("p") >= t)).sort("p", descending=True).unique("cand", keep="first").select("s1", pl.col("cand").alias("mid"))
        pred = pl.concat([base_pred, add])
        st = score(pred, TUNE)
        if best is None or st > best[1]:
            best = (t, st, add.height)
    add = Uu.filter(e & (pl.col("p") >= best[0])).sort("p", descending=True).unique("cand", keep="first").select("s1", pl.col("cand").alias("mid"))
    se = score(pl.concat([base_pred, add]), EVAL)
    print(f"ADD rule {name:<18} best t={best[0]:<5} tune {best[1]:.5f} ({best[1] - b_t:+.5f})  eval {se:.5f} ({se - b_e:+.5f})  adds {best[2]:,}")

# FP analysis ------------------------------------------------------------------------------------------
Sel = R.filter(pl.col("sel"))
FP = Sel.filter(~pl.col("label"))
owner = gt_all.rename({"mid": "cand", "s1": "owner"})
FP = FP.join(owner, on="cand", how="left")
FP = FP.join(s1_ids.with_columns(pl.lit(True).alias("owner_in_oof")).rename({"s1": "owner"}), on="owner", how="left").with_columns(
    pl.col("owner_in_oof").fill_null(False))
print(f"\nFP links: {FP.height:,}; distractor (no owner) {FP['owner'].is_null().mean():.4f}; owner is another S1 {FP['owner'].is_not_null().mean():.4f} "
      f"(owner inside OOF sample {FP['owner_in_oof'].mean():.4f}); FP empty address {FP['aempty'].mean():.4f}")
# is the FP an address twin (same source, same raw address) of another copy of its TRUE owner?
pool_attr = []
for c in ("US", "India"):
    pool_attr.append(pl.scan_parquet(f"{NORM}/train_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), "business_address", "business_name").collect())
pool_attr = pl.concat(pool_attr).with_columns(
    pl.col("cand").str.slice(0, 2).alias("src"),
    pl.col("business_address").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("acf"),
    pl.col("business_name").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("ncf")).drop("business_address", "business_name")
own_copies = owner.join(pool_attr, on="cand").rename({"cand": "oc", "acf": "acf_o", "src": "src_o", "ncf": "ncf_o"})
fo = FP.filter(pl.col("owner").is_not_null()).select("s1", "cand", "owner", "src", "acf", "ncf").join(own_copies, on="owner").filter(pl.col("oc") != pl.col("cand"))
fo = fo.group_by("s1", "cand").agg(((pl.col("acf") != "") & (pl.col("acf") == pl.col("acf_o")) & (pl.col("src") == pl.col("src_o"))).any().alias("twin_of_owner_copy"),
                                   (pl.col("ncf") == pl.col("ncf_o")).any().alias("name_twin_of_owner_copy"))
print("FP owned by another S1: share that is an address twin (same src) / name twin of another copy of its TRUE owner:",
      round(float(fo["twin_of_owner_copy"].mean()), 4), round(float(fo["name_twin_of_owner_copy"].mean()), 4), "n", fo.height)
# does the FP have an address twin among OUR OTHER selected links of the same S1? (sibling support)
sp = FP.select("s1", "cand", "src", "acf", "ncf").join(S, on="s1").filter(pl.col("cand") != pl.col("cand_s"))
sp = sp.group_by("s1", "cand").agg(((pl.col("acf") != "") & (pl.col("acf") == pl.col("acf_s")) & (pl.col("src") == pl.col("src_s"))).any().alias("sib_addr_twin"),
                                   (pl.col("ncf") == pl.col("ncf_s")).any().alias("sib_name_twin"))
TPl = Sel.filter(pl.col("label")).select("s1", "cand", "src", "acf", "ncf").join(S, on="s1").filter(pl.col("cand") != pl.col("cand_s"))
TPl = TPl.group_by("s1", "cand").agg(((pl.col("acf") != "") & (pl.col("acf") == pl.col("acf_s")) & (pl.col("src") == pl.col("src_s"))).any().alias("sib_addr_twin"),
                                     (pl.col("ncf") == pl.col("ncf_s")).any().alias("sib_name_twin"))
print("sibling support among selected links: FP share with same-src address twin / name twin:", round(float(sp["sib_addr_twin"].mean()), 4),
      round(float(sp["sib_name_twin"].mean()), 4), "| TP share:", round(float(TPl["sib_addr_twin"].mean()), 4), round(float(TPl["sib_name_twin"].mean()), 4))
# REMOVE rule: selected link whose same-src address twin is selected for ANOTHER S1 and that has no same-src twin in its own S1
tw = Sel.filter(~pl.col("aempty")).select("s1", "cand", "src", "acf", "label", "p")
grp = tw.group_by("src", "acf").agg(pl.col("s1").n_unique().alias("n_s1"))
conf = tw.join(grp, on=["src", "acf"]).filter(pl.col("n_s1") > 1)
own_tw = tw.group_by("s1", "src", "acf").agg(pl.len().alias("n_own"))
conf = conf.join(own_tw, on=["s1", "src", "acf"]).with_columns((pl.col("n_own") == 1).alias("orphan"))
print("selected links whose same-src address string is ALSO selected under another S1:", conf.height,
      "label rate:", round(float(conf["label"].mean()), 4) if conf.height else None,
      "| orphan (no own twin) label rate:", round(float(conf.filter(pl.col("orphan"))["label"].mean()), 4) if conf.filter(pl.col("orphan")).height else None,
      "n", conf.filter(pl.col("orphan")).height)
# remove orphans whose twin-S1 has higher p on the twin
if conf.height:
    rem = conf.filter(pl.col("orphan"))
    for t in (0.5, 0.7, 0.9, 1.01):
        r2 = rem.filter(pl.col("p") < t).select("s1", pl.col("cand").alias("mid"))
        pred = base_pred.join(r2, on=["s1", "mid"], how="anti")
        print(f"REMOVE orphan-twin links with p<{t}: removes {r2.height:,}; tune {score(pred, TUNE) - b_t:+.5f} eval {score(pred, EVAL) - b_e:+.5f}")

# empty-address FN profile -----------------------------------------------------------------------------
fe = FN.filter(pl.col("aempty"))
print(f"\nempty-address FN: {fe.height:,}; p quantiles {np.quantile(fe['p'].to_numpy(), [0.1, 0.5, 0.9]) if fe.height else None}; "
      f"name_twin {fe['name_twin'].mean():.4f}; nn_twin {fe['nn_twin'].mean():.4f}; claimed by another S1 {fe['claimed'].mean():.4f}")
# how many OOF S1 score this candidate > 0.05 (competition)?
comp = oof.filter(pl.col("p") >= 0.05).group_by("cand").agg(pl.len().alias("n_s1_p05"))
fe = fe.join(comp, on="cand", how="left").with_columns(pl.col("n_s1_p05").fill_null(0))
print("empty-address FN: #OOF S1 with p>=0.05 for the same candidate:", fe.group_by(pl.col("n_s1_p05").clip(0, 5)).agg(pl.len()).sort("n_s1_p05").rows())
print("empty-address FN examples:", fe.sort("p", descending=True).head(8).select("s1", "cand", "p", "ncf", "name_twin", "n_sel").rows())
ex = fe.head(6)["s1"].to_list()
s1n = []
for c in ("US", "India"):
    s1n.append(pl.scan_parquet(f"{NORM}/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1"), "business_name", "business_address")
               .filter(pl.col("s1").is_in(ex)).collect())
print("their S1:", pl.concat(s1n).rows())
U.select("s1", "cand", "label", "p", "claimed", "aempty", "addr_twin_src", "name_twin", "nn_twin", "max_nrat", "n_sel_same_src", "n_sel", "fold").write_parquet(f"{OUT}/c_unselected.parquet")
log("done")
