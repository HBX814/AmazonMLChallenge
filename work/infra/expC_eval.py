# -*- coding: utf-8 -*-
"""expC_eval.py -- compare two candidate frames (base vs new blocking) for one split/country.

  * pairs, cands/S1, added / removed pairs; added pairs by producing pass (akey / stname / both / neither = legacy row
    re-admitted by the reserve)
  * train: precision of the added pairs (true share), pair recall / oracle F0.5 before and after, recall gain split by
    candidate address-empty / same state / cross state
  * label-free miss proxies (all S1, not a sample):
      E3  (diag_fr_e) pool records with the same diag address key (base house number | street words | city) and
          name_core token_set >= 80 that are NOT candidates, per S1 (train: true share among them)
      ST  pool records on the same street (street words | city, any / no house number) with EQUAL name_core that are
          NOT candidates, per S1
Writes /vol/exp/expC/eval_<split>_<country>_<tag>.json (+ .md with France examples).
"""
import argparse
import json
import re
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import blocking as BL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--country", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--tag", default="new")
a = ap.parse_args()
T0 = time.time()
OUT = "/vol/exp/expC"
STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "d", "l", "near", "opp", "unit", "floor", "box", "city",
        "county", "town", "township", "village", "no", "hno", "dno", "plot", "flat", "shop", "apt", "appt", "appartement",
        "etage", "bat", "pmb", "suite", "cdp", "twp"}
NUMRE = re.compile(r"\d")


def log(*x):
    print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)


def diag_keys(df):
    """diag_fr_e akey (base hn | street words | city) and street key (street words | city)."""
    ak, sk = [], []
    for adr, hn, city, state in zip(df["addr_norm"].to_list(), df["house_nums"].to_list(), df["city_key"].to_list(),
                                    df["state_key"].to_list()):
        b = None
        for x in hn or []:
            m = re.match(r"\D*(\d+)", x)
            if m:
                b = m.group(1).lstrip("0") or "0"
                break
        drop = set((city or "").split()) | set((state or "").split())
        sw = sorted({w for w in (adr or "").split() if len(w) >= 2 and not NUMRE.search(w) and w not in drop and w not in STOP})
        s = " ".join(sw)
        ak.append(f"{b}|{s}|{city}" if b is not None and sw else "")
        sk.append(f"{s}|{city}" if sw and city else "")
    return df.select("entity_id", "name_core").with_columns(pl.Series("akey", ak), pl.Series("skey", sk))


COLS = ["entity_id", "name_core", "addr_norm", "house_nums", "city_key", "state_key"]
s1 = pl.read_parquet(f"/vol/work_v3/norm/{a.split}_{a.country}_s1.parquet", columns=COLS)
pool = pl.read_parquet(f"/vol/work_v3/norm/{a.split}_{a.country}_pool.parquet", columns=COLS + ["business_name"])
n_s1 = s1.height
base = pl.read_parquet(a.base, columns=["s1", "cand"])
newc = pl.read_parquet(a.new)
keepc = [c for c in ("p_key_akey", "p_key_stname", "p_key_hit", "n_passes") if c in newc.columns]
newc = newc.select(["s1", "cand"] + keepc)
log("loaded", n_s1, pool.height, base.height, newc.height)
rep = {"split": a.split, "country": a.country, "tag": a.tag, "n_s1": n_s1, "pairs_base": base.height, "pairs_new": newc.height,
       "cands_per_s1_base": base.height / n_s1, "cands_per_s1_new": newc.height / n_s1}
added = newc.join(base, on=["s1", "cand"], how="anti")
removed = base.join(newc.select("s1", "cand"), on=["s1", "cand"], how="anti")
rep["added"], rep["removed"] = added.height, removed.height
if "p_key_akey" in added.columns:
    ak, st = added["p_key_akey"].fill_null(False), added["p_key_stname"].fill_null(False)
    rep["added_by_pass"] = {"akey_only": int((ak & ~st).sum()), "stname_only": int((~ak & st).sum()), "both": int((ak & st).sum()),
                            "neither": int((~ak & ~st).sum())}
    rep["new_rows_akey"] = int(newc["p_key_akey"].sum())
    rep["new_rows_stname"] = int(newc["p_key_stname"].sum())
per = newc.group_by("s1").len()
rep["new_cands_per_s1_max"] = int(per["len"].max())
rep["new_s1_over_60"] = int((per["len"] > 60).sum())
log("added", added.height, "removed", removed.height)

geo = lambda df, p: df.select(pl.col("entity_id").alias(p), pl.col("state_key").alias(f"st_{p}"), pl.col("addr_norm").alias(f"ad_{p}"))
if a.split == "train":
    gt = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"]).join(
        s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
    lab = gt.select("s1", pl.col("mid").alias("cand"), pl.lit(1, pl.Int8).alias("label"))
    ev = s1.select("entity_id")
    rb = BL.candidate_recall(base, gt, ev["entity_id"])
    rn = BL.candidate_recall(newc.select("s1", "cand"), gt, ev["entity_id"])
    rep["recall_base"], rep["recall_new"] = rb, rn
    A = added.join(lab, on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
    rep["added_true"] = int(A["label"].sum())
    rep["added_precision"] = float(A["label"].mean()) if A.height else float("nan")
    R = removed.join(lab, on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
    rep["removed_true"] = int(R["label"].sum())
    A = A.join(geo(s1, "s1"), on="s1", how="left").join(geo(pool, "cand"), on="cand", how="left")
    cat = (pl.when((pl.col("ad_cand").fill_null("") == "") | (pl.col("ad_s1").fill_null("") == "")).then(pl.lit("addr_empty"))
             .when((pl.col("st_s1") == pl.col("st_cand")) & (pl.col("st_s1") != "")).then(pl.lit("same_state"))
             .otherwise(pl.lit("cross_state")))
    A = A.with_columns(cat.alias("cat"))
    rep["added_by_cat"] = {r["cat"]: {"pairs": r["n"], "true": r["t"], "precision": r["t"] / r["n"],
                                      "recall_gain": r["t"] / gt.height}
                           for r in A.group_by("cat").agg(pl.len().alias("n"), pl.col("label").sum().alias("t")).iter_rows(named=True)}
    if "p_key_akey" in A.columns:
        rep["added_precision_by_pass"] = {
            nm: {"pairs": int(X.height), "true": int(X["label"].sum()), "precision": float(X["label"].mean()) if X.height else None}
            for nm, X in (("akey_only", A.filter(pl.col("p_key_akey") & ~pl.col("p_key_stname"))),
                          ("stname_only", A.filter(~pl.col("p_key_akey") & pl.col("p_key_stname"))),
                          ("both", A.filter(pl.col("p_key_akey") & pl.col("p_key_stname"))),
                          ("neither", A.filter(~pl.col("p_key_akey") & ~pl.col("p_key_stname"))))}
    # missed true links (after) by category, for context
    miss = gt.select("s1", pl.col("mid").alias("cand")).join(newc.select("s1", "cand"), on=["s1", "cand"], how="anti") \
        .join(geo(s1, "s1"), on="s1", how="left").join(geo(pool, "cand"), on="cand", how="left").with_columns(cat.alias("cat"))
    rep["missed_after_by_cat"] = {r["cat"]: r["n"] for r in miss.group_by("cat").len("n").iter_rows(named=True)}
    log("train metrics done", rep["added_true"], rep["added_precision"])

# ---- label-free miss proxies -----------------------------------------------------------------------------
ks, kp = diag_keys(s1), diag_keys(pool)
log("diag keys built")
S = ks.rename({"entity_id": "s1", "name_core": "n1"})
P = kp.rename({"entity_id": "cand", "name_core": "n2"})
J = S.filter(pl.col("akey") != "").join(P.filter(pl.col("akey") != "").select("cand", "akey", "n2"), on="akey")
J = J.with_columns(pl.Series("ts", process.cpdist(J["n1"].fill_null("").to_list(), J["n2"].fill_null("").to_list(),
                                                  scorer=fuzz.token_set_ratio, workers=-1)))
J = J.filter(pl.col("ts") >= 80).select("s1", "cand")
K = S.filter(pl.col("skey") != "").join(P.filter(pl.col("skey") != "").select("cand", "skey", "n2"), on="skey") \
    .filter((pl.col("n1") == pl.col("n2")) & (pl.col("n1") != "")).select("s1", "cand")
if a.split == "train":
    J = J.join(lab, on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
    K = K.join(lab, on=["s1", "cand"], how="left").with_columns(pl.col("label").fill_null(0))
for nm, X in (("E3_akey_ts80", J), ("ST_street_core_equal", K)):
    d = {"records_per_s1": X.height / n_s1}
    for tag, C in (("base", base), ("new", newc.select("s1", "cand"))):
        o = X.join(C, on=["s1", "cand"], how="anti")
        d[f"out_per_s1_{tag}"] = o.height / n_s1
        d[f"out_{tag}"] = o.height
        if "label" in o.columns:
            d[f"out_true_{tag}"] = int(o["label"].sum())
            d[f"out_true_share_{tag}"] = float(o["label"].mean()) if o.height else None
    rep[nm] = d
    log(nm, d)
md = [f"# expC eval {a.split}/{a.country} ({a.tag})", "", "```", json.dumps(rep, indent=1, default=str), "```", ""]
if a.split == "test":
    nm1 = s1.select(pl.col("entity_id").alias("s1"), pl.col("name_core").alias("n1"), pl.col("addr_norm").alias("a1"))
    nm2 = pool.select(pl.col("entity_id").alias("cand"), pl.col("business_name").alias("raw2"), pl.col("addr_norm").alias("a2"))
    for nm, X in (("E3 still OUT after", J), ("ST still OUT after", K)):
        o = X.join(newc.select("s1", "cand"), on=["s1", "cand"], how="anti").sort(["s1", "cand"]).head(25) \
            .join(nm1, on="s1", how="left").join(nm2, on="cand", how="left")
        md.append(f"## {nm} (first 25 by id)")
        md += [f"- {r['s1']} `{r['n1']}` [{r['a1']}] || {r['cand']} `{r['raw2']}` [{r['a2']}]" for r in o.iter_rows(named=True)]
        md.append("")
    ad = added.sort(["s1", "cand"]).head(30).join(nm1, on="s1", how="left").join(nm2, on="cand", how="left")
    md.append("## added pairs (first 30 by id)")
    md += [f"- {r['s1']} `{r['n1']}` [{r['a1']}] || {r['cand']} `{r['raw2']}` [{r['a2']}] akey={r.get('p_key_akey')} "
           f"stname={r.get('p_key_stname')}" for r in ad.iter_rows(named=True)]
with open(f"{OUT}/eval_{a.split}_{a.country}_{a.tag}.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(rep, f, indent=1, default=str)
with open(f"{OUT}/eval_{a.split}_{a.country}_{a.tag}.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(md) + "\n")
print("EVAL", json.dumps(rep, default=str), flush=True)
log("done")
