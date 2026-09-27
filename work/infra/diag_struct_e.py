# -*- coding: utf-8 -*-
"""diag_struct_e.py -- (1) record-level linked prior: P(linked | record formatting attribute) on TRAIN pools and the
attribute shares on TEST pools; (2) hard negatives: pairs (S1, pool record) sharing name_core + state + city, split into
true copy / unlinked distractor / record linked to another S1, by house-number relation; (3) the same relation on the
stage-1 OOF pairs (label rate vs mean p, FP / FN counts) to see whether the model already captures it.
    python infra/diag_struct_e.py
"""
import os
import re
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide, metric  # noqa: E402

OUT = os.environ.get("DIAG_OUT", "/vol/diag/struct")
NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
GTP = os.environ.get("DIAG_GT", "/vol/work/cache/train_gt_long.parquet")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(70)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def attrs(df):
    n = pl.col("business_name")
    a = pl.col("business_address")
    return df.with_columns(
        (a.str.strip_chars() == "").alias("r_addr_empty"),
        a.str.contains(r"(?i)\b(null|none)\b").alias("r_addr_null"),
        ((a == a.str.to_uppercase()) & (a.str.strip_chars() != "")).alias("r_addr_upper"),
        (pl.col("script").fill_null("latin") != "latin").alias("r_name_native"),
        pl.col("name_is_domain").fill_null(False).alias("r_name_domain"),
        n.str.contains(r"(?i)d\.?b\.?a\b").alias("r_name_dba"),
        n.str.contains(r"[\[\]]").alias("r_name_bracket"),
        n.str.contains(r"^\s*[^\w\s]").alias("r_name_junklead"),
        n.str.contains(r"  ").alias("r_name_dblspace"),
        n.str.contains(r"[À-ɏ]").alias("r_name_accent"),
        ((n == n.str.to_uppercase()) & n.str.contains(r"[A-Za-z]")).alias("r_name_upper"),
        ((n == n.str.to_lowercase()) & n.str.contains(r"[A-Za-z]")).alias("r_name_lower"),
        n.str.contains(r"\d{6,}").alias("r_name_phone"),
    )


RCOLS = ["r_addr_empty", "r_addr_null", "r_addr_upper", "r_name_native", "r_name_domain", "r_name_dba", "r_name_bracket", "r_name_junklead",
         "r_name_dblspace", "r_name_accent", "r_name_upper", "r_name_lower", "r_name_phone"]
gt = pl.read_parquet(GTP)

# ------------------------------------------------------------------ (1) record-level prior
for c in ("US", "India"):
    pool = attrs(pl.read_parquet(f"{NORM}/train_{c}_pool.parquet", columns=["entity_id", "business_name", "business_address", "script", "name_is_domain"]))
    pool = pool.join(gt.select(pl.col("mid").alias("entity_id")).with_columns(pl.lit(True).alias("linked")), on="entity_id", how="left").with_columns(
        pl.col("linked").fill_null(False), pl.col("entity_id").str.slice(0, 2).alias("src"))
    rows = []
    for rc in RCOLS:
        for sname in ("S2", "S3"):
            x = pool.filter(pl.col("src") == sname)
            k = x.filter(pl.col(rc))
            rows.append((rc, sname, k.height, round(k.height / x.height, 4), round(float(k["linked"].mean()) if k.height else float("nan"), 4),
                         round(float(x.filter(pl.col("linked"))[rc].mean()), 4), round(float(x.filter(~pl.col("linked"))[rc].mean()), 4)))
    print(f"\n[E1 train {c}] attribute, src, n, share, P(linked|attr), share among linked, share among unlinked  (base P(linked) S2 "
          f"{pool.filter(pl.col('src') == 'S2')['linked'].mean():.4f} S3 {pool.filter(pl.col('src') == 'S3')['linked'].mean():.4f})")
    for r in rows:
        print("   ", r)
    # any 'noise attribute' at all
    anyn = pool.with_columns(pl.any_horizontal([pl.col(x) for x in RCOLS if x not in ("r_addr_upper", "r_name_native")]).alias("anynoise"))
    print(f"[E1 train {c}] record with ANY name/addr noise flag (excl. upper addr, native): share {anyn['anynoise'].mean():.4f}, "
          f"P(linked|any) {anyn.filter(pl.col('anynoise'))['linked'].mean():.4f}, P(linked|none) {anyn.filter(~pl.col('anynoise'))['linked'].mean():.4f}")
for c in ("US", "India", "France"):
    tp = attrs(pl.read_parquet(f"{NORM}/test_{c}_pool.parquet", columns=["entity_id", "business_name", "business_address", "script", "name_is_domain"]))
    print(f"[E1 test {c}] attribute shares:", {rc: round(float(tp[rc].mean()), 4) for rc in RCOLS})
log("E1 done")

# ------------------------------------------------------------------ (2) hard negatives by house-number relation
_num = re.compile(r"\d+")


def hn_rel(h1, h2):
    """relation between two house-number lists (first element of each)."""
    if not h1 or not h2:
        return "one_missing"
    a, b = h1[0], h2[0]
    if a == b:
        return "exact"
    da, db = re.sub(r"\D", "", a), re.sub(r"\D", "", b)
    if da == db and da:
        return "letter_change"
    if not da or not db:
        return "other"
    if db in da and len(db) < len(da):
        return "digit_dropped" if (da.startswith(db) or da.endswith(db)) else "digit_dropped_mid"
    if da in db and len(da) < len(db):
        return "digit_added"
    try:
        d = abs(int(da[:12]) - int(db[:12]))
    except ValueError:
        return "other"
    if len(da) == len(db) and sum(x != y for x, y in zip(da, db)) == 1:
        return f"one_digit_subst_d{min(d, 10) if d < 10 else ('10-99' if d < 100 else '100+')}"
    if d <= 2:
        return "diff_1_2"
    if d <= 10:
        return "diff_3_10"
    if d <= 100:
        return "diff_11_100"
    return "diff_100+"


def nums(a):
    return sorted(_num.findall(a or ""))


def num_rel(a1, a2):
    n1, n2 = nums(a1), nums(a2)
    if not n2:
        return "cand_no_numbers"
    if n1 == n2:
        return "same_multiset"
    s1, s2 = set(n1), set(n2)
    if s2 < s1:
        return "subset"
    if s2 > s1:
        return "superset"
    if len(s1) == len(s2) and len(s1 ^ s2) == 2:
        return "one_changed"
    return "other"


hard = []
for c in ("US", "India"):
    s1 = pl.read_parquet(f"{NORM}/train_{c}_s1.parquet", columns=["entity_id", "business_address", "name_core", "name_norm", "state_key", "city_key", "house_nums"])
    pool = pl.read_parquet(f"{NORM}/train_{c}_pool.parquet", columns=["entity_id", "business_address", "business_name", "name_core", "name_norm", "state_key", "city_key", "house_nums"])
    own = gt.rename({"mid": "cand", "s1": "owner"})
    k1 = s1.filter((pl.col("name_core").str.len_chars() >= 2) & (pl.col("city_key").fill_null("") != "")).select(
        pl.col("entity_id").alias("s1"), "name_core", "state_key", "city_key", pl.col("business_address").alias("a1"), pl.col("house_nums").alias("h1"),
        pl.col("name_norm").alias("nn1"))
    k2 = pool.filter((pl.col("name_core").str.len_chars() >= 2) & (pl.col("city_key").fill_null("") != "")).select(
        pl.col("entity_id").alias("cand"), "name_core", "state_key", "city_key", pl.col("business_address").alias("a2"), pl.col("house_nums").alias("h2"),
        pl.col("name_norm").alias("nn2"), pl.col("business_name").alias("bn2"))
    kk = ["name_core", "state_key", "city_key"]
    sz = k1.group_by(kk).agg(pl.len().alias("n1")).join(k2.group_by(kk).agg(pl.len().alias("n2")), on=kk).with_columns((pl.col("n1") * pl.col("n2")).alias("np"))
    log(f"[E2 {c}] join size {sz['np'].sum():,}; keys with n1*n2>2000: {sz.filter(pl.col('np') > 2000).height:,} ({sz.filter(pl.col('np') > 2000)['np'].sum():,} pairs) -> dropped")
    k1 = k1.join(sz.filter(pl.col("np") <= 2000).select(kk), on=kk, how="semi")
    pr = k1.join(k2, on=kk).join(own, on="cand", how="left")
    pr = pr.with_columns(pl.when(pl.col("owner") == pl.col("s1")).then(pl.lit("true")).when(pl.col("owner").is_null()).then(pl.lit("unlinked"))
                         .otherwise(pl.lit("other_s1")).alias("cls"))
    log(f"[E2 {c}] pairs sharing name_core+state+city: {pr.height:,}", pr.group_by("cls").agg(pl.len()).rows())
    if pr.height > 12_000_000:
        pr = pr.sample(n=12_000_000, seed=0)
    h1l, h2l, a1l, a2l = pr["h1"].to_list(), pr["h2"].to_list(), pr["a1"].to_list(), pr["a2"].to_list()
    pr = pr.with_columns(pl.Series("hrel", [hn_rel(x, y) for x, y in zip(h1l, h2l)]),
                         pl.Series("nrel", [num_rel(x, y) for x, y in zip(a1l, a2l)]),
                         (pl.col("nn1") == pl.col("nn2")).alias("name_norm_eq"))
    t = (pr.group_by("hrel").agg(pl.len().alias("n"), (pl.col("cls") == "true").sum().alias("n_true"), (pl.col("cls") == "unlinked").sum().alias("n_unl"),
                                 (pl.col("cls") == "other_s1").sum().alias("n_other"))
         .with_columns((pl.col("n_true") / pl.col("n")).alias("P_true")).sort("n", descending=True))
    print(f"[E2 {c}] house-number relation (first house number) for same name_core+state+city pairs:\n", t)
    t2 = (pr.group_by("nrel").agg(pl.len().alias("n"), (pl.col("cls") == "true").sum().alias("n_true"), (pl.col("cls") == "unlinked").sum().alias("n_unl"),
                                  (pl.col("cls") == "other_s1").sum().alias("n_other"))
          .with_columns((pl.col("n_true") / pl.col("n")).alias("P_true")).sort("n", descending=True))
    print(f"[E2 {c}] all-number multiset relation:\n", t2)
    # hard negatives per S1: singleton vs matched
    ks = gt.group_by("s1").agg(pl.len().alias("k"))
    hn_s1 = (pr.filter(pl.col("cls") == "unlinked").select("s1").unique().with_columns(pl.lit(True).alias("has_unl_same_name_city")))
    s1k = s1.select(pl.col("entity_id").alias("s1")).join(ks, on="s1", how="left").with_columns(pl.col("k").fill_null(0)).join(hn_s1, on="s1", how="left").with_columns(
        pl.col("has_unl_same_name_city").fill_null(False))
    print(f"[E2 {c}] S1 with an UNLINKED record of same name_core+state+city, by true k:",
          s1k.group_by(pl.col("k").clip(0, 7)).agg(pl.len(), pl.col("has_unl_same_name_city").mean()).sort("k").rows())
    exf = pr.filter((pl.col("cls") == "unlinked") & (pl.col("hrel") != "one_missing"))
    ex = exf.sample(n=min(10, exf.height), seed=1).select("s1", "a1", "bn2", "a2", "hrel").rows()
    print(f"[E2 {c}] unlinked hard-negative examples:")
    for r in ex:
        print("     ", r)
    hard.append(pr.select("s1", "cand", "cls", "hrel", "nrel", "name_norm_eq"))
    del pr, pool, s1
hard = pl.concat(hard)
log("E2 done")

# ------------------------------------------------------------------ (3) the same relations on OOF pairs
oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
s1_ids = oof.select("s1").unique()
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=0.05303060038344832, empty_bias=2.0)
oof = oof.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
R = oof.filter((pl.col("p") >= 0.002) | (pl.col("label") == 1) | pl.col("sel"))
A1, A2 = [], []
for c in ("US", "India"):
    A1.append(pl.read_parquet(f"{NORM}/train_{c}_s1.parquet", columns=["entity_id", "business_address", "house_nums"]).rename(
        {"entity_id": "s1", "business_address": "a1", "house_nums": "h1"}).join(s1_ids, on="s1", how="semi"))
    A2.append(pl.scan_parquet(f"{NORM}/train_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), pl.col("business_address").alias("a2"),
                                                                       pl.col("house_nums").alias("h2")).join(R.select("cand").unique().lazy(), on="cand", how="semi").collect())
R = R.join(pl.concat(A1), on="s1").join(pl.concat(A2), on="cand")
R = R.with_columns(pl.Series("hrel", [hn_rel(x, y) for x, y in zip(R["h1"].to_list(), R["h2"].to_list())]))
t = (R.group_by("hrel").agg(pl.len().alias("n"), pl.col("label").mean().alias("label_rate"), pl.col("p").mean().alias("mean_p"),
                            (pl.col("sel") & (pl.col("label") == 0)).sum().alias("FP"), (~pl.col("sel") & (pl.col("label") == 1)).sum().alias("FN"),
                            (pl.col("sel") & (pl.col("label") == 1)).sum().alias("TP"))
     .sort("n", descending=True))
print("\n[E3] OOF candidate rows (p>=0.002 | true | selected) by house-number relation: label rate vs mean p, FP/FN counts\n", t)
# within-relation calibration for the ambiguous relations
for hr in ("diff_1_2", "diff_3_10", "digit_dropped", "digit_added", "letter_change", "one_digit_subst_d1", "one_digit_subst_d2", "one_digit_subst_d3"):
    x = R.filter(pl.col("hrel") == hr)
    if x.height:
        cb = x.with_columns(pl.col("p").cut([0.1, 0.3, 0.5, 0.7, 0.9]).alias("b")).group_by("b").agg(pl.len(), pl.col("p").mean(), pl.col("label").mean()).sort("b")
        print(f"   [E3] calibration within {hr}:", [(str(r[0]), r[1], round(r[2], 3), round(r[3], 3)) for r in cb.rows()])
log("done")
