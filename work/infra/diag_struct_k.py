# -*- coding: utf-8 -*-
"""diag_struct_k.py -- anatomy of selected hard-negative-shift links (house number shifted 3-100 or one digit substituted
by 3..9, same name): TRAIN OOF (TP vs FP) vs TEST. p, name equality, whether the S1 also has a selected exact-hn link,
S1 list size; simulate on OOF the rule 'drop a selected HARD-shift link when the S1 has another selected link with the
exact house number' (cost on OOF) and count what it drops on TEST; print test examples.
    python infra/diag_struct_k.py
"""
import os
import re
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.environ.get("DIAG_SRC", "/root/proj/code/business_entity_resolution/src"))
from ber import decide, metric  # noqa: E402

NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
PRED = os.environ.get("DIAG_PRED", "/vol/work_v3/pred")
OOF = os.environ.get("DIAG_OOF", "/vol/work_v3/model/oof.parquet")
GTP = os.environ.get("DIAG_GT", "/vol/work/cache/train_gt_long.parquet")
T0 = time.time()
KW = dict(exclusivity="soft", lam_missing=0.05303060038344832, empty_bias=2.0)
HARD = {"diff_3_10", "diff_11_100", "one_digit_subst_d3", "one_digit_subst_d4", "one_digit_subst_d5", "one_digit_subst_d7", "one_digit_subst_d9"}
SMALL = {"one_digit_subst_d1", "one_digit_subst_d2", "diff_1_2"}
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_width_chars(250)


def hn_rel(h1, h2):
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
        return "digit_dropped"
    if da in db and len(da) < len(db):
        return "digit_added"
    try:
        d = abs(int(da[:12]) - int(db[:12]))
    except ValueError:
        return "other"
    if len(da) == len(db) and sum(x != y for x, y in zip(da, db)) == 1:
        return f"one_digit_subst_d{d}" if d < 10 else "one_digit_subst_big"
    if d <= 2:
        return "diff_1_2"
    if d <= 10:
        return "diff_3_10"
    if d <= 100:
        return "diff_11_100"
    return "diff_100+"


def tag(df, split, c):
    a1 = pl.scan_parquet(f"{NORM}/{split}_{c}_s1.parquet").select(pl.col("entity_id").alias("s1"), pl.col("house_nums").alias("h1"),
                                                                  pl.col("name_norm").alias("nn1"), pl.col("business_name").alias("bn1"),
                                                                  pl.col("business_address").alias("ba1")).join(df.select("s1").unique().lazy(), on="s1", how="semi").collect()
    a2 = pl.scan_parquet(f"{NORM}/{split}_{c}_pool.parquet").select(pl.col("entity_id").alias("cand"), pl.col("house_nums").alias("h2"),
                                                                    pl.col("name_norm").alias("nn2"), pl.col("business_name").alias("bn2"),
                                                                    pl.col("business_address").alias("ba2")).join(df.select("cand").unique().lazy(), on="cand", how="semi").collect()
    x = df.join(a1, on="s1").join(a2, on="cand")
    x = x.with_columns(pl.Series("hrel", [hn_rel(u, v) for u, v in zip(x["h1"].to_list(), x["h2"].to_list())]),
                       (pl.col("nn1") == pl.col("nn2")).alias("name_eq"))
    x = x.with_columns(pl.when(pl.col("hrel").is_in(list(HARD))).then(pl.lit("HARD")).when(pl.col("hrel").is_in(list(SMALL))).then(pl.lit("SMALL"))
                       .otherwise(pl.lit("other")).alias("cls"))
    ex = x.group_by("s1").agg((pl.col("hrel") == "exact").sum().alias("n_exact_sel"), pl.len().alias("m"))
    return x.join(ex, on="s1")


def prof(tag_, x, by=None):
    g = ["cls"] + ([by] if by else [])
    t = (x.filter(pl.col("cls") != "other").group_by(g)
         .agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p"), pl.col("name_eq").mean().alias("name_eq"),
              (pl.col("n_exact_sel") > 0).mean().alias("s1_has_exact_sel"), pl.col("m").mean().alias("mean_list")).sort(g))
    print(f"[K {tag_}]\n", t, flush=True)


oof = pl.read_parquet(OOF, columns=["s1", "cand", "label", "p"])
s1_ids = oof.select("s1").unique()
gt = pl.read_parquet(GTP).join(s1_ids, on="s1", how="semi")
links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", **KW).rename({"mid": "cand"})
F0 = metric.macro_f05(links.rename({"cand": "mid"}), gt, s1_ids)
sel = links.join(oof, on=["s1", "cand"])
cty = pl.concat([pl.scan_parquet(f"{NORM}/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).with_columns(pl.lit(c).alias("country")).collect()
                 for c in ("US", "India")])
X = pl.concat([tag(sel.join(cty.filter(pl.col("country") == c), on="s1").drop("country"), "train", c).with_columns(pl.lit(c).alias("country"))
               for c in ("US", "India")])
for c in ("US", "India"):
    prof(f"OOF {c} selected links by class x label", X.filter(pl.col("country") == c), by="label")
rules = {
    "drop HARD": pl.col("cls") == "HARD",
    "drop HARD if S1 has exact": (pl.col("cls") == "HARD") & (pl.col("n_exact_sel") > 0),
    "drop HARD|SMALL if S1 has exact": pl.col("cls").is_in(["HARD", "SMALL"]) & (pl.col("n_exact_sel") > 0),
    "drop HARD if S1 has exact & p<0.99": (pl.col("cls") == "HARD") & (pl.col("n_exact_sel") > 0) & (pl.col("p") < 0.99),
    "drop HARD|SMALL if exact & p<0.99": pl.col("cls").is_in(["HARD", "SMALL"]) & (pl.col("n_exact_sel") > 0) & (pl.col("p") < 0.99),
}
for name, e in rules.items():
    d = X.filter(e)
    pred = links.join(d.select("s1", "cand"), on=["s1", "cand"], how="anti").rename({"cand": "mid"})
    f = metric.macro_f05(pred, gt, s1_ids)
    by = {c: (d.filter(pl.col("country") == c).height, round(float((d.filter(pl.col("country") == c)["label"] == 0).mean()), 3) if d.filter(pl.col("country") == c).height else None)
          for c in ("US", "India")}
    print(f"[K OOF rule] {name:<38} drops {d.height:,} (per country n, FP-rate {by}); macro F0.5 {F0:.5f} -> {f:.5f} ({f - F0:+.5f})", flush=True)
for c in ("US", "India", "France"):
    sc = pl.read_parquet(f"{PRED}/test_{c}_scored.parquet", columns=["s1", "cand", "p"])
    tl = pl.read_parquet(f"{PRED}/test_{c}_links.parquet").rename({"mid": "cand"}).join(sc, on=["s1", "cand"])
    n = pl.scan_parquet(f"{NORM}/test_{c}_s1.parquet").select(pl.len()).collect().item()
    x = tag(tl, "test", c)
    prof(f"TEST {c} selected links by class", x)
    for name, e in rules.items():
        print(f"[K TEST {c} rule] {name:<38} would drop {x.filter(e).height:,} ({x.filter(e).height / n:.4f}/S1)", flush=True)
    exf = x.filter(pl.col("cls") == "HARD")
    ex = exf.sample(n=min(8, exf.height), seed=2)
    for r in ex.select("s1", "bn1", "ba1", "bn2", "ba2", "hrel", "p").rows():
        print("    ", r)
    del sc
print(f"done {time.time() - T0:.0f}s")
