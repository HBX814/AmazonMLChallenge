# -*- coding: utf-8 -*-
"""diag_fr_g.py -- covariate-shift (importance-reweighted) estimate of per-country test F from train OOF.

S1-level label-free covariates:
  c_name : # S1 sharing the same name_key AND city_key (1 / 2 / 3+)
  c_addr : # S1 sharing the same address key (base hn | street words | city)  (1 / 2 / 3+)
  c_unc  : # candidates with 0.05 <= p < 0.95                                (0 / 1 / 2 / 3+)
  c_emp  : predicted list empty (yes / no)
OOF (train US+India, true F per S1) is binned on the same covariates; the test estimate is
  F_hat(country) = sum_cell share_test(cell) * F_oof(cell)       (cells with < 30 OOF S1 fall back to (c_unc, c_emp))
Validation: the same estimator applied to test US / India should land near their OOF F.
Writes /vol/diag/fr/reweight.md
"""
import re
import sys
import time

import polars as pl

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide, metric  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
LAM, EB = 0.05303060038344832, 2.0
T0 = time.time()
MD = []
STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "d", "l", "near", "opp", "unit", "floor", "box", "city",
        "county", "town", "township", "village", "no", "hno", "dno", "plot", "flat", "shop", "apt", "appt", "appartement",
        "etage", "bat", "pmb", "suite", "cdp", "twp"}
NUMRE = re.compile(r"\d")


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def md(*lines):
    MD.extend(lines)


def table(header, rows):
    md("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
    md("")


def akeys(df):
    out = []
    for a, hn, city, state in zip(df["addr_norm"].to_list(), df["house_nums"].to_list(), df["city_key"].to_list(), df["state_key"].to_list()):
        b = None
        for x in hn or []:
            m = re.match(r"\D*(\d+)", x)
            if m:
                b = m.group(1).lstrip("0") or "0"
                break
        drop = set((city or "").split()) | set((state or "").split())
        sw = sorted({w for w in (a or "").split() if len(w) >= 2 and not NUMRE.search(w) and w not in drop and w not in STOP})
        out.append(f"{b}|{' '.join(sw)}|{city}" if b is not None and sw else "")
    return out


def covariates(split, country, scored, links):
    s1 = pl.read_parquet(f"{V3}/norm/{split}_{country}_s1.parquet", columns=["entity_id", "name_key", "city_key", "addr_norm", "house_nums", "state_key"])
    s1 = s1.with_columns(pl.Series("akey", akeys(s1)))
    s1 = s1.with_columns(
        pl.when(pl.col("name_key") != "").then(pl.len().over(["name_key", "city_key"])).otherwise(1).alias("n_name"),
        pl.when(pl.col("akey") != "").then(pl.len().over("akey")).otherwise(1).alias("n_addr"))
    unc = scored.filter((pl.col("p") >= 0.05) & (pl.col("p") < 0.95)).group_by("s1").agg(pl.len().alias("n_unc"))
    m = links.group_by("s1").agg(pl.len().alias("m"))
    cv = s1.select(pl.col("entity_id").alias("s1"), "n_name", "n_addr").join(unc, on="s1", how="left").join(m, on="s1", how="left") \
        .with_columns(pl.col("n_unc").fill_null(0), pl.col("m").fill_null(0))
    return cv.with_columns(
        pl.when(pl.col("n_name") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_name").cast(pl.Utf8)).alias("c_name"),
        pl.when(pl.col("n_addr") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_addr").cast(pl.Utf8)).alias("c_addr"),
        pl.when(pl.col("n_unc") >= 3).then(pl.lit("3+")).otherwise(pl.col("n_unc").cast(pl.Utf8)).alias("c_unc"),
        pl.when(pl.col("m") == 0).then(pl.lit("empty")).otherwise(pl.lit("list")).alias("c_emp"))


CELL = ["c_name", "c_addr", "c_unc", "c_emp"]
FALL = ["c_unc", "c_emp"]
oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "p"])
gt = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"])
tr = []
for c in ("US", "India"):
    ids = pl.scan_parquet(f"{V3}/norm/train_{c}_s1.parquet").select(pl.col("entity_id").alias("s1")).collect()
    sc = oof.join(ids, on="s1", how="semi")
    links = decide.select_links(sc, "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
    per = metric.per_s1_scores(links, gt, sc["s1"].unique())
    cv = covariates("train", c, sc, links).join(per.select("s1", "f"), on="s1", how="inner").with_columns(pl.lit(c).alias("country"))
    tr.append(cv)
    log("train", c, cv.height, cv["f"].mean())
TR = pl.concat(tr)
cell_f = TR.group_by(CELL).agg(pl.col("f").mean().alias("f_cell"), pl.len().alias("n_cell"))
fall_f = TR.group_by(FALL).agg(pl.col("f").mean().alias("f_fall"))

md("# Covariate-reweighted estimate of test F from train OOF", "")
md("## marginal distributions (share of S1) and OOF F per level", "")
te = {}
for c in ("France", "US", "India"):
    sc = pl.read_parquet(f"{V3}/pred/test_{c}_scored.parquet")
    links = pl.read_parquet(f"{V3}/pred/test_{c}_links.parquet")
    te[c] = covariates("test", c, sc, links)
    del sc
    log("test", c, te[c].height)
for cov in CELL:
    levels = sorted(set(TR[cov].to_list()) | set().union(*[set(te[c][cov].to_list()) for c in te]))
    rows = []
    for lv in levels:
        r = [cov, lv]
        for c in ("US", "India"):
            x = TR.filter((pl.col("country") == c) & (pl.col(cov) == lv))
            r += [x.height / TR.filter(pl.col("country") == c).height, float(x["f"].mean()) if x.height else float("nan")]
        for c in ("US", "India", "France"):
            r.append(float((te[c][cov] == lv).mean()))
        rows.append(r)
    table(["covariate", "level", "OOF US share", "OOF US F", "OOF IN share", "OOF IN F", "test US share", "test IN share", "test FR share"], rows)

md("## reweighted estimate", "", "cells = (c_name, c_addr, c_unc, c_emp); OOF cell F pooled over US+India; cells with < 30 OOF S1 fall back to (c_unc, c_emp).", "")
rows = []
for name, frame in [("OOF US (self-check)", TR.filter(pl.col("country") == "US")), ("OOF India (self-check)", TR.filter(pl.col("country") == "India"))] + \
                   [(f"test {c}", te[c]) for c in ("US", "India", "France")]:
    w = frame.group_by(CELL).agg(pl.len().alias("n")).join(cell_f, on=CELL, how="left").join(fall_f, on=FALL, how="left")
    w = w.with_columns(pl.when(pl.col("n_cell").fill_null(0) >= 30).then(pl.col("f_cell")).otherwise(pl.col("f_fall")).alias("f_use"))
    est = float((w["n"] * w["f_use"].fill_null(0.98)).sum() / w["n"].sum())
    w2 = frame.group_by(FALL).agg(pl.len().alias("n")).join(fall_f, on=FALL, how="left")
    est2 = float((w2["n"] * w2["f_fall"].fill_null(0.98)).sum() / w2["n"].sum())
    true = float(frame["f"].mean()) if "f" in frame.columns else float("nan")
    rows.append((name, frame.height, est, est2, true))
table(["population", "n_s1", "F_hat full cells", "F_hat (c_unc,c_emp) only", "true F"], rows)
# cells contributing most to the France gap vs pooled OOF
tot_fr = te["France"].height
tot_tr = TR.height
cfr = te["France"].group_by(CELL).agg((pl.len() / tot_fr).alias("fr_share"))
ctr = TR.group_by(CELL).agg((pl.len() / tot_tr).alias("tr_share"))
g = cfr.join(ctr, on=CELL, how="full", coalesce=True).join(cell_f, on=CELL, how="left").fill_null(0.0)
g = g.with_columns(((pl.col("fr_share") - pl.col("tr_share")) * (1 - pl.col("f_cell"))).alias("loss_contrib")).sort("loss_contrib", descending=True)
md("## cells driving the France estimate (loss_contrib = (FR share - OOF share) x (1 - OOF cell F))", "")
table(CELL + ["FR share", "OOF share", "OOF cell F", "n OOF", "loss_contrib"],
      [(r["c_name"], r["c_addr"], r["c_unc"], r["c_emp"], r["fr_share"], r["tr_share"], r["f_cell"], int(r["n_cell"]), r["loss_contrib"]) for r in g.head(15).iter_rows(named=True)])
with open(f"{OUT}/reweight.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("done")
