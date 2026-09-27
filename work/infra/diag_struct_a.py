# -*- coding: utf-8 -*-
"""diag_struct_a.py -- dataset-generation structure, part A (cheap, polars only):
copies per S1 (joint S2/S3), row-order and entity-id ARTIFACT checks on train (with GT) and test (with v3 links).
Writes compact tables to /vol/diag/struct/. Train labels only for measurement.
    python infra/diag_struct_a.py
"""
import os
import time

import numpy as np
import polars as pl

OUT = "/vol/diag/struct"
os.makedirs(OUT, exist_ok=True)
C = "/vol/work/cache"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def src(split, k):
    return (pl.scan_parquet(f"{C}/{split}_source{k}.parquet")
            .with_row_index("row")
            .select("row", "entity_id", "country",
                    pl.col("entity_id").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("num"),
                    pl.col("entity_id").str.extract(r"(\d+)$", 1).str.len_chars().alias("ndig"))
            .collect())


def spearman(a, b):
    ra = pl.Series(a).rank().to_numpy()
    rb = pl.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)

# ------------------------------------------------------------------------------------------ train
s1 = src("train", 1)
s2 = src("train", 2)
s3 = src("train", 3)
gt = pl.read_parquet(f"{C}/train_gt_long.parquet")
log("loaded", s1.height, s2.height, s3.height, gt.height)
n1, n2, n3 = s1.height, s2.height, s3.height

# 1. joint copies distribution ------------------------------------------------------------------
cnt = (gt.with_columns(pl.col("mid").str.slice(0, 2).alias("src"))
       .group_by("s1").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3")))
allc = (s1.select(pl.col("entity_id").alias("s1"), "country").join(cnt, on="s1", how="left")
        .with_columns(pl.col("n2").fill_null(0), pl.col("n3").fill_null(0)))
joint = allc.group_by("n2", "n3").agg(pl.len().alias("n")).with_columns((pl.col("n") / allc.height).alias("share")).sort("n2", "n3")
joint.write_csv(f"{OUT}/a_joint_n2_n3.csv")
log("joint (n2,n3) top 30:\n", joint.sort("n", descending=True).head(30))
print("marginal n2:", allc.group_by("n2").agg(pl.len()).sort("n2").rows())
print("marginal n3:", allc.group_by("n3").agg(pl.len()).sort("n3").rows())
print("by country mean n2/n3:", allc.group_by("country").agg(pl.col("n2").mean(), pl.col("n3").mean(),
                                                                 ((pl.col("n2") + pl.col("n3")) == 0).mean().alias("single")).rows())
# independence test: P(n2=a, n3=b) vs P(n2=a)P(n3=b) among matched S1
m = allc.filter((pl.col("n2") + pl.col("n3")) > 0)
p2 = m.group_by("n2").agg((pl.len() / m.height).alias("p2"))
p3 = m.group_by("n3").agg((pl.len() / m.height).alias("p3"))
jj = (m.group_by("n2", "n3").agg((pl.len() / m.height).alias("pj")).join(p2, on="n2").join(p3, on="n3")
      .with_columns((pl.col("pj") / (pl.col("p2") * pl.col("p3"))).alias("lift")).sort("pj", descending=True))
print("matched S1: joint vs independent (lift) top 20:\n", jj.head(20))
print("corr(n2,n3) matched:", float(np.corrcoef(m["n2"].to_numpy(), m["n3"].to_numpy())[0, 1]))
tot = (m["n2"] + m["n3"])
print("total copies dist (matched):", m.group_by((pl.col("n2") + pl.col("n3")).alias("k")).agg(pl.len()).sort("k").rows())

# 4. ROW ORDER / ID ARTIFACTS on train -----------------------------------------------------------
pool = pl.concat([s2.with_columns(pl.lit("S2").alias("src")), s3.with_columns(pl.lit("S3").alias("src"))])
pool = pool.join(gt.rename({"mid": "entity_id"}), on="entity_id", how="left").with_columns(pl.col("s1").is_not_null().alias("linked"))
n_of = {"S2": n2, "S3": n3}
pool = pool.with_columns((pl.col("row") / pl.when(pl.col("src") == "S2").then(n2).otherwise(n3)).alias("rpos"))
dec = (pool.with_columns((pl.col("rpos") * 10).floor().cast(pl.Int32).alias("dec"))
       .group_by("src", "dec").agg(pl.col("linked").mean().alias("linked_rate"), pl.len().alias("n")).sort("src", "dec"))
print("ARTIFACT? linked rate by row decile:\n", dec)
dec.write_csv(f"{OUT}/a_linked_by_row_decile.csv")
s1r = s1.join(allc.select(pl.col("s1").alias("entity_id"), "n2", "n3"), on="entity_id").with_columns(((pl.col("n2") + pl.col("n3")) == 0).alias("single"))
d1 = (s1r.with_columns((pl.col("row") / n1 * 10).floor().cast(pl.Int32).alias("dec"))
      .group_by("dec").agg(pl.col("single").mean().alias("single_rate"), (pl.col("n2") + pl.col("n3")).mean().alias("mean_k"),
                           (pl.col("country") == "US").mean().alias("us_share")).sort("dec"))
print("ARTIFACT? S1 singleton rate / mean k by row decile:\n", d1)
# rank correlation S1 row vs linked row
lk = (gt.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("row").alias("r1"), pl.col("num").alias("id1"), pl.col("ndig").alias("nd1")), on="s1")
      .join(pool.select(pl.col("entity_id").alias("mid"), "src", pl.col("row").alias("r2"), pl.col("num").alias("id2"), pl.col("ndig").alias("nd2")), on="mid"))
for sname in ("S2", "S3"):
    x = lk.filter(pl.col("src") == sname).sample(n=min(2_000_000, lk.height), seed=1)
    print(f"ARTIFACT? spearman(S1 row, {sname} row) = {spearman(x['r1'], x['r2']):+.5f}; spearman(S1 id, {sname} id) = {spearman(x['id1'], x['id2']):+.5f}")
# sibling adjacency within a source file
for sname, n in (("S2", n2), ("S3", n3)):
    g = (lk.filter(pl.col("src") == sname).sort("s1", "r2")
         .with_columns((pl.col("r2") - pl.col("r2").shift(1).over("s1")).alias("gap"),
                       (pl.col("id2") - pl.col("id2").shift(1).over("s1")).abs().alias("idgap"))
         .filter(pl.col("gap").is_not_null()))
    gap = g["gap"].to_numpy()
    # expectation for random placement of 2 records in n rows: gap ~ n/3
    qs = np.quantile(gap, [0.001, 0.01, 0.1, 0.5])
    print(f"ARTIFACT? {sname} sibling consecutive row gaps: n={len(gap):,} quantiles(0.1%,1%,10%,50%)={qs} "
          f"frac gap<=10: {np.mean(gap <= 10):.6f} (random expectation ~{2 * 10 / n:.6f}); median/n={qs[3] / n:.4f}")
    idg = g["idgap"].to_numpy().astype(float)
    print(f"   id gaps quantiles(1%,10%,50%) = {np.quantile(idg, [0.01, 0.1, 0.5])}")
# id digit-length distribution linked vs unlinked
ndt = pool.group_by("src", "linked", "ndig").agg(pl.len().alias("n")).sort("src", "linked", "ndig")
ndt = ndt.with_columns((pl.col("n") / pl.col("n").sum().over("src", "linked")).alias("share"))
print("ARTIFACT? id digit length share by linked:\n", ndt)
ndt.write_csv(f"{OUT}/a_iddigits_linked.csv")
# AUC of numeric id / row position for linked vs unlinked (Mann-Whitney via ranks)
for sname in ("S2", "S3"):
    x = pool.filter(pl.col("src") == sname)
    for col in ("num", "row"):
        r = x[col].rank().to_numpy()
        y = x["linked"].to_numpy()
        npos, nneg = y.sum(), (~y).sum()
        auc = (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        print(f"ARTIFACT? AUC(linked | {sname} {col}) = {auc:.5f}")
    # last digit / mod patterns
    md = x.with_columns((pl.col("num") % 10).alias("d")).group_by("d").agg(pl.col("linked").mean()).sort("d")
    print(f"  {sname} linked rate by last id digit:", [round(v, 4) for v in md["linked"].to_list()])
for col in ("num", "row"):
    r = s1r[col].rank().to_numpy()
    y = s1r["single"].to_numpy()
    npos, nneg = y.sum(), (~y).sum()
    auc = (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    print(f"ARTIFACT? AUC(singleton | S1 {col}) = {auc:.5f}")
print("S1 ndig dist by singleton:", s1r.group_by("single", "ndig").agg(pl.len()).sort("single", "ndig").rows())
# country interleaving in files
for nm, fr in (("S1", s1), ("S2", s2), ("S3", s3)):
    c = fr["country"].to_numpy()
    runs = 1 + int(np.sum(c[1:] != c[:-1]))
    print(f"{nm}: country runs {runs:,} over {len(c):,} rows (random interleave ~{2 * len(c) * np.mean(c == 'US') * (1 - np.mean(c == 'US')):,.0f})")
# is the linked S1 of consecutive pool rows correlated (pool sorted by entity?)
for sname in ("S2", "S3"):
    x = pool.filter(pl.col("src") == sname).sort("row")
    same = (x["s1"] == x["s1"].shift(1)).fill_null(False).sum()
    print(f"ARTIFACT? {sname}: consecutive rows with same linked S1: {same:,} of {x.height:,}")

# ------------------------------------------------------------------------------------------ test
log("test")
t1 = src("test", 1)
t2 = src("test", 2)
t3 = src("test", 3)
tl = pl.concat([pl.read_parquet(f"/vol/work_v3/pred/test_{c}_links.parquet") for c in ("US", "India", "France")])
tp = pl.concat([t2.with_columns(pl.lit("S2").alias("src")), t3.with_columns(pl.lit("S3").alias("src"))])
tp = tp.join(tl.rename({"mid": "entity_id"}), on="entity_id", how="left").with_columns(pl.col("s1").is_not_null().alias("linked"))
tp = tp.with_columns((pl.col("row") / pl.when(pl.col("src") == "S2").then(t2.height).otherwise(t3.height)).alias("rpos"))
tdec = (tp.with_columns((pl.col("rpos") * 10).floor().cast(pl.Int32).alias("dec"))
        .group_by("src", "dec").agg(pl.col("linked").mean().alias("pred_linked_rate")).sort("src", "dec"))
print("TEST predicted-linked rate by row decile:\n", tdec.pivot(on="src", index="dec", values="pred_linked_rate"))
tnd = tp.group_by("src", "linked", "ndig").agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum().over("src", "linked")).alias("share")).sort("src", "linked", "ndig")
print("TEST id digit length share by predicted linked:\n", tnd)
tlk = (tl.join(t1.select(pl.col("entity_id").alias("s1"), pl.col("row").alias("r1"), pl.col("num").alias("id1")), on="s1")
       .join(tp.select(pl.col("entity_id").alias("mid"), "src", pl.col("row").alias("r2"), pl.col("num").alias("id2")), on="mid"))
for sname in ("S2", "S3"):
    x = tlk.filter(pl.col("src") == sname)
    print(f"TEST spearman(S1 row, {sname} row) = {spearman(x['r1'], x['r2']):+.5f}; ids {spearman(x['id1'], x['id2']):+.5f}")
for nm, fr in (("S1", t1), ("S2", t2), ("S3", t3)):
    print(f"TEST {nm} country share by row decile:",
          fr.with_columns((pl.col("row") / fr.height * 10).floor().cast(pl.Int32).alias("dec"))
          .group_by("dec").agg((pl.col("country") == "France").mean().alias("fr")).sort("dec")["fr"].round(3).to_list())
log("done")
