# -*- coding: utf-8 -*-
"""diag_fr_a.py -- inspection sheets for test France / US / India (label-free).

For N random test S1 per country (fixed seed): the S1 record, its top-6 candidates by p (+ any selected beyond
top-6) with raw name/address, p, selected flag, the OTHER S1 that owns the candidate (if selected elsewhere), and
up to 4 "outside" pool records (not candidates) sharing a house number with high address similarity or the exact
name core -> possible blocking misses. Writes /vol/diag/fr/inspect_<country>.md and prints schemas / counts.
"""
import os
import sys
import time

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
os.makedirs(OUT, exist_ok=True)
SEED = 20260926
PLAN = [("France", 80), ("US", 40), ("India", 40)]
RAW = ["entity_id", "business_name", "business_address"]
NORMC = ["name_core", "addr_norm", "house_nums", "state_key", "city_key"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def clip(s, n=70):
    s = (s or "").replace("\t", " ").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "~"


for f in (f"{V3}/pred/test_France_scored.parquet", f"{V3}/pred/test_France_links.parquet",
          f"{V3}/norm/test_France_s1.parquet", f"{V3}/norm/test_France_pool.parquet", f"{V3}/feats/test_France.parquet",
          f"{V3}/model/oof.parquet"):
    try:
        print(f, dict(pl.scan_parquet(f).collect_schema()))
    except Exception as e:  # noqa
        print(f, "ERR", e)

for country, n in PLAN:
    s1 = pl.read_parquet(f"{V3}/norm/test_{country}_s1.parquet", columns=RAW + NORMC)
    samp = s1.sample(n, seed=SEED)
    ids = samp["entity_id"].to_list()
    sc = (pl.scan_parquet(f"{V3}/pred/test_{country}_scored.parquet").filter(pl.col("s1").is_in(ids)).collect())
    links = pl.read_parquet(f"{V3}/pred/test_{country}_links.parquet")
    cset = sc["cand"].unique().to_list()
    own = links.filter(pl.col("mid").is_in(cset)).rename({"s1": "owner"})
    own = own.join(s1.select(pl.col("entity_id").alias("owner"), pl.col("business_name").alias("owner_name"),
                             pl.col("business_address").alias("owner_addr")), on="owner", how="left")
    pool_lf = pl.scan_parquet(f"{V3}/norm/test_{country}_pool.parquet").select(RAW + NORMC)
    pool = pool_lf.filter(pl.col("entity_id").is_in(cset)).collect()
    # outside candidates: same house number + same state (or empty) + address similarity, or identical name core
    hn = samp.select(pl.col("entity_id").alias("s1"), "house_nums", pl.col("addr_norm").alias("a1"),
                     pl.col("name_core").alias("n1"), pl.col("state_key").alias("st1")).explode("house_nums")
    hn = hn.filter(pl.col("house_nums").is_not_null() & (pl.col("house_nums") != ""))
    hn_set = hn["house_nums"].unique().to_list()
    ph = (pool_lf.select("entity_id", "house_nums", "addr_norm", "name_core", "state_key").explode("house_nums")
          .filter(pl.col("house_nums").is_in(hn_set)).collect())
    j = hn.join(ph, on="house_nums").filter((pl.col("state_key") == pl.col("st1")) | (pl.col("state_key") == ""))
    j = j.with_columns(pl.Series("asim", [fuzz.token_set_ratio(a, b) for a, b in zip(j["a1"].to_list(), j["addr_norm"].to_list())], dtype=pl.Float64),
                       pl.Series("nsim", [fuzz.token_set_ratio(a, b) for a, b in zip(j["n1"].to_list(), j["name_core"].to_list())], dtype=pl.Float64))
    outs = j.filter((pl.col("asim") >= 85) & (pl.col("nsim") >= 60)).select("s1", "entity_id", "asim", "nsim")
    nm = samp.select(pl.col("entity_id").alias("s1"), pl.col("name_core").alias("nc"))
    nc_set = [x for x in nm["nc"].to_list() if x]
    pn = pool_lf.select("entity_id", "name_core").filter(pl.col("name_core").is_in(nc_set)).collect()
    outs2 = nm.join(pn.rename({"name_core": "nc"}), on="nc").select("s1", "entity_id").with_columns(
        pl.lit(-1.0).alias("asim"), pl.lit(100.0).alias("nsim"))
    outs = pl.concat([outs, outs2]).unique(["s1", "entity_id"], keep="first")
    incand = sc.select(pl.col("s1"), pl.col("cand").alias("entity_id"))
    outs = outs.join(incand, on=["s1", "entity_id"], how="anti")
    out_ids = outs["entity_id"].unique().to_list()
    pout = pool_lf.filter(pl.col("entity_id").is_in(out_ids)).select(RAW).collect()
    outs = outs.join(pout, on="entity_id", how="left")
    oown = links.filter(pl.col("mid").is_in(out_ids)).rename({"s1": "owner", "mid": "entity_id"})
    outs = outs.join(oown, on="entity_id", how="left")

    sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("r"))
    sel = links.filter(pl.col("s1").is_in(ids)).with_columns(pl.lit(True).alias("sel")).rename({"mid": "cand"})
    sc = sc.join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
    sc = sc.join(pool.rename({"entity_id": "cand"}).select("cand", "business_name", "business_address"), on="cand", how="left")
    sc = sc.join(own.rename({"mid": "cand"}), on="cand", how="left")
    lines = [f"# Inspection sheet test {country} (n={n}, seed={SEED})", "",
             "cols: rank | p | SEL | cand | name | address | owner (if selected by ANOTHER S1)", ""]
    stats = {"n": n, "empty": 0, "sel": 0, "outside": 0}
    for k, rec in enumerate(samp.sort("entity_id").iter_rows(named=True), 1):
        g = sc.filter(pl.col("s1") == rec["entity_id"]).sort("r")
        nsel = int(g["sel"].sum())
        stats["sel"] += nsel
        stats["empty"] += int(nsel == 0)
        lines.append(f"## [{country} {k:02d}] {rec['entity_id']} | {clip(rec['business_name'], 60)} | "
                     f"{clip(rec['business_address'], 90)}   (ncand={g.height}, nsel={nsel})")
        show = g.filter((pl.col("r") <= 6) | pl.col("sel"))
        for r in show.iter_rows(named=True):
            ow = ""
            if r["owner"] is not None and r["owner"] != rec["entity_id"]:
                ow = f"  -> OWNED BY {r['owner']} '{clip(r['owner_name'], 40)}' | {clip(r['owner_addr'], 50)}"
            lines.append(f"    {r['r']:>2} {r['p']:.3f} {'SEL' if r['sel'] else '   '} {r['cand']:<13} "
                         f"{clip(r['business_name'], 50):<50} | {clip(r['business_address'], 70)}{ow}")
        o = outs.filter(pl.col("s1") == rec["entity_id"]).sort("asim", descending=True).head(4)
        stats["outside"] += int(o.height > 0)
        for r in o.iter_rows(named=True):
            ow = f"  -> owned by {r['owner']}" if r["owner"] else "  (unowned)"
            lines.append(f"    OUT asim={r['asim']:.0f} nsim={r['nsim']:.0f} {r['entity_id']:<13} "
                         f"{clip(r['business_name'], 50):<50} | {clip(r['business_address'], 70)}{ow}")
        lines.append("")
    with open(f"{OUT}/inspect_{country}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    log(country, stats)
log("done")
