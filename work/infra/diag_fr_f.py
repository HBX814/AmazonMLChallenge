# -*- coding: utf-8 -*-
"""diag_fr_f.py -- same house number + street, SAME vs DIFFERENT city: do true duplicates change city?

For 20k sampled S1 per split/country: pool records sharing (base house number | street words) with the S1,
split by city relation (same city / other city / pool city empty) x name relation (core equal / ts>=80 / other).
Reports per S1: rate, share that are candidates, mean p, selected share, and (train) true-match share.
Also: city_key pair counts for France "other city, name ts>=80" records, and examples.
Writes /vol/diag/fr/city_swap.md
"""
import collections
import re
import sys
import time

import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
LAM, EB = 0.05303060038344832, 2.0
T0 = time.time()
MD = []
STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "d", "l", "near", "opp", "unit", "floor", "box", "city",
        "county", "town", "township", "village", "no", "hno", "dno", "plot", "flat", "shop", "apt", "appt", "appartement",
        "etage", "bat", "pmb", "suite", "cdp", "twp"}
NUMRE = re.compile(r"\d")
COLS = ["entity_id", "business_name", "business_address", "name_core", "addr_norm", "house_nums", "city_key", "state_key"]


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def md(*lines):
    MD.extend(lines)


def table(header, rows):
    md("| " + " | ".join(header) + " |", "|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
    md("")


def key2(df):
    keys = []
    for a, hn, city, state in zip(df["addr_norm"].to_list(), df["house_nums"].to_list(), df["city_key"].to_list(),
                                  df["state_key"].to_list()):
        b = None
        for x in hn or []:
            m = re.match(r"\D*(\d+)", x)
            if m:
                b = m.group(1).lstrip("0") or "0"
                break
        drop = set((city or "").split()) | set((state or "").split())
        sw = sorted({w for w in (a or "").split() if len(w) >= 2 and not NUMRE.search(w) and w not in drop and w not in STOP})
        keys.append(f"{b}|{' '.join(sw)}" if b is not None and len(sw) >= 1 else "")
    return df.with_columns(pl.Series("k2", keys))


oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
gt_all = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"])
rows = []
for split, country in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "US"), ("test", "India")):
    s1 = pl.read_parquet(f"{V3}/norm/{split}_{country}_s1.parquet", columns=COLS)
    if split == "train":
        sc = oof.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
        links = decide.select_links(sc.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
    else:
        sc = pl.read_parquet(f"{V3}/pred/test_{country}_scored.parquet")
        links = pl.read_parquet(f"{V3}/pred/test_{country}_links.parquet")
    ids = sc["s1"].unique().sample(20_000, seed=11).to_list()
    S = key2(s1.filter(pl.col("entity_id").is_in(ids))).filter(pl.col("k2") != "")
    pool = key2(pl.read_parquet(f"{V3}/norm/{split}_{country}_pool.parquet", columns=COLS)).filter(pl.col("k2") != "")
    J = S.select(pl.col("entity_id").alias("s1"), "k2", pl.col("city_key").alias("c1"), pl.col("name_core").alias("n1"),
                 pl.col("business_name").alias("raw1"), pl.col("business_address").alias("addr1")) \
        .join(pool.select(pl.col("entity_id").alias("cand"), "k2", pl.col("city_key").alias("c2"), pl.col("name_core").alias("n2"),
                          pl.col("business_name").alias("raw2"), pl.col("business_address").alias("addr2")), on="k2")
    J = J.join(sc.filter(pl.col("s1").is_in(ids)).select("s1", "cand", "p"), on=["s1", "cand"], how="left")
    J = J.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("sel").fill_null(False))
    if split == "train":
        J = J.join(gt_all.rename({"mid": "cand"}).with_columns(pl.lit(1).alias("lab")), on=["s1", "cand"], how="left") \
            .with_columns(pl.col("lab").fill_null(0))
    else:
        J = J.with_columns(pl.lit(None, dtype=pl.Int32).alias("lab"))
    ts = [fuzz.token_set_ratio(a or "", b or "") for a, b in zip(J["n1"].to_list(), J["n2"].to_list())]
    ceq = [" ".join(sorted(set((a or "").split()))) == " ".join(sorted(set((b or "").split()))) for a, b in zip(J["n1"].to_list(), J["n2"].to_list())]
    J = J.with_columns(pl.Series("ts", ts), pl.Series("ceq", ceq))
    J = J.with_columns(
        pl.when(pl.col("c2") == "").then(pl.lit("pool city empty")).when(pl.col("c1") == pl.col("c2")).then(pl.lit("same city"))
          .otherwise(pl.lit("other city")).alias("crel"),
        pl.when(pl.col("ceq")).then(pl.lit("core equal")).when(pl.col("ts") >= 80).then(pl.lit("ts>=80")).otherwise(pl.lit("other name")).alias("nrel"))
    for (crel, nrel), g in sorted(J.group_by(["crel", "nrel"]), key=lambda kv: kv[0]):
        rows.append((split, country, crel, nrel, g.height / 20_000, float(g["p"].is_not_null().mean()),
                     float(g["p"].fill_null(0).mean()), float(g["sel"].mean()),
                     float(g["lab"].mean()) if split == "train" else float("nan")))
    if country == "France" or split == "train":
        X = J.filter((pl.col("crel") == "other city") & (pl.col("nrel") != "other name"))
        cc = collections.Counter(zip(X["c1"].to_list(), X["c2"].to_list()))
        md(f"### {split} {country}: other-city same-(hn,street) similar-name records: {X.height} ({X.height / 20_000:.4f}/S1); "
           f"top city pairs: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in cc.most_common(12)))
        for r in X.sort("p", descending=True, nulls_last=True).head(12).iter_rows(named=True):
            md(f"- p={r['p']} sel={int(r['sel'])} lab={r['lab']} | `{r['raw1']}` | `{r['addr1']}` || `{r['raw2']}` | `{r['addr2']}`")
        md("")
    log("done", split, country, J.height)
MDH = ["# Same house number + street: same vs other city", "",
       "per S1 = records per sampled S1; in_cand = share that are candidates; p = mean p (0 if not a candidate); "
       "sel = share selected; true = share true (train).", ""]
MD[:0] = MDH
table(["split", "country", "city rel", "name rel", "per S1", "in_cand", "mean p", "sel", "true (train)"], rows)
with open(f"{OUT}/city_swap.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("all done")
