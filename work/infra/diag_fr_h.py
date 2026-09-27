# -*- coding: utf-8 -*-
"""diag_fr_h.py -- same street + same city, ANY house number: are house-number-perturbed duplicates present in the
POOL but missing from the candidate list / top ranks? (test France vs train US / India with labels)

key3 = street words | city. For 20k sampled S1 per split/country: pool records with the same key3 and a similar name
(name-core equal or token_set >= 80), split by house-number relation to the S1. Reports per S1: pool rate, share that
are candidates, share in the top-10 by p, mean p, selected share, true share (train only).
Writes /vol/diag/fr/street_hn.md
"""
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


def base(hn):
    for x in hn or []:
        m = re.match(r"\D*(\d+)", x)
        if m:
            return m.group(1).lstrip("0") or "0"
    return None


def hn_rel(a, b):
    if a is None or b is None:
        return "missing"
    if a == b:
        return "equal"
    if len(a) == len(b):
        d = sum(x != y for x, y in zip(a, b))
        if d == 1:
            return "1-digit subst"
        if d == 2 and sorted(a) == sorted(b):
            return "transposition"
    elif abs(len(a) - len(b)) == 1:
        s, l = (a, b) if len(a) < len(b) else (b, a)
        if any(l[:i] + l[i + 1:] == s for i in range(len(l))):
            return "1-digit ins/del"
    if abs(int(a) - int(b)) <= 10:
        return "offset<=10"
    return "other"


def key3(df):
    out = []
    for a, city, state in zip(df["addr_norm"].to_list(), df["city_key"].to_list(), df["state_key"].to_list()):
        drop = set((city or "").split()) | set((state or "").split())
        sw = sorted({w for w in (a or "").split() if len(w) >= 2 and not NUMRE.search(w) and w not in drop and w not in STOP})
        out.append(f"{' '.join(sw)}|{city}" if sw and city else "")
    return df.with_columns(pl.Series("k3", out))


oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
gt_all = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"])
rows = []
examples = {}
for split, country in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "US"), ("test", "India")):
    s1 = pl.read_parquet(f"{V3}/norm/{split}_{country}_s1.parquet", columns=COLS)
    if split == "train":
        sc = oof.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi").drop("label")
        links = decide.select_links(sc.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
    else:
        sc = pl.read_parquet(f"{V3}/pred/test_{country}_scored.parquet")
        links = pl.read_parquet(f"{V3}/pred/test_{country}_links.parquet")
    ids = sc["s1"].unique().sample(20_000, seed=13).to_list()
    scs = sc.filter(pl.col("s1").is_in(ids)).with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("r"))
    S = key3(s1.filter(pl.col("entity_id").is_in(ids))).filter(pl.col("k3") != "")
    pool = key3(pl.read_parquet(f"{V3}/norm/{split}_{country}_pool.parquet", columns=COLS)).filter(pl.col("k3") != "")
    J = S.select(pl.col("entity_id").alias("s1"), "k3", pl.col("name_core").alias("n1"), pl.col("house_nums").alias("h1"),
                 pl.col("business_name").alias("raw1"), pl.col("business_address").alias("addr1")) \
        .join(pool.select(pl.col("entity_id").alias("cand"), "k3", pl.col("name_core").alias("n2"), pl.col("house_nums").alias("h2"),
                          pl.col("business_name").alias("raw2"), pl.col("business_address").alias("addr2")), on="k3")
    ts = [fuzz.token_set_ratio(a or "", b or "") for a, b in zip(J["n1"].to_list(), J["n2"].to_list())]
    J = J.with_columns(pl.Series("ts", ts)).filter(pl.col("ts") >= 80)
    J = J.with_columns(pl.Series("hnrel", [hn_rel(base(a), base(b)) for a, b in zip(J["h1"].to_list(), J["h2"].to_list())]))
    J = J.join(scs.select("s1", "cand", "p", "r"), on=["s1", "cand"], how="left")
    J = J.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("sel").fill_null(False))
    if split == "train":
        J = J.join(gt_all.rename({"mid": "cand"}).with_columns(pl.lit(1).alias("lab")), on=["s1", "cand"], how="left") \
            .with_columns(pl.col("lab").fill_null(0))
    else:
        J = J.with_columns(pl.lit(None, dtype=pl.Int32).alias("lab"))
    for rel in ("equal", "1-digit ins/del", "1-digit subst", "transposition", "offset<=10", "other", "missing"):
        g = J.filter(pl.col("hnrel") == rel)
        if g.height == 0:
            continue
        rows.append((split, country, rel, g.height / 20_000, float(g["p"].is_not_null().mean()),
                     float((g["r"].fill_null(999) <= 10).mean()), float(g["p"].fill_null(0).mean()), float(g["sel"].mean()),
                     float(g["lab"].mean()) if split == "train" else float("nan"),
                     float(g.filter(pl.col("p").is_null())["lab"].mean()) if split == "train" and g.filter(pl.col("p").is_null()).height else float("nan")))
    examples[(split, country)] = J.filter(pl.col("hnrel") == "1-digit ins/del").sort("p", descending=True, nulls_last=True).head(10)
    log("done", split, country, J.height)
md("# Same street + same city, similar name (ts >= 80), by house-number relation", "",
   "per S1 = pool records per sampled S1; cand = share that are candidates; top10 = share in the S1's top-10 by p; "
   "p = mean p (0 if not a candidate); sel = share selected; true = share true (train); true|not cand = true share "
   "among those that are NOT candidates (train) = blocking loss.", "")
table(["split", "country", "hn relation", "per S1", "cand", "top10", "mean p", "sel", "true (train)", "true | not cand"], rows)
for k, E in examples.items():
    md(f"### {k[0]} {k[1]}: 1-digit ins/del examples")
    for r in E.iter_rows(named=True):
        md(f"- p={r['p']} r={r['r']} sel={int(r['sel'])} lab={r['lab']} | `{r['raw1']}` | `{r['addr1']}` || `{r['raw2']}` | `{r['addr2']}`")
    md("")
with open(f"{OUT}/street_hn.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("all done")
