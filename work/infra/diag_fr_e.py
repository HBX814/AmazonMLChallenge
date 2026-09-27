# -*- coding: utf-8 -*-
"""diag_fr_e.py -- address collisions between S1 records, exclusivity conflicts, strict blocking-miss proxy.

E1 share of S1 whose address key (base house number + street words + city_key) is shared with another S1
   (train US / India, test France / US / India); OOF F of shared vs unshared S1 (train labels)
E2 high-p candidates (p >= 0.9) NOT selected: share owned by another S1, and share where that owner has the
   same address key
E3 strict blocking-miss proxy: pool records with the SAME address key and name-core token_set >= 80 that are NOT
   candidates of the S1 (per S1); on train, the true-match share of those records (labels)
E4 acronym records: candidate compact name == initials of the S1 name (with / without legal words)
Writes /vol/diag/fr/collisions.md
"""
import os
import re
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide, metric, normalize as N  # noqa: E402

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


def akey_frame(df):
    """entity_id -> akey (base house number | street words | city) ; '' when no house number or no street words."""
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
        keys.append(f"{b}|{' '.join(sw)}|{city}" if b is not None and sw else "")
    return df.select("entity_id").with_columns(pl.Series("akey", keys))


COLS = ["entity_id", "business_name", "name_core", "name_norm", "addr_norm", "house_nums", "city_key", "state_key"]
res1, res2, res3, res4 = [], [], [], []
oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
gt_all = pl.read_parquet("/vol/work/cache/train_gt_long.parquet", columns=["s1", "mid"])
for split, country in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "US"), ("test", "India")):
    s1 = pl.read_parquet(f"{V3}/norm/{split}_{country}_s1.parquet", columns=COLS)
    ak = akey_frame(s1)
    cnt = ak.filter(pl.col("akey") != "").group_by("akey").agg(pl.len().alias("n_s1_key"))
    ak = ak.join(cnt, on="akey", how="left").with_columns(pl.col("n_s1_key").fill_null(1))
    shared = float((ak["n_s1_key"] >= 2).mean())
    nokey = float((ak["akey"] == "").mean())
    log(split, country, "s1", s1.height, "shared", shared)
    if split == "train":
        sc = oof.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
        links = decide.select_links(sc.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
        ids = sc["s1"].unique()
        per = metric.per_s1_scores(links, gt_all, ids).join(ak.rename({"entity_id": "s1"}), on="s1", how="left")
        f_sh = float(per.filter(pl.col("n_s1_key") >= 2)["f"].mean())
        f_un = float(per.filter(pl.col("n_s1_key") < 2)["f"].mean())
        sh_oof = float((per["n_s1_key"] >= 2).mean())
    else:
        sc = pl.read_parquet(f"{V3}/pred/test_{country}_scored.parquet")
        links = pl.read_parquet(f"{V3}/pred/test_{country}_links.parquet")
        f_sh = f_un = sh_oof = float("nan")
    res1.append((split, country, s1.height, nokey, shared, float((ak["n_s1_key"] >= 3).mean()), sh_oof, f_sh, f_un))

    # E2 high-p unselected
    own = links.rename({"s1": "owner", "mid": "cand"})
    hp = sc.filter(pl.col("p") >= 0.9).join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")),
                                           on=["s1", "cand"], how="left").filter(pl.col("sel").is_null())
    hp = hp.join(own, on="cand", how="left").join(ak.select(pl.col("entity_id").alias("s1"), "akey"), on="s1", how="left") \
        .join(ak.select(pl.col("entity_id").alias("owner"), pl.col("akey").alias("akey_o")), on="owner", how="left")
    n_s1 = sc["s1"].n_unique()
    res2.append((split, country, hp.height / n_s1, float(hp["owner"].is_not_null().mean()) if hp.height else 0.0,
                 float(((hp["akey"] == hp["akey_o"]) & (hp["akey"] != "")).mean()) if hp.height else 0.0,
                 float(hp["label"].mean()) if "label" in hp.columns and hp.height else float("nan")))

    # E3 strict blocking-miss proxy on a 20k sample
    samp_ids = sc["s1"].unique().sample(20_000, seed=9).to_list()
    pool = pl.read_parquet(f"{V3}/norm/{split}_{country}_pool.parquet", columns=COLS)
    pak = akey_frame(pool)
    S = ak.filter(pl.col("entity_id").is_in(samp_ids) & (pl.col("akey") != "")).rename({"entity_id": "s1"})
    J = S.join(pak.rename({"entity_id": "cand"}), on="akey", how="inner")
    J = J.join(sc.select("s1", "cand").with_columns(pl.lit(1).alias("in_c")), on=["s1", "cand"], how="left")
    nm1 = s1.select(pl.col("entity_id").alias("s1"), pl.col("name_core").alias("n1"))
    nm2 = pool.select(pl.col("entity_id").alias("cand"), pl.col("name_core").alias("n2"), pl.col("business_name").alias("raw2"))
    J = J.join(nm1, on="s1", how="left").join(nm2, on="cand", how="left")
    J = J.with_columns(pl.Series("ts", [fuzz.token_set_ratio(a or "", b or "") for a, b in zip(J["n1"].to_list(), J["n2"].to_list())]))
    if split == "train":
        J = J.join(gt_all.rename({"mid": "cand"}).with_columns(pl.lit(1).alias("lab")), on=["s1", "cand"], how="left") \
            .with_columns(pl.col("lab").fill_null(0))
    else:
        J = J.with_columns(pl.lit(None, dtype=pl.Int32).alias("lab"))
    for thr in (80, 0):
        X = J.filter((pl.col("ts") >= thr))
        out = X.filter(pl.col("in_c").is_null())
        res3.append((split, country, thr, X.height / 20_000, out.height / 20_000,
                     float(out["lab"].mean()) if split == "train" and out.height else float("nan"),
                     float(X.filter(pl.col("in_c") == 1)["lab"].mean()) if split == "train" and X.height else float("nan")))
    if split == "test" and country == "France":
        ex = J.filter((pl.col("ts") >= 80) & pl.col("in_c").is_null()).head(25)
        md_ex = [f"- S1 {r['s1']} `{r['n1']}` [{r['akey']}] || OUT {r['cand']} `{r['raw2']}` ts={r['ts']:.0f}" for r in ex.iter_rows(named=True)]
    # E4 acronyms
    S1n = s1.filter(pl.col("entity_id").is_in(samp_ids)).select(pl.col("entity_id").alias("s1"), "name_norm", "name_core")
    C = sc.filter(pl.col("s1").is_in(samp_ids)).join(S1n, on="s1", how="left").join(
        pool.select(pl.col("entity_id").alias("cand"), pl.col("business_name").alias("raw2")), on="cand", how="left")
    C = C.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("sel").fill_null(False))
    ini_n = [''.join(t[0] for t in (x or "").split()) for x in C["name_norm"].to_list()]
    ini_c = [''.join(t[0] for t in (x or "").split()) for x in C["name_core"].to_list()]
    comp = [re.sub(r"[^a-z]", "", N.fold(x or "")) for x in C["raw2"].to_list()]
    acr = [len(c) >= 2 and (c == a or c == b) for c, a, b in zip(comp, ini_n, ini_c)]
    C = C.with_columns(pl.Series("acr", acr))
    A = C.filter(pl.col("acr"))
    res4.append((split, country, A.height / 20_000, float(A["p"].mean()) if A.height else float("nan"),
                 float(A["sel"].mean()) if A.height else float("nan"),
                 float(A["label"].mean()) if "label" in A.columns and A.height else float("nan"),
                 float(C.filter(~pl.col("acr") & (pl.col("raw2").str.len_chars() <= 4))["p"].mean())))
    del pool, pak, J, C
    log("done", split, country)

md("# Address collisions, exclusivity conflicts, strict blocking-miss proxy", "")
md("## E1 S1 address-key collisions (akey = base house number | street words | city)", "")
table(["split", "country", "n_s1", "no akey", "shared (>=2 S1)", "shared (>=3 S1)", "OOF shared share", "OOF F shared", "OOF F unshared"], res1)
md("## E2 high-p (>=0.9) candidates NOT selected, per S1", "")
table(["split", "country", "per S1", "owned by other S1", "owner has same akey", "true share (train)"], res2)
md("## E3 strict blocking-miss proxy: pool records with the same akey (and name token_set >= thr), per S1", "",
   "all = such records per S1; OUT = not in the candidate list; train: true share among OUT / among in-candidates", "")
table(["split", "country", "name thr", "same-akey recs/S1", "OUT/S1", "OUT true share", "in-cand true share"], res3)
md("France examples of OUT (same akey, name ts>=80, not a candidate):", *md_ex, "")
md("## E4 acronym candidates (compact name == initials of S1 name_norm or name_core)", "")
table(["split", "country", "per S1", "mean p", "selected", "true share (train)", "mean p of other <=4-char names"], res4)
with open(f"{OUT}/collisions.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("all done")
