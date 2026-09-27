# -*- coding: utf-8 -*-
"""diag_fr_j.py -- empty-address candidates: name uniqueness at the FULL-name level (name_norm incl. legal forms,
'& fils/freres/cie') vs at the core level. diag_struct measured P(linked | empty address) = 0.977 on train, so the
only question for an empty-address copy is WHICH S1 owns it.
Groups (top-10 candidates of 20k sampled S1 whose candidate address is empty):
  norm_uniq : sorted name_norm tokens equal to the S1's AND no other S1 of the country has that name_norm key
  core_uniq : name_core key equal AND no other S1 shares the core key
  core_amb_norm_uniq : core key shared by >= 2 S1 but the full name_norm is unique to this S1 (the 'Danimation &
                       Fils' vs '& Freres' case)
  core_amb : core key shared by >= 2 S1 and name_norm not unique / not equal
Reports rate per S1, mean p, selected share, true share (train). Writes /vol/diag/fr/empty_addr.md
"""
import sys
import time

import polars as pl

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
LAM, EB = 0.05303060038344832, 2.0
T0 = time.time()
MD = []


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def table(header, rows):
    MD.extend(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)])
    for r in rows:
        MD.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |")
    MD.append("")


def keys(df):
    return df.with_columns(
        pl.col("name_norm").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ").alias("nkey"),
        pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort().list.join(" ").alias("ckey"))


oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
rows, ex = [], []
for split, country in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "US"), ("test", "India")):
    s1 = keys(pl.read_parquet(f"{V3}/norm/{split}_{country}_s1.parquet", columns=["entity_id", "business_name", "name_norm", "name_core"]))
    s1 = s1.with_columns(pl.len().over("nkey").alias("n_nkey"), pl.len().over("ckey").alias("n_ckey"))
    if split == "train":
        sc = oof.join(s1.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
        links = decide.select_links(sc.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
    else:
        sc = pl.read_parquet(f"{V3}/pred/test_{country}_scored.parquet").with_columns(pl.lit(None, dtype=pl.Int8).alias("label"))
        links = pl.read_parquet(f"{V3}/pred/test_{country}_links.parquet")
    ids = sc["s1"].unique().sample(20_000, seed=17).to_list()
    C = sc.filter(pl.col("s1").is_in(ids)).with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("r")).filter(pl.col("r") <= 10)
    pool = keys(pl.scan_parquet(f"{V3}/norm/{split}_{country}_pool.parquet").select("entity_id", "business_name", "name_norm", "name_core", "addr_norm")
                .filter(pl.col("entity_id").is_in(C["cand"].unique().to_list())).collect())
    C = C.join(pool.filter(pl.col("addr_norm") == "").select(pl.col("entity_id").alias("cand"), pl.col("nkey").alias("nk_c"),
                                                             pl.col("ckey").alias("ck_c"), pl.col("business_name").alias("raw_c")), on="cand", how="inner")
    C = C.join(s1.select(pl.col("entity_id").alias("s1"), pl.col("nkey").alias("nk_q"), pl.col("ckey").alias("ck_q"), "n_nkey", "n_ckey",
                         pl.col("business_name").alias("raw_q")), on="s1", how="left")
    C = C.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left").with_columns(pl.col("sel").fill_null(False))
    C = C.with_columns(
        pl.when((pl.col("nk_c") == pl.col("nk_q")) & (pl.col("n_nkey") == 1) & (pl.col("n_ckey") == 1)).then(pl.lit("norm eq, name unique"))
          .when((pl.col("nk_c") == pl.col("nk_q")) & (pl.col("n_nkey") == 1) & (pl.col("n_ckey") >= 2)).then(pl.lit("norm eq unique, core shared"))
          .when((pl.col("ck_c") == pl.col("ck_q")) & (pl.col("n_ckey") == 1)).then(pl.lit("core eq, core unique"))
          .when((pl.col("ck_c") == pl.col("ck_q")) & (pl.col("nk_c") == pl.col("nk_q"))).then(pl.lit("norm eq, norm shared"))
          .when(pl.col("ck_c") == pl.col("ck_q")).then(pl.lit("core eq, core shared"))
          .otherwise(pl.lit("name differs")).alias("grp"))
    for grp, g in sorted(C.group_by("grp"), key=lambda kv: kv[0]):
        rows.append((split, country, grp[0], g.height / 20_000, float(g["p"].mean()), float(g["sel"].mean()),
                     float(g["label"].cast(pl.Float64).mean()) if split == "train" else float("nan"),
                     float(g["n_ckey"].mean())))
    if country == "France":
        for r in C.filter(pl.col("grp") == "norm eq unique, core shared").sort("p").head(12).iter_rows(named=True):
            ex.append(f"- p={r['p']:.3f} sel={int(r['sel'])} n_ckey={r['n_ckey']} | S1 `{r['raw_q']}` || cand `{r['raw_c']}` (empty address)")
    log("done", split, country, C.height)
MD[:0] = ["# Empty-address candidates by full-name vs core-name uniqueness", "",
          "per S1 = empty-address candidates in the S1's top-10 per sampled S1; true = share true (train); n_ckey = mean # S1 sharing the core key.", ""]
table(["split", "country", "group", "per S1", "mean p", "sel", "true (train)", "mean n_ckey"], rows)
MD.append("France examples, full name unique but core shared (lowest p first):")
MD.extend(ex)
with open(f"{OUT}/empty_addr.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("all done")
