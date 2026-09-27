# -*- coding: utf-8 -*-
"""diag_fr_d.py -- generator-invariant segments: true share on train OOF (US, India) vs rate / p / selection on
test France, US, India. If a segment has the SAME true share in US and India (two different languages, one data
generator), France probably has it too -> a France segment whose p is far below that share = missed true links.

Segments are built from normalized frames (label-free definitions):
  street_same : street words (addr_norm words >=3 letters, no digits, minus city/state/street-type/stop words)
                Jaccard >= 0.6, both non-empty
  hn relation : first house numbers (digits): equal / 1-digit substitution / 1-digit insertion-deletion /
                transposition / |diff| <= 10 / other
  name        : core equal (sorted tokens) / c_tset >= 0.8 / alias-like (token_set < 0.5) / acronym
Candidates: top-10 by p of 20k sampled S1 per (split, country).
Writes /vol/diag/fr/segments.md
"""
import os
import re
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")
from ber import decide, normalize as N  # noqa: E402

V3 = "/vol/work_v3"
OUT = "/vol/diag/fr"
LAM, EB = 0.05303060038344832, 2.0
T0 = time.time()
MD = []
COLS = ["entity_id", "business_name", "business_address", "name_core", "addr_norm", "house_nums", "city_key", "state_key"]
STOP = {"the", "and", "of", "de", "du", "des", "la", "le", "les", "rue", "avenue", "road", "street", "near", "opp",
        "unit", "floor", "box", "city", "county", "town", "township", "village", "nagar", "colony", "sector", "main",
        "cross", "layout", "marg", "block", "plot", "flat", "shop", "hno", "dno", "no", "house", "door", "bldg", "apt",
        "appartement", "appt", "etage", "bat", "pmb", "suite", "ste", "cdp", "twp", "region", "district", "dist"}
STYPES = set().union(*N.STREET_TYPE_CANON.values())
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


def street_words(addr, city, state):
    drop = set(city.split()) | set(state.split())
    return {w for w in addr.split() if len(w) >= 3 and not NUMRE.search(w) and w not in drop and w not in STOP and w not in STYPES}


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


def name_rel(q, c, raw_c):
    if not c:
        return "empty"
    if " ".join(sorted(set(q.split()))) == " ".join(sorted(set(c.split()))):
        return "core equal"
    ts = fuzz.token_set_ratio(q, c)
    if ts >= 80:
        return "tset>=80"
    compact = re.sub(r"[^a-z]", "", N.fold(raw_c or ""))
    if len(compact) <= 4 and compact and compact[0] == (q[:1] if q else ""):
        return "acronym"
    if ts < 50:
        return "alias(tset<50)"
    return "tset50-80"


def build(split, country, n_s1=20_000, top=10):
    if split == "train":
        oof = pl.read_parquet(f"{V3}/model/oof.parquet", columns=["s1", "cand", "label", "p"])
        ids_c = pl.scan_parquet(f"{V3}/norm/train_{country}_s1.parquet").select(pl.col("entity_id").alias("s1")).collect()
        oof = oof.join(ids_c, on="s1", how="semi")
        links = decide.select_links(oof.select("s1", "cand", "p"), "expected_f", exclusivity="soft", lam_missing=LAM, empty_bias=EB)
        ids = oof["s1"].unique().sample(n_s1, seed=5).to_list()
        sc = oof.filter(pl.col("s1").is_in(ids))
        links = links.filter(pl.col("s1").is_in(ids))
    else:
        ids = pl.scan_parquet(f"{V3}/norm/test_{country}_s1.parquet").select("entity_id").collect()["entity_id"].sample(n_s1, seed=5).to_list()
        sc = pl.scan_parquet(f"{V3}/pred/test_{country}_scored.parquet").filter(pl.col("s1").is_in(ids)).collect()
        links = pl.scan_parquet(f"{V3}/pred/test_{country}_links.parquet").filter(pl.col("s1").is_in(ids)).collect()
    sc = sc.join(links.rename({"mid": "cand"}).with_columns(pl.lit(True).alias("sel")), on=["s1", "cand"], how="left") \
        .with_columns(pl.col("sel").fill_null(False), pl.col("p").rank("ordinal", descending=True).over("s1").alias("r"))
    sc = sc.filter(pl.col("r") <= top)
    s1 = pl.scan_parquet(f"{V3}/norm/{split}_{country}_s1.parquet").select(COLS).filter(pl.col("entity_id").is_in(ids)).collect()
    pool = pl.scan_parquet(f"{V3}/norm/{split}_{country}_pool.parquet").select(COLS).filter(
        pl.col("entity_id").is_in(sc["cand"].unique().to_list())).collect()
    P = sc.join(s1.rename({c: c + "_q" for c in COLS[1:]}).rename({"entity_id": "s1"}), on="s1", how="left") \
          .join(pool.rename({c: c + "_c" for c in COLS[1:]}).rename({"entity_id": "cand"}), on="cand", how="left")
    st_same, hr, nr = [], [], []
    for r in P.iter_rows(named=True):
        a = street_words(r["addr_norm_q"] or "", r["city_key_q"] or "", r["state_key_q"] or "")
        b = street_words(r["addr_norm_c"] or "", r["city_key_c"] or "", r["state_key_c"] or "")
        if not r["addr_norm_c"]:
            st_same.append("cand addr empty")
        elif a and b and len(a & b) / len(a | b) >= 0.6:
            st_same.append("same street")
        else:
            st_same.append("other street")
        hr.append(hn_rel(base(r["house_nums_q"]), base(r["house_nums_c"])))
        nr.append(name_rel(r["name_core_q"] or "", r["name_core_c"] or "", r["business_name_c"]))
    P = P.with_columns(pl.Series("street", st_same), pl.Series("hnrel", hr), pl.Series("namerel", nr))
    if "label" not in P.columns:
        P = P.with_columns(pl.lit(None, dtype=pl.Int8).alias("label"))
    log("built", split, country, P.height)
    return P, len(ids)


frames = {}
for split, c in (("train", "US"), ("train", "India"), ("test", "France"), ("test", "US"), ("test", "India")):
    frames[(split, c)] = build(split, c)

md("# Generator-invariant segments: train true share vs test France / US / India", "",
   "Top-10 candidates by p of 20k sampled S1 per split/country. rate = pairs per S1; true = share of true matches "
   "(train only); p = mean p; sel = share selected. A segment whose true share is similar in US and India is "
   "assumed generator-level; France rows with p/sel far below that share = likely missed true links.", "")
keys = ["street", "hnrel", "namerel"]
agg = {}
for k, (P, n) in frames.items():
    agg[k] = (P.group_by(keys).agg(pl.len().alias("n"), pl.col("label").cast(pl.Float64).mean().alias("true"),
                                   pl.col("p").mean().alias("p"), pl.col("sel").cast(pl.Float64).mean().alias("sel"))
              .with_columns((pl.col("n") / n).alias("rate")))
base_tab = agg[("train", "US")].select(*keys, pl.col("rate").alias("US_rate"), pl.col("true").alias("US_true"), pl.col("p").alias("US_p"),
                                       pl.col("sel").alias("US_sel"))
for k, lab in ((("train", "India"), "IN"), (("test", "France"), "tFR"), (("test", "US"), "tUS"), (("test", "India"), "tIN")):
    a = agg[k].select(*keys, pl.col("rate").alias(f"{lab}_rate"), *( [pl.col("true").alias(f"{lab}_true")] if lab == "IN" else []),
                      pl.col("p").alias(f"{lab}_p"), pl.col("sel").alias(f"{lab}_sel"))
    base_tab = base_tab.join(a, on=keys, how="full", coalesce=True)
base_tab = base_tab.fill_null(0.0).with_columns((pl.col("US_rate") + pl.col("IN_rate") + pl.col("tFR_rate")).alias("_w")) \
    .filter(pl.col("_w") >= 0.01).sort("_w", descending=True).drop("_w")
cols = base_tab.columns
table(cols, [tuple(r) for r in base_tab.rows()])
# expected FN / FP for France assuming the train true share (mean of US, India) holds per segment
rows = []
tot_fn = {"tFR": 0.0, "tUS": 0.0, "tIN": 0.0}
tot_fp = {"tFR": 0.0, "tUS": 0.0, "tIN": 0.0}
for r in base_tab.iter_rows(named=True):
    t = np.nanmean([r["US_true"] if r["US_rate"] > 0 else np.nan, r["IN_true"] if r["IN_rate"] > 0 else np.nan])
    if np.isnan(t):
        continue
    for lab in ("tFR", "tUS", "tIN"):
        rate, sel = r[f"{lab}_rate"], r[f"{lab}_sel"]
        # truth count in segment = rate * t ; selected = rate * sel ; if selection were precise:
        fn = max(rate * t - rate * sel, 0.0)
        fp = max(rate * sel - rate * t, 0.0)
        tot_fn[lab] += fn
        tot_fp[lab] += fp
    rows.append((r["street"], r["hnrel"], r["namerel"], float(t), r["tFR_rate"], r["tFR_sel"], r["tFR_p"],
                 max(r["tFR_rate"] * (t - r["tFR_sel"]), 0.0), max(r["tFR_rate"] * (r["tFR_sel"] - t), 0.0)))
md("## implied France shortfall per segment (assuming the train true share holds)", "",
   "fn/S1 = rate x max(true - sel, 0); fp/S1 = rate x max(sel - true, 0). Only a heuristic.", "")
rows.sort(key=lambda x: -(x[7] + x[8]))
table(["street", "hnrel", "namerel", "train true", "FR rate", "FR sel", "FR p", "FR fn/S1", "FR fp/S1"], rows[:25])
md(f"totals implied fn/S1: " + ", ".join(f"{k} {v:.4f}" for k, v in tot_fn.items()) +
   " ; fp/S1: " + ", ".join(f"{k} {v:.4f}" for k, v in tot_fp.items()), "")

# examples
def ex(P, cond, n, title, sort="p"):
    E = P.filter(cond).sort(sort, descending=True).head(n)
    md(f"### {title} ({P.filter(cond).height} pairs)")
    for r in E.iter_rows(named=True):
        md(f"- p={r['p']:.3f} sel={int(r['sel'])} lab={r['label']} | S1 `{r['business_name_q']}` | `{r['business_address_q']}` "
           f"|| cand `{r['business_name_c']}` | `{r['business_address_c']}` [{r['hnrel']}, {r['namerel']}]")
    md("")


seg = (pl.col("street") == "same street") & pl.col("hnrel").is_in(["1-digit subst", "1-digit ins/del", "transposition", "offset<=10", "other"])
Pus = frames[("train", "US")][0]
Pin = frames[("train", "India")][0]
Pfr = frames[("test", "France")][0]
ex(Pus, seg & (pl.col("label") == 1), 25, "train US TRUE same street, house number differs")
ex(Pus, seg & (pl.col("label") == 0) & pl.col("namerel").is_in(["core equal", "tset>=80"]), 25, "train US FALSE same street, hn differs, similar name")
ex(Pin, seg & (pl.col("label") == 1), 15, "train India TRUE same street, house number differs")
ex(Pfr, seg & ~pl.col("sel") & pl.col("namerel").is_in(["core equal", "tset>=80"]), 30, "test France UNSELECTED same street, hn differs, similar name")
ex(Pfr, seg & pl.col("sel"), 15, "test France SELECTED same street, hn differs")
ex(Pus, (pl.col("street") == "cand addr empty") & (pl.col("namerel") == "core equal") & (pl.col("label") == 1), 12, "train US TRUE empty-address exact core")
ex(Pus, (pl.col("street") == "cand addr empty") & (pl.col("namerel") == "core equal") & (pl.col("label") == 0), 12, "train US FALSE empty-address exact core")
ex(Pfr, (pl.col("street") == "cand addr empty") & (pl.col("namerel") == "core equal") & ~pl.col("sel"), 20, "test France UNSELECTED empty-address exact core")
ex(Pus, (pl.col("street") == "same street") & (pl.col("hnrel") == "equal") & pl.col("namerel").is_in(["alias(tset<50)", "acronym"]) & (pl.col("label") == 0), 12, "train US FALSE same address, alias/acronym name")
ex(Pfr, (pl.col("street") == "same street") & (pl.col("hnrel") == "equal") & pl.col("namerel").is_in(["alias(tset<50)", "acronym", "tset50-80"]) & ~pl.col("sel"), 20, "test France UNSELECTED same address, different name")
with open(f"{OUT}/segments.md", "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(MD) + "\n")
log("done")
