# -*- coding: utf-8 -*-
"""diag_struct_b.py -- dataset-generation structure, part B (TRAIN labels for measurement only):
  B1 sibling structure (star vs chain vs per-source base record): name/address similarity S1-copy vs copy-copy
  B2 noise-operator rates on true links (per copy) and #operators per copy
  B3 distractors: key-sharing with S1 / linked / other distractors; test mixture estimate of the linked fraction
  B4 singletons vs matched S1: near-matching pool records
    python infra/diag_struct_b.py [country ...]
"""
import os
import re
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, "/root/proj/code/business_entity_resolution/src")

OUT = os.environ.get("DIAG_OUT", "/vol/diag/struct")
os.makedirs(OUT, exist_ok=True)
NORM = os.environ.get("DIAG_NORM", "/vol/work_v3/norm")
GTP = os.environ.get("DIAG_GT", "/vol/work/cache/train_gt_long.parquet")
T0 = time.time()
W = int(os.environ.get("OMP_NUM_THREADS", "8"))
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_fmt_str_lengths(60)


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


COLS = ["entity_id", "business_name", "business_address", "name_norm", "name_core", "name_key", "addr_norm",
        "house_nums", "city_key", "state_key", "script", "name_is_domain"]


def load(split, c, kind):
    lf = pl.scan_parquet(f"{NORM}/{split}_{c}_{kind}.parquet")
    have = lf.collect_schema().names()
    return lf.select([x for x in COLS if x in have]).collect()


def keys(df):
    hn = pl.col("house_nums").list.first()
    nc = pl.col("name_core").fill_null("")
    st = pl.col("state_key").fill_null("")
    ci = pl.col("city_key").fill_null("")
    return df.with_columns(
        pl.when(nc.str.len_chars() >= 2).then(nc + "|" + st).alias("k_nst"),
        pl.when(hn.is_not_null() & (ci != "")).then(hn + "|" + ci + "|" + st).alias("k_addr"),
        pl.when((nc.str.len_chars() >= 2) & hn.is_not_null()).then(nc + "|" + hn + "|" + st).alias("k_na"),
        pl.when(pl.col("addr_norm").fill_null("") != "").then(pl.col("addr_norm")).alias("k_an"),
        pl.col("business_address").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("addr_cf"),
        pl.col("business_name").str.strip_chars().str.to_lowercase().str.replace_all(r"\s+", " ").alias("name_cf"),
    )


def cp(a, b, scorer):
    return process.cpdist([x or "" for x in a], [x or "" for x in b], scorer=scorer, workers=W)


def comps(a):
    return [c.strip().lower() for c in (a or "").split(",") if c.strip()]


_ACC = re.compile(r"[À-ɏ]")
_NULL = re.compile(r"\b(null|none)\b", re.I)
_JUNK = re.compile(r"^[^\w\s]")


def fold(s):
    return "".join(ch for ch in unicodedata.normalize("NFD", s or "") if unicodedata.category(ch) != "Mn")


countries = sys.argv[1:] or ["US", "India"]
gt = pl.read_parquet(GTP)
summary = {}
for c in countries:
    log(f"==================== {c}")
    s1 = keys(load("train", c, "s1"))
    pool = keys(load("train", c, "pool"))
    if c == countries[0]:
        print("pool schema:", pool.schema)
        print(pool.head(3))
    g = gt.join(s1.select(pl.col("entity_id").alias("s1")), on="s1")
    pool = pool.join(g.rename({"mid": "entity_id"}), on="entity_id", how="left").with_columns(
        pl.col("s1").is_not_null().alias("linked"), pl.col("entity_id").str.slice(0, 2).alias("src"))
    log("loaded", s1.height, pool.height, "linked", int(pool["linked"].sum()))

    # ---------------------------------------------------------------- B1 sibling structure
    rng = np.random.default_rng(0)
    kk = g.group_by("s1").agg(pl.len().alias("k")).filter(pl.col("k") >= 2)
    samp = kk.sample(n=min(60000, kk.height), seed=0).select("s1")
    gs = g.join(samp, on="s1")
    P = pool.select("entity_id", "src", "business_name", "business_address", "name_norm", "name_core", "addr_norm",
                    "name_cf", "addr_cf", "house_nums", "script", "name_is_domain", "state_key", "city_key")
    S = s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("bn1"), pl.col("business_address").alias("ba1"),
                  pl.col("name_norm").alias("nn1"), pl.col("name_core").alias("nc1"), pl.col("addr_norm").alias("an1"),
                  pl.col("name_cf").alias("ncf1"), pl.col("addr_cf").alias("acf1"), pl.col("house_nums").alias("hn1"),
                  pl.col("state_key").alias("st1"), pl.col("city_key").alias("ci1"))
    cop = gs.rename({"mid": "entity_id"}).join(P, on="entity_id").join(S, on="s1")
    a = cop.select(pl.col("s1"), *[pl.col(x).alias(x + "_a") for x in P.columns])
    b = cop.select(pl.col("s1"), *[pl.col(x).alias(x + "_b") for x in P.columns])
    pr = a.join(b, on="s1").filter(pl.col("entity_id_a") < pl.col("entity_id_b")).join(S, on="s1")
    pr = pr.with_columns(pl.concat_str([pl.col("src_a"), pl.col("src_b")], separator="-").alias("ptype"))
    # S1-copy similarities
    cop = cop.with_columns(
        pl.Series("n_rat", cp(cop["nn1"].to_list(), cop["name_norm"].to_list(), fuzz.ratio)),
        pl.Series("a_rat", cp(cop["an1"].to_list(), cop["addr_norm"].to_list(), fuzz.ratio)),
        (pl.col("ncf1") == pl.col("name_cf")).alias("ncf_eq"), (pl.col("nn1") == pl.col("name_norm")).alias("nn_eq"),
        (pl.col("acf1") == pl.col("addr_cf")).alias("acf_eq"), (pl.col("an1") == pl.col("addr_norm")).alias("an_eq"),
        (pl.col("addr_cf") == "").alias("aempty"))
    pr = pr.with_columns(
        pl.Series("n_rat", cp(pr["name_norm_a"].to_list(), pr["name_norm_b"].to_list(), fuzz.ratio)),
        pl.Series("a_rat", cp(pr["addr_norm_a"].to_list(), pr["addr_norm_b"].to_list(), fuzz.ratio)),
        pl.Series("n_rat1a", cp(pr["nn1"].to_list(), pr["name_norm_a"].to_list(), fuzz.ratio)),
        pl.Series("a_rat1a", cp(pr["an1"].to_list(), pr["addr_norm_a"].to_list(), fuzz.ratio)),
        (pl.col("name_cf_a") == pl.col("name_cf_b")).alias("ncf_eq"), (pl.col("name_norm_a") == pl.col("name_norm_b")).alias("nn_eq"),
        (pl.col("addr_cf_a") == pl.col("addr_cf_b")).alias("acf_eq"), (pl.col("addr_norm_a") == pl.col("addr_norm_b")).alias("an_eq"),
        ((pl.col("addr_cf_a") == "") | (pl.col("addr_cf_b") == "")).alias("aempty"),
        (pl.col("ncf1") != pl.col("name_cf_a")).alias("a_ne1n"), (pl.col("ncf1") != pl.col("name_cf_b")).alias("b_ne1n"),
        (pl.col("acf1") != pl.col("addr_cf_a")).alias("a_ne1a"), (pl.col("acf1") != pl.col("addr_cf_b")).alias("b_ne1a"))
    print(f"\n[B1 {c}] S1-copy similarity by source (sample {samp.height:,} S1 with k>=2):")
    print(cop.group_by("src").agg(pl.len().alias("n"), pl.col("n_rat").mean(), pl.col("a_rat").filter(~pl.col("aempty")).mean().alias("a_rat_ne"),
                                  pl.col("ncf_eq").mean(), pl.col("nn_eq").mean(), pl.col("acf_eq").mean(), pl.col("an_eq").mean(),
                                  pl.col("aempty").mean()).sort("src"))
    print(f"[B1 {c}] copy-copy similarity by pair type (n_rat1a = S1 vs first copy of the same pairs):")
    ne = ~pl.col("aempty")
    print(pr.group_by("ptype").agg(pl.len().alias("n"), pl.col("n_rat").mean(), pl.col("n_rat1a").mean(),
                                   pl.col("a_rat").filter(ne).mean().alias("a_rat_ne"), pl.col("a_rat1a").filter(ne).mean().alias("a_rat1a_ne"),
                                   pl.col("ncf_eq").mean(), pl.col("nn_eq").mean(), pl.col("acf_eq").filter(ne).mean().alias("acf_eq_ne"),
                                   pl.col("an_eq").filter(ne).mean().alias("an_eq_ne")).sort("ptype"))
    # shared-noise test: both copies differ from S1, are they identical to each other?
    both_n = pr.filter(pl.col("a_ne1n") & pl.col("b_ne1n"))
    both_a = pr.filter(pl.col("a_ne1a") & pl.col("b_ne1a") & ne)
    print(f"[B1 {c}] P(copy names equal (casefold) | both differ from S1) by ptype:",
          both_n.group_by("ptype").agg(pl.len(), pl.col("ncf_eq").mean()).sort("ptype").rows())
    print(f"[B1 {c}] P(copy ADDRESSES equal (casefold) | both differ from S1, non-empty) by ptype:",
          both_a.group_by("ptype").agg(pl.len(), pl.col("acf_eq").mean(), pl.col("an_eq").mean()).sort("ptype").rows())
    # typo-sharing test on names: copy a has a small edit vs S1 (1-2 edits on casefold), is b the same string?
    la = np.array(cp(pr["ncf1"].to_list(), pr["name_cf_a"].to_list(), Levenshtein.distance))
    lb = np.array(cp(pr["ncf1"].to_list(), pr["name_cf_b"].to_list(), Levenshtein.distance))
    typo_a = (la >= 1) & (la <= 2)
    eqab = pr["ncf_eq"].to_numpy()
    print(f"[B1 {c}] typo-sharing: P(b == a | a is 1-2 edits from S1) = {eqab[typo_a].mean():.4f} (n={typo_a.sum():,}); "
          f"P(b also 1-2 edits from S1 | a is) = {((lb >= 1) & (lb <= 2))[typo_a].mean():.4f}")
    ne_np = (~pr["aempty"]).to_numpy()
    ta = np.array(cp(pr["acf1"].to_list(), pr["addr_cf_a"].to_list(), Levenshtein.distance))
    ta = (ta >= 1) & (ta <= 2) & ne_np
    acfeq = pr["acf_eq"].to_numpy()
    ptype = pr["ptype"].to_numpy()
    for pt in ("S2-S2", "S3-S3", "S2-S3"):
        msk = ta & (ptype == pt)
        print(f"[B1 {c}] address small-edit sharing {pt}: P(addr b == addr a | a is 1-2 edits from S1) = {acfeq[msk].mean() if msk.any() else float('nan'):.4f} (n={msk.sum():,})")
    # per-source address base: fraction of S1 whose same-source copies (non-empty) all share one address string
    ss = (cop.filter(~pl.col("aempty")).group_by("s1", "src").agg(pl.len().alias("n"), pl.col("addr_cf").n_unique().alias("nu"),
                                                                 pl.col("addr_norm").n_unique().alias("nun"))
          .filter(pl.col("n") >= 2))
    print(f"[B1 {c}] same-source copies with >=2 non-empty addresses: share with ONE distinct raw(casefold) address / one addr_norm:",
          ss.group_by("src").agg(pl.len(), (pl.col("nu") == 1).mean().alias("one_raw"), (pl.col("nun") == 1).mean().alias("one_norm"),
                                 (pl.col("nu") / pl.col("n")).mean().alias("distinct_per_copy")).sort("src").rows())
    # component multiset equality (shuffle-only differences)
    def cset(x):
        return tuple(sorted(comps(x)))
    pa = pr.select("addr_cf_a", "addr_cf_b", "acf1", "ptype", "aempty").to_dicts()
    cnt = Counter()
    for r in pa:
        if r["aempty"]:
            continue
        sa, sb = cset(r["addr_cf_a"]), cset(r["addr_cf_b"])
        cnt[(r["ptype"], "sameset_ab")] += sa == sb
        cnt[(r["ptype"], "n")] += 1
        cnt[(r["ptype"], "sameset_1a")] += sa == cset(r["acf1"])
    print(f"[B1 {c}] address component-multiset equal: copy-copy vs S1-copy:",
          {pt: (round(cnt[(pt, 'sameset_ab')] / max(1, cnt[(pt, 'n')]), 4), round(cnt[(pt, 'sameset_1a')] / max(1, cnt[(pt, 'n')]), 4))
           for pt in ("S2-S2", "S3-S3", "S2-S3")})

    # ---------------------------------------------------------------- B2 noise operators on true links (per copy vs S1)
    rows = cop.select("src", "bn1", "business_name", "ba1", "business_address", "nn1", "name_norm", "nc1", "name_core",
                      "an1", "addr_norm", "hn1", "house_nums", "script", "name_is_domain", "st1", "state_key", "ci1", "city_key").to_dicts()
    tset = cp([r["nc1"] for r in rows], [r["name_core"] for r in rows], fuzz.token_set_ratio)
    lev = cp([r["nc1"] for r in rows], [r["name_core"] for r in rows], Levenshtein.distance)
    ops = Counter()
    nops = Counter()
    nops_src = Counter()
    for r, ts, lv in zip(rows, tset, lev):
        n1, n2 = r["bn1"] or "", r["business_name"] or ""
        a1, a2 = r["ba1"] or "", r["business_address"] or ""
        f = {}
        f["name_raw_eq"] = n1 == n2
        f["name_cf_eq"] = n1.lower().split() == n2.lower().split()
        f["n_case_change"] = (n1 != n2) and (n1.lower() == n2.lower())
        f["n_double_space"] = "  " in n2
        f["n_accent_inj"] = bool(_ACC.search(n2)) and not _ACC.search(n1)
        f["n_junk_lead"] = bool(_JUNK.match(n2.strip()))
        f["n_bracket"] = ("[" in n2) or ("(" in n2 and "(" not in n1)
        f["n_dba"] = bool(re.search(r"d\.?b\.?a\.?\b", n2, re.I))
        f["n_domain"] = bool(r["name_is_domain"])
        f["n_native"] = (r["script"] or "latin") not in ("latin", "")
        nn_eq = r["nn1"] == r["name_norm"]
        core_eq = r["nc1"] == r["name_core"]
        t1, t2 = (r["nc1"] or "").split(), (r["name_core"] or "").split()
        f["n_norm_eq"] = nn_eq
        f["n_legal_change"] = (not nn_eq) and core_eq
        f["n_shuffle"] = (not core_eq) and sorted(t1) == sorted(t2)
        f["n_typo"] = (not core_eq) and len(t1) == len(t2) and 1 <= lv <= 2 and not f["n_shuffle"]
        f["n_trunc_drop"] = (not core_eq) and set(t2) < set(t1)
        f["n_insert"] = (not core_eq) and set(t2) > set(t1)
        f["n_alias"] = ts < 40 and not f["n_native"] and not f["n_domain"]
        f["n_subst1"] = (not core_eq) and len(t1) == len(t2) and len(set(t1) ^ set(t2)) == 2 and not f["n_typo"]
        # address
        e2 = a2.strip() == ""
        f["a_empty"] = e2
        if not e2:
            c1, c2 = comps(a1), comps(a2)
            f["a_raw_eq"] = a1 == a2
            f["a_cf_eq"] = a1.lower() == a2.lower()
            f["a_upper"] = a2 == a2.upper() and a2 != a1
            f["a_null_tok"] = bool(_NULL.search(a2))
            f["a_norm_eq"] = r["an1"] == r["addr_norm"]
            f["a_comp_shuffle"] = sorted(c1) == sorted(c2) and c1 != c2
            f["a_comp_drop"] = len(c2) < len(c1)
            f["a_comp_add"] = len(c2) > len(c1)
            h1, h2 = set(r["hn1"] or []), set(r["house_nums"] or [])
            f["a_hn_missing"] = bool(h1) and not h2
            f["a_hn_change"] = bool(h1) and bool(h2) and not (h1 & h2)
            if f["a_hn_change"]:
                x1, x2 = sorted(h1)[0], sorted(h2)[0]
                d1, d2 = re.sub(r"\D", "", x1), re.sub(r"\D", "", x2)
                if d1 == d2:
                    ops["  hn_change: letter suffix/prefix"] += 1
                elif d1 and d2 and abs(int(d1[:9]) - int(d2[:9])) <= 2:
                    ops["  hn_change: +-1..2"] += 1
                elif d2 and d2 in d1:
                    ops["  hn_change: digit dropped"] += 1
                elif d1 and d1 in d2:
                    ops["  hn_change: digit added"] += 1
                else:
                    ops["  hn_change: other"] += 1
            f["a_state_diff"] = bool(r["st1"]) and bool(r["state_key"]) and r["st1"] != r["state_key"]
            f["a_city_diff"] = bool(r["ci1"]) and bool(r["city_key"]) and r["ci1"] != r["city_key"]
            f["a_city_missing"] = bool(r["ci1"]) and not r["city_key"]
        for k, v in f.items():
            ops[(r["src"], k)] += bool(v)
        ops[(r["src"], "_n")] += 1
        if not e2:
            ops[(r["src"], "_n_ne")] += 1
        name_ops = sum(bool(f[k]) for k in ("n_case_change", "n_double_space", "n_accent_inj", "n_junk_lead", "n_bracket", "n_dba", "n_domain",
                                            "n_native", "n_legal_change", "n_shuffle", "n_typo", "n_trunc_drop", "n_insert", "n_alias", "n_subst1"))
        addr_ops = sum(bool(f.get(k)) for k in ("a_empty", "a_null_tok", "a_comp_shuffle", "a_comp_drop", "a_comp_add", "a_hn_missing", "a_hn_change",
                                                "a_city_diff", "a_city_missing"))
        nops[("name", min(name_ops, 5))] += 1
        nops[("addr", min(addr_ops, 5))] += 1
        nops[("any_raw_identical", f["name_raw_eq"] and (not e2) and f.get("a_raw_eq", False))] += 1
    print(f"\n[B2 {c}] noise-operator rate per true copy (name ops over all copies; a_* over NON-EMPTY-address copies except a_empty):")
    keys_ = sorted({k for (_, k) in [x for x in ops if isinstance(x, tuple)] if not k.startswith("_")})
    tab = []
    for k in keys_:
        row = [k]
        for sname in ("S2", "S3"):
            den = ops[(sname, "_n")] if (k.startswith("n_") or k.startswith("name") or k == "a_empty") else ops[(sname, "_n_ne")]
            row.append(round(ops[(sname, k)] / max(1, den), 4))
        tab.append(row)
    for r in tab:
        print(f"   {r[0]:<18} S2 {r[1]:.4f}   S3 {r[2]:.4f}")
    print("   hn change subtypes:", {k: v for k, v in ops.items() if isinstance(k, str)})
    tot = sum(v for (t, _), v in nops.items() if t == "name")
    print(f"[B2 {c}] #name ops per copy:", {k[1]: round(v / tot, 4) for k, v in sorted(nops.items(), key=lambda x: str(x)) if k[0] == "name"})
    print(f"[B2 {c}] #addr ops per copy:", {k[1]: round(v / tot, 4) for k, v in sorted(nops.items(), key=lambda x: str(x)) if k[0] == "addr"})
    print(f"[B2 {c}] copy raw-identical to S1 (name and address):", round(nops[("any_raw_identical", True)] / tot, 5))

    # ---------------------------------------------------------------- B3 distractors
    def has_key(left, right, kcol, name):
        rk = right.select(pl.col(kcol)).drop_nulls().unique().with_columns(pl.lit(True).alias(name))
        return left.join(rk, on=kcol, how="left").with_columns(pl.col(name).fill_null(False))

    pk = pool.select("entity_id", "s1", "linked", "src", "k_nst", "k_addr", "k_na", "k_an", "addr_cf", "name_cf")
    # own-S1 key sharing for linked, any-S1 key sharing for everyone
    s1k = s1.select(pl.col("entity_id").alias("s1o"), "k_nst", "k_addr", "k_na", "k_an")
    for kc in ("k_nst", "k_addr", "k_na", "k_an"):
        pk = has_key(pk, s1k, kc, f"s1_{kc}")
        own = (pk.filter(pl.col("linked")).select("entity_id", "s1", kc)
               .join(s1k.select(pl.col("s1o").alias("s1"), pl.col(kc).alias("own_" + kc)), on="s1")
               .select("entity_id", (pl.col(kc) == pl.col("own_" + kc)).fill_null(False).alias(f"own_{kc}")))
        pk = pk.join(own, on="entity_id", how="left").with_columns(pl.col(f"own_{kc}").fill_null(False))
        # other pool records with the same key: counts linked / unlinked (excluding self)
        cnts = (pk.filter(pl.col(kc).is_not_null()).group_by(kc)
                .agg(pl.col("linked").sum().alias("cl"), (~pl.col("linked")).sum().alias("cu")))
        pk = (pk.join(cnts, on=kc, how="left")
              .with_columns((pl.col("cl").fill_null(0) - pl.col("linked").cast(pl.UInt32)).alias(f"pl_{kc}"),
                            (pl.col("cu").fill_null(0) - (~pl.col("linked")).cast(pl.UInt32)).alias(f"pu_{kc}"))
              .drop("cl", "cu"))
    # raw address string shared with another record of the same source
    ra = (pk.filter(pl.col("addr_cf") != "").group_by("src", "addr_cf")
          .agg(pl.col("linked").sum().alias("cl"), (~pl.col("linked")).sum().alias("cu")))
    pk = (pk.join(ra, on=["src", "addr_cf"], how="left")
          .with_columns((pl.col("cl").fill_null(0) - pl.col("linked").cast(pl.UInt32)).alias("pl_rawaddr_samesrc"),
                        (pl.col("cu").fill_null(0) - (~pl.col("linked")).cast(pl.UInt32)).alias("pu_rawaddr_samesrc")).drop("cl", "cu"))
    aggs = [pl.len().alias("n")]
    for kc in ("k_nst", "k_addr", "k_na", "k_an"):
        aggs += [pl.col(kc).is_not_null().mean().alias(f"has_{kc}"), pl.col(f"s1_{kc}").mean().alias(f"anyS1_{kc}"),
                 pl.col(f"own_{kc}").mean().alias(f"ownS1_{kc}"),
                 (pl.col(f"pl_{kc}") > 0).mean().alias(f"oLinked_{kc}"), (pl.col(f"pu_{kc}") > 0).mean().alias(f"oUnl_{kc}")]
    aggs += [(pl.col("pl_rawaddr_samesrc") > 0).mean().alias("oLinked_rawaddr_src"), (pl.col("pu_rawaddr_samesrc") > 0).mean().alias("oUnl_rawaddr_src"),
             (pl.col("addr_cf") == "").mean().alias("addr_empty")]
    b3 = pk.group_by("linked").agg(aggs).sort("linked")
    b3t = b3.transpose(include_header=True, column_names=["unlinked", "linked"])
    print(f"\n[B3 {c}] key sharing: anyS1 = some S1 (same country) has the key; ownS1 = its own S1 has it; oLinked/oUnl = another linked/unlinked pool record has it")
    print(b3t)
    b3.write_csv(f"{OUT}/b3_keys_{c}.csv")
    # distractor profile: which key relation with any S1
    prof = (pk.filter(~pl.col("linked"))
            .with_columns(pl.when(pl.col("s1_k_na")).then(pl.lit("A name+hn+state = some S1"))
                          .when(pl.col("s1_k_nst") & pl.col("s1_k_addr")).then(pl.lit("B name@state and addr of (maybe different) S1"))
                          .when(pl.col("s1_k_addr")).then(pl.lit("C address of some S1, other name"))
                          .when(pl.col("s1_k_nst")).then(pl.lit("D name@state of some S1, other address"))
                          .when(pl.col("addr_cf") == "").then(pl.lit("E empty addr, name not an S1 name@state"))
                          .otherwise(pl.lit("F no key shared with any S1")).alias("type"))
            .group_by("type").agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum()).alias("share")).sort("type"))
    print(f"[B3 {c}] unlinked-record profile vs S1 keys:\n", prof)
    profl = (pk.filter(pl.col("linked"))
             .with_columns(pl.when(pl.col("own_k_na")).then(pl.lit("A own S1 name+hn+state"))
                           .when(pl.col("own_k_nst") & pl.col("own_k_addr")).then(pl.lit("B own name@state + own addr"))
                           .when(pl.col("own_k_addr")).then(pl.lit("C own address, other name"))
                           .when(pl.col("own_k_nst")).then(pl.lit("D own name@state, other address"))
                           .when(pl.col("addr_cf") == "").then(pl.lit("E empty addr"))
                           .otherwise(pl.lit("F no own key")).alias("type"))
             .group_by("type").agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum()).alias("share")).sort("type"))
    print(f"[B3 {c}] linked-record profile vs OWN S1 keys (same key hierarchy):\n", profl)
    # distractor clusters among themselves: group sizes by k_na / k_addr among unlinked only vs copies per S1 by k_na
    for kc in ("k_na", "k_addr", "k_nst"):
        u = pk.filter(~pl.col("linked") & pl.col(kc).is_not_null()).group_by(kc).agg(pl.len().alias("sz"))
        print(f"[B3 {c}] unlinked groups by {kc}: size dist", u.group_by(pl.col("sz").clip(1, 8)).agg(pl.len()).sort("sz").rows())
    lg = pk.filter(pl.col("linked") & pl.col("k_na").is_not_null()).group_by("s1", "k_na").agg(pl.len().alias("sz"))
    print(f"[B3 {c}] linked groups by (s1,k_na): size dist", lg.group_by(pl.col("sz").clip(1, 8)).agg(pl.len()).sort("sz").rows())
    # distractor examples by type
    ex = pool.filter(~pl.col("linked")).sample(n=12, seed=3).select("entity_id", "business_name", "business_address")
    print(f"[B3 {c}] random unlinked examples:", ex.rows())
    summary[c] = {"pk": pk.select("linked", "k_nst", "k_addr", "k_na", "s1_k_nst", "s1_k_addr", "s1_k_na", "s1_k_an", "pl_k_na", "pu_k_na",
                                  "pl_k_addr", "pu_k_addr", "addr_cf").with_columns(
        ((pl.col("pl_k_na") + pl.col("pu_k_na")) > 0).alias("other_k_na"), ((pl.col("pl_k_addr") + pl.col("pu_k_addr")) > 0).alias("other_k_addr"))}

    # ---------------------------------------------------------------- B4 singletons
    s1s = s1.select("entity_id", "business_name", "business_address", "name_core", "k_nst", "k_addr", "k_na", "k_an", "house_nums")
    ks = g.group_by("s1").agg(pl.len().alias("k"))
    s1s = s1s.join(ks.rename({"s1": "entity_id"}), on="entity_id", how="left").with_columns(pl.col("k").fill_null(0))
    pp = pool.select("entity_id", "s1", "linked", "k_nst", "k_addr", "k_na", "k_an")
    for kc in ("k_nst", "k_addr", "k_na", "k_an"):
        cc = (pp.filter(pl.col(kc).is_not_null()).group_by(kc)
              .agg((~pl.col("linked")).sum().alias(f"u_{kc}"), pl.col("s1").alias("_own")))
        s1s = s1s.join(cc, on=kc, how="left").with_columns(
            pl.col(f"u_{kc}").fill_null(0), pl.col("_own").list.drop_nulls().list.len().fill_null(0).alias("_nl"))
        # own-linked count sharing the key
        own = (pp.filter(pl.col("linked") & pl.col(kc).is_not_null()).join(s1.select(pl.col("entity_id").alias("s1"), pl.col(kc).alias("k1")), on="s1")
               .filter(pl.col(kc) == pl.col("k1")).group_by("s1").agg(pl.len().alias(f"own_{kc}")))
        s1s = (s1s.join(own.rename({"s1": "entity_id"}), on="entity_id", how="left").with_columns(pl.col(f"own_{kc}").fill_null(0))
               .with_columns((pl.col("_nl") - pl.col(f"own_{kc}")).alias(f"oth_{kc}")).drop("_own", "_nl"))
        mult = s1.filter(pl.col(kc).is_not_null()).group_by(kc).agg(pl.len().alias(f"m1_{kc}"))
        s1s = s1s.join(mult, on=kc, how="left").with_columns(pl.col(f"m1_{kc}").fill_null(0))
    s1s = s1s.with_columns((pl.col("k") == 0).alias("single"))
    ag = [pl.len().alias("n")]
    for kc in ("k_nst", "k_addr", "k_na", "k_an"):
        ag += [(pl.col(f"u_{kc}") > 0).mean().alias(f"unlinked_{kc}"), (pl.col(f"oth_{kc}") > 0).mean().alias(f"otherLinked_{kc}"),
               (pl.col(f"own_{kc}") > 0).mean().alias(f"own_{kc}"), (pl.col(f"m1_{kc}") > 1).mean().alias(f"otherS1_{kc}")]
    ag += [pl.col("business_name").str.len_chars().mean().alias("name_len"), pl.col("business_address").str.len_chars().mean().alias("addr_len"),
           pl.col("house_nums").list.len().gt(0).mean().alias("has_hn"), pl.col("m1_k_nst").mean().alias("mean_S1_same_name@state")]
    b4 = s1s.group_by("single").agg(ag).sort("single")
    print(f"\n[B4 {c}] singleton vs matched S1 (unlinked_* = an UNLINKED pool record shares the key; otherLinked = a record linked to another S1; otherS1 = another S1 shares it):")
    print(b4.transpose(include_header=True, column_names=["matched", "single"]))
    sing = s1s.filter(pl.col("single") & (pl.col("u_k_na") > 0)).head(5).select("entity_id", "business_name", "business_address")
    print(f"[B4 {c}] singletons with an unlinked pool record sharing name+hn+state:", sing.rows())
    for sid in sing["entity_id"].to_list()[:3]:
        kv = s1s.filter(pl.col("entity_id") == sid)["k_na"][0]
        print("    ", sid, "->", pool.filter(pl.col("k_na") == kv).select("entity_id", "business_name", "business_address", "s1").rows())
    s1s.select("entity_id", "k", *[x for x in s1s.columns if x.startswith(("u_", "oth_", "own_", "m1_"))]).write_parquet(f"{OUT}/b4_s1keys_{c}.parquet")
    del pool, s1, pk, pr, cop

# ---------------------------------------------------------------- B5 test mixture estimate
log("B5 test")
for c in ["US", "India", "France"]:
    s1 = keys(load("test", c, "s1"))
    pool = keys(load("test", c, "pool")).with_columns(pl.col("entity_id").str.slice(0, 2).alias("src"))
    for kc in ("k_nst", "k_addr", "k_na", "k_an"):
        rk = s1.select(pl.col(kc)).drop_nulls().unique().with_columns(pl.lit(True).alias(f"s1_{kc}"))
        pool = pool.join(rk, on=kc, how="left").with_columns(pl.col(f"s1_{kc}").fill_null(False))
    for kc in ("k_na", "k_addr"):
        cn = pool.filter(pl.col(kc).is_not_null()).group_by(kc).agg(pl.len().alias("_c"))
        pool = pool.join(cn, on=kc, how="left").with_columns((pl.col("_c").fill_null(1) > 1).alias(f"other_{kc}")).drop("_c")
    r = {kc: float(pool[f"s1_{kc}"].mean()) for kc in ("k_nst", "k_addr", "k_na", "k_an")}
    r.update({f"other_{kc}": float(pool[f"other_{kc}"].mean()) for kc in ("k_na", "k_addr")})
    r["addr_empty"] = float((pool["addr_cf"] == "").mean())
    print(f"[B5 test {c}] pool {pool.height:,} S1 {s1.height:,} ratio {pool.height / s1.height:.3f}; share of pool records with an S1 sharing key:",
          {k: round(v, 4) for k, v in r.items()})
    # per-S1 count of pool records sharing k_na (proxy for copies)
    cn = pool.filter(pl.col("k_na").is_not_null()).group_by("k_na").agg(pl.len().alias("np"))
    m1 = s1.join(cn, on="k_na", how="left").with_columns(pl.col("np").fill_null(0))
    print(f"[B5 test {c}] per-S1 #pool sharing k_na: mean {m1['np'].mean():.3f}, dist", m1.group_by(pl.col("np").clip(0, 8)).agg(pl.len()).sort("np").rows())
    for tc, d in summary.items():
        pk = d["pk"]
        est = {}
        for kc in ("k_nst", "k_addr", "k_na", "k_an"):
            rl = float(pk.filter(pl.col("linked"))[f"s1_{kc}"].mean())
            ru = float(pk.filter(~pl.col("linked"))[f"s1_{kc}"].mean())
            est[kc] = round((r[kc] - ru) / (rl - ru), 4) if rl != ru else None
        print(f"      mixture estimate of LINKED fraction in test {c} using train-{tc} class rates:", est)
# train per-S1 #pool sharing k_na for reference
for tc in countries:
    s1 = keys(load("train", tc, "s1"))
    pool = keys(load("train", tc, "pool"))
    cn = pool.filter(pl.col("k_na").is_not_null()).group_by("k_na").agg(pl.len().alias("np"))
    m1 = s1.join(cn, on="k_na", how="left").with_columns(pl.col("np").fill_null(0))
    r = {kc: None for kc in ()}
    print(f"[B5 train {tc}] pool/S1 ratio {pool.height / s1.height:.3f}; per-S1 #pool sharing k_na: mean {m1['np'].mean():.3f}, dist",
          m1.group_by(pl.col("np").clip(0, 8)).agg(pl.len()).sort("np").rows())
log("done")
