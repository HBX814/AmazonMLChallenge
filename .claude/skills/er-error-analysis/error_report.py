# -*- coding: utf-8 -*-
"""error_report.py -- error analysis for the Business Entity Resolution pipeline (Master Bolt).

Turns one validation run (scored candidate pairs + final links + ground truth + normalized frames) into a
markdown report that says WHERE the macro-F0.5 points are lost and WHY, so the next change can be chosen by
expected gain instead of by guess. Also: label-free shift checks (France / test vs train OOF) and a manual
inspection sheet.

Inputs (CONTRACT column names; frames may be polars DataFrame / LazyFrame / .parquet path / .tsv path):
    scored : s1, cand, p [, label]      every pair the model scored (= the candidate set)
    links  : s1, mid                    final decisions (or a submission-format TSV)
    gt     : s1, mid                    ground truth long form (or train_ground_truth.tsv); pass the FULL gt
                                        so "belongs to another S1" can be detected
    s1     : entity_id, business_name, business_address, country [+ normalized columns]
    pool   : S2 u S3, same columns (name_norm, name_core, name_compact, addr_norm, house_nums, city_key,
             script, name_is_domain). Missing normalized columns are computed on the needed rows only
             (normalize.py if importable, else a crude regex fallback) -- slow (~2k rows/s), so pass
             normalized frames when you have them.
    s1_ids : the evaluated S1 ids (validation fold). Default = every entity_id of `s1`. Singletons count.

Contract functions
------------------
build_report(scored, links, gt, s1, pool, s1_ids=None, n_examples=4, max_tag_pairs=30_000, seed=0) -> dict
render_markdown(rep, title="") -> str
write_report(out_md, **same kwargs as build_report) -> dict      (also saves per_s1/tagged parquet)
per_s1_table(scored, links, gt, s1_meta, s1_ids=None) -> pl.DataFrame  (s1, country, k, m, tp, f, kc, mm, bm,
                                                        n_cand, f_ceiling, f_oracle_cut + loss components)
loss_decomposition(per_s1) -> pl.DataFrame         components that sum EXACTLY to 1 - macro F0.5
tag_error_pairs(err, s1n, pooln, links, gt_full, gen) -> pl.DataFrame   heuristic cause tags per error pair
unlabeled_stats(scored, links, s1_meta, split) -> pl.DataFrame   per-country label-free statistics
shift_report(ref, cur, features_ref=None, features_cur=None) -> (markdown str, dict)
inspection_sheet(scored, links, s1, pool, country="France", n=50, top=3, seed=0) -> markdown str
psi(ref_values, cur_values, bins=10) -> float

CLI
---
python error_report.py report --scored work/oof/scored.parquet --links work/oof/links.parquet \
       --gt work/cache/gt_long.parquet --s1 work/norm/train_s1.parquet --pool work/norm/train_pool.parquet \
       --ids work/folds/val_fold0.txt --out work/reports/err_2026-09-26_fold0.md
python error_report.py shift --ref-scored work/oof/scored.parquet --ref-links work/oof/links.parquet \
       --ref-s1 work/norm/train_s1.parquet --cur-scored work/test/scored.parquet --cur-links work/test/links.parquet \
       --cur-s1 work/norm/test_s1.parquet [--ref-features ... --cur-features ...] --out work/reports/shift.md
python error_report.py inspect --scored work/test/scored.parquet --links work/test/links.parquet \
       --s1 work/norm/test_s1.parquet --pool work/norm/test_pool.parquet --country France --n 50 \
       --out work/reports/france_inspect.md

Library example
---------------
    import error_report as er
    rep = er.build_report(oof_scored, oof_links, gt_long, s1_norm, pool_norm, s1_ids=val_ids)
    print(rep["headline"]["macro_f"], rep["loss_rollup"])
    open("work/reports/err.md", "w", encoding="utf-8").write(er.render_markdown(rep, "fold0 baseline"))
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import re
import sys
import unicodedata
from typing import Iterable, Optional

import polars as pl

BETA2 = 0.25
K_ORDER = ["0", "1", "2", "3", "4", "5", "6", "7+"]
ETYPES = ["FP", "FN_model", "FN_blocking"]
ETYPE_HELP = {
    "FP": "predicted link not in ground truth (false merge)",
    "FN_model": "true link that WAS a candidate but was not selected",
    "FN_blocking": "true link that never became a candidate (blocking miss)",
}
TAGS = ["generic_name_same_city", "house_number_mismatch", "alias_name", "native_script", "domain_form",
        "truncated_name", "empty_address", "legal_form_only_diff", "cross_duplicate", "competition_loss",
        "belongs_to_other_s1"]
TAG_HELP = {
    "generic_name_same_city": "near-identical generic name (shared by >=2 S1 or all-frequent tokens) in the same city",
    "house_number_mismatch": "house numbers on both sides, none agree (digits), pair otherwise similar (name >= 0.8 or addr >= 0.6)",
    "alias_name": "name unrelated (token sort & set < 0.5) but address similar (>= 0.8): alias / DBA / gibberish",
    "native_script": "candidate name written in an Indic script",
    "domain_form": "candidate name is a domain / handle form (x.com, www., @x)",
    "truncated_name": "candidate core tokens are a strict subset of the S1 core tokens",
    "empty_address": "candidate address empty after normalization",
    "legal_form_only_diff": "same core name, differs only by legal form / honorific",
    "cross_duplicate": "candidate is a near-duplicate of another record linked to the same S1",
    "competition_loss": "(FN) the true candidate was assigned to a different S1",
    "belongs_to_other_s1": "(FP) the candidate is a true link of a different S1",
}
LOSS_LABELS = {
    "singleton_fp": "FP on singleton S1 (truth empty, list non-empty)",
    "empty_blocking": "empty list on matched S1 - no true link among candidates",
    "empty_model": "empty list on matched S1 - true link was a candidate",
    "allwrong_blocking": "all-wrong list (false merge) - no true link among candidates",
    "allwrong_model": "all-wrong list (false merge) - true link was a candidate",
    "partial_fp": "partial list: extra false links",
    "partial_recall_model": "partial list: missed links that were candidates",
    "partial_recall_blocking": "partial list: missed links not in candidates",
}
_INDIC_RE = re.compile("[\u0900-\u0dff]")
_DOMAIN_RE = re.compile(r"(?i)(\.(com|net|org|in|co|fr|biz|info|us)\b|www\.|^\W*@|\s@\w)")
_DIGITS_RE = re.compile(r"\d+")


# =====================================================================================================
# imports of sibling modules (inside src/ber/ they are package siblings; in the skill folder they live in
# neighbouring skill folders)
# =====================================================================================================
def _import_sibling(name: str, skill_dir: str):
    try:
        from importlib import import_module
        if __package__:
            return import_module(f"{__package__}.{name}")
    except ImportError:
        pass
    try:
        return __import__(name)
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(os.path.dirname(here), skill_dir)
    if os.path.isfile(os.path.join(cand, name + ".py")):
        sys.path.insert(0, cand)
        try:
            return __import__(name)
        finally:
            sys.path.remove(cand)
    return None


metric = _import_sibling("metric", "er-f05-decisions")
if metric is None:
    raise ImportError("error_report.py needs metric.py (er-f05-decisions) next to it or on sys.path")
_nz = _import_sibling("normalize", "er-text-normalization")


# =====================================================================================================
# input helpers
# =====================================================================================================
def _as_lazy(x) -> Optional[pl.LazyFrame]:
    if x is None:
        return None
    if isinstance(x, pl.LazyFrame):
        return x
    if isinstance(x, pl.DataFrame):
        return x.lazy()
    if isinstance(x, (list, tuple)):
        return pl.concat([_as_lazy(v) for v in x], how="diagonal_relaxed")
    p = str(x)
    if p.lower().endswith(".parquet"):
        return pl.scan_parquet(p)
    if p.lower().endswith((".tsv", ".txt")):
        # quoting OFF and "" kept as "" (the data has quotes/apostrophes and literal NULL tokens)
        return pl.scan_csv(p, separator="\t", quote_char=None, infer_schema=False, empty_string_is_null=False)
    raise ValueError(f"unsupported input {p!r} (use DataFrame, LazyFrame, .parquet or .tsv)")


def _cols(lf: pl.LazyFrame) -> list:
    return lf.collect_schema().names()


def load_long(x) -> pl.DataFrame:
    """(s1, mid) frame from a long frame / parquet, or from a submission-format / ground-truth TSV."""
    lf = _as_lazy(x)
    cols = _cols(lf)
    if "s1" in cols and "mid" in cols:
        out = lf.select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8))
    elif "s1" in cols and "cand" in cols:
        out = lf.select(pl.col("s1").cast(pl.Utf8), pl.col("cand").cast(pl.Utf8).alias("mid"))
    else:
        a, b = cols[0], cols[1]
        out = (lf.select(pl.col(a).cast(pl.Utf8).str.strip_chars().alias("s1"),
                         pl.col(b).cast(pl.Utf8).fill_null("").str.split(",").alias("mid"))
                 .explode("mid", empty_as_null=True).with_columns(pl.col("mid").str.strip_chars())
                 .filter(pl.col("mid").is_not_null() & (pl.col("mid") != "")))
    return out.unique().collect()


def load_scored(x) -> pl.DataFrame:
    lf = _as_lazy(x)
    cols = _cols(lf)
    sel = [pl.col("s1").cast(pl.Utf8), pl.col("cand").cast(pl.Utf8),
           (pl.col("p").cast(pl.Float64) if "p" in cols else pl.lit(float("nan")).alias("p"))]
    if "label" in cols:
        sel.append(pl.col("label").cast(pl.Int8))
    return lf.select(sel).unique(subset=["s1", "cand"], keep="first").collect()


def _ids_frame(s1_ids) -> pl.DataFrame:
    if isinstance(s1_ids, pl.DataFrame):
        col = "s1" if "s1" in s1_ids.columns else s1_ids.columns[0]
        return s1_ids.select(pl.col(col).cast(pl.Utf8).alias("s1")).unique(maintain_order=True)
    if isinstance(s1_ids, str) and os.path.isfile(s1_ids):
        ids = metric.load_ids_file(s1_ids)
        return pl.DataFrame({"s1": pl.Series(ids, dtype=pl.Utf8)}).unique(maintain_order=True)
    return pl.DataFrame({"s1": pl.Series(list(s1_ids), dtype=pl.Utf8)}).unique(maintain_order=True)


def _fb(tp, m, k):
    """F_beta from counts (vectorised polars expressions; same edge cases as metric.f_beta_counts)."""
    return (pl.when((k == 0) & (m == 0)).then(1.0).when(tp == 0).then(0.0)
            .otherwise((1 + BETA2) * tp / (BETA2 * k + m)))


# =====================================================================================================
# per-S1 table, loss decomposition, breakdowns
# =====================================================================================================
def per_s1_table(scored, links, gt, s1_meta, s1_ids=None) -> pl.DataFrame:
    """One row per evaluated S1: k (true size), m (pred size), tp, f (metric.per_s1_scores), plus
    kc = true links among candidates, mm = true candidates not selected (model misses),
    bm = true links not in candidates and not selected (blocking misses), n_cand,
    f_ceiling = best F with perfect decisions on these candidates (blocking ceiling),
    f_oracle_cut = best F of a prefix of the p-ranked candidate list (ranking ceiling), and the 8 loss
    components of loss_decomposition (they sum to 1 - f per S1)."""
    sc = scored if isinstance(scored, pl.DataFrame) else load_scored(scored)
    lk = links if isinstance(links, pl.DataFrame) else load_long(links)
    g = gt if isinstance(gt, pl.DataFrame) else load_long(gt)
    meta = _as_lazy(s1_meta)
    mc = _cols(meta)
    idc = "entity_id" if "entity_id" in mc else "s1"
    meta = meta.select(pl.col(idc).cast(pl.Utf8).alias("s1"),
                       (pl.col("country").cast(pl.Utf8).fill_null("?") if "country" in mc
                        else pl.lit("?").alias("country")))
    if s1_ids is None:
        ev = meta.collect().unique(subset=["s1"], keep="first", maintain_order=True)
    else:
        ev = _ids_frame(s1_ids).join(meta.collect().unique(subset=["s1"], keep="first"), on="s1", how="left") \
                               .with_columns(pl.col("country").fill_null("?"))
    ids = ev.select("s1")
    r = metric.per_s1_scores(lk, g, ids)                                    # s1, k, m, tp, f
    sc = sc.join(ids, on="s1", how="semi")
    g_ev = g.join(ids, on="s1", how="semi")
    lk_ev = lk.join(ids, on="s1", how="semi")
    cand_key = sc.select("s1", pl.col("cand").alias("mid"), pl.lit(True).alias("in_cand"))
    link_key = lk_ev.select("s1", "mid", pl.lit(True).alias("in_link"))
    gl = (g_ev.join(cand_key, on=["s1", "mid"], how="left").join(link_key, on=["s1", "mid"], how="left")
              .with_columns(pl.col("in_cand").fill_null(False), pl.col("in_link").fill_null(False)))
    per = gl.group_by("s1").agg(
        pl.col("in_cand").sum().alias("kc"),
        (pl.col("in_cand") & ~pl.col("in_link")).sum().alias("mm"),
        (~pl.col("in_cand") & ~pl.col("in_link")).sum().alias("bm"))
    ncand = sc.group_by("s1").agg(pl.len().alias("n_cand"))
    # oracle cut: best prefix of the p-sorted candidate list, labels from gt
    lab = (sc.select("s1", "cand", "p")
             .join(g_ev.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("_y")), on=["s1", "cand"], how="left")
             .with_columns(pl.col("_y").fill_null(0))
             .sort(["s1", "p", "cand"], descending=[False, True, False], nulls_last=True)
             .with_columns(pl.col("_y").cum_sum().over("s1").alias("_tp"),
                           (pl.int_range(pl.len()).over("s1") + 1).alias("_m"))
             .join(r.select("s1", "k"), on="s1", how="left"))
    oc = (lab.with_columns(pl.when(pl.col("_tp") > 0)
                           .then((1 + BETA2) * pl.col("_tp") / (BETA2 * pl.col("k") + pl.col("_m")))
                           .otherwise(0.0).alias("_f"))
             .group_by("s1").agg(pl.col("_f").max().alias("_oc")))
    r = (r.join(ev, on="s1", how="left").join(per, on="s1", how="left").join(ncand, on="s1", how="left")
          .join(oc, on="s1", how="left")
          .with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in ("kc", "mm", "bm", "n_cand")]))
    k, m, tp, kc, mm = (pl.col(c) for c in ("k", "m", "tp", "kc", "mm"))
    r = r.with_columns(
        pl.when(k == 0).then(1.0).when(kc == 0).then(0.0)
          .otherwise((1 + BETA2) * kc / (BETA2 * k + kc)).alias("f_ceiling"),
        pl.when(k == 0).then(1.0).otherwise(pl.col("_oc").fill_null(0.0)).alias("f_oracle_cut"),
    ).drop("_oc")
    f1 = _fb(tp, tp, k)                     # false links removed
    f2 = _fb(tp + mm, tp + mm, k)           # model misses fixed too
    part = (k > 0) & (tp > 0)
    r = r.with_columns(
        ((k == 0) & (m > 0)).cast(pl.Float64).alias("singleton_fp"),
        ((k > 0) & (m == 0) & (kc == 0)).cast(pl.Float64).alias("empty_blocking"),
        ((k > 0) & (m == 0) & (kc > 0)).cast(pl.Float64).alias("empty_model"),
        ((k > 0) & (m > 0) & (tp == 0) & (kc == 0)).cast(pl.Float64).alias("allwrong_blocking"),
        ((k > 0) & (m > 0) & (tp == 0) & (kc > 0)).cast(pl.Float64).alias("allwrong_model"),
        pl.when(part).then(f1 - pl.col("f")).otherwise(0.0).alias("partial_fp"),
        pl.when(part).then(f2 - f1).otherwise(0.0).alias("partial_recall_model"),
        pl.when(part).then(1.0 - f2).otherwise(0.0).alias("partial_recall_blocking"),
        pl.when(k >= 7).then(pl.lit("7+")).otherwise(k.cast(pl.Utf8)).alias("k_bucket"),
        pl.when(m >= 7).then(pl.lit("7+")).otherwise(m.cast(pl.Utf8)).alias("m_bucket"),
    )
    return r


def loss_decomposition(per: pl.DataFrame) -> tuple:
    """Rows: component, label, n_s1, points (= contribution to 1 - macro F), share. Plus a roll-up into
    precision / recall-model / recall-blocking. Both sum exactly to 1 - macro F0.5."""
    n = max(per.height, 1)
    tot = float((1 - per["f"]).sum()) or 1.0
    rows = []
    for c, lab in LOSS_LABELS.items():
        s = float(per[c].sum())
        rows.append((c, lab, int((per[c] > 1e-12).sum()), s / n, s / tot))
    comp = pl.DataFrame(rows, schema=["component", "meaning", "n_s1", "points", "share"], orient="row")
    groups = {"precision (false links)": ["singleton_fp", "allwrong_blocking", "allwrong_model", "partial_fp"],
              "recall - model (true candidate not selected)": ["empty_model", "partial_recall_model"],
              "recall - blocking (true link not a candidate)": ["empty_blocking", "partial_recall_blocking"]}
    roll = pl.DataFrame([(gname, float(sum(per[c].sum() for c in cs)) / n,
                          float(sum(per[c].sum() for c in cs)) / tot) for gname, cs in groups.items()],
                        schema=["bucket", "points", "share"], orient="row")
    return comp, roll


def _group_table(per: pl.DataFrame, by, order=None) -> pl.DataFrame:
    n = max(per.height, 1)
    aggs = [pl.len().alias("n"), (pl.len() / n).alias("share"), pl.col("f").mean().alias("macro_f"),
            pl.col("k").mean().alias("mean_k"), pl.col("m").mean().alias("mean_m"),
            (pl.col("tp").sum() / pl.col("m").sum()).alias("micro_p"),
            (pl.col("tp").sum() / pl.col("k").sum()).alias("micro_r"),
            (pl.col("m") == 0).mean().alias("empty_rate"),
            pl.col("f_ceiling").mean().alias("ceiling_f"), pl.col("f_oracle_cut").mean().alias("oracle_cut_f"),
            ((1 - pl.col("f")).sum() / n).alias("loss_pts"),
            ((pl.col("f_ceiling") - pl.col("f")).sum() / n).alias("model_gap_pts"),
            ((1 - pl.col("f_ceiling")).sum() / n).alias("blocking_pts")]
    by = [by] if isinstance(by, str) else list(by)
    t = per.group_by(by).agg(aggs)
    if order is not None and len(by) == 1:
        t = t.with_columns(pl.col(by[0]).replace_strict({v: i for i, v in enumerate(order)}, default=len(order),
                                                        return_dtype=pl.Int64).alias("_o")).sort("_o").drop("_o")
    else:
        t = t.sort(by)
    return t


# =====================================================================================================
# normalized columns (computed only when missing) + generic-name statistics
# =====================================================================================================
_NEED = ["business_name", "business_address", "country", "name_norm", "name_core", "name_compact", "addr_norm",
         "house_nums", "city_key", "script", "name_is_domain"]
_CRUDE_LEGAL = {"llc", "inc", "corp", "co", "ltd", "pvt", "private", "limited", "llp", "lp", "pc", "pllc", "sarl",
                "sas", "sasu", "eurl", "sa", "sci", "the", "and", "of", "sri", "shri", "smt", "dr", "mr", "m", "s"}


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def _crude_norm(name: str, addr: str):
    nn = " ".join(re.findall(r"[0-9a-z]+", _fold(name)))
    core = " ".join(t for t in nn.split() if t not in _CRUDE_LEGAL) or nn
    an = " ".join(t for t in re.findall(r"[0-9a-z]+", _fold(addr)) if t not in ("null", "none"))
    hn = re.findall(r"\b\d+[a-z]?\b", an)
    comps = [c.strip() for c in addr.split(",") if c.strip()]
    city = " ".join(re.findall(r"[0-9a-z]+", _fold(comps[-2]))) if len(comps) >= 2 else ""
    return nn, core, an, hn, city


def _script_of(name: str) -> str:
    if _INDIC_RE.search(name or ""):
        return "indic"
    if any(ord(c) > 0x24F and c.isalpha() for c in (name or "")):
        return "other"
    return "latin"


_BATCH = []


def _batch_normalizer():
    """add_normalized_columns of the pipeline (normalize.py inside src/ber/, else normalize_frame.py) or None.
    Using it keeps house_nums / city_key identical to what the features saw."""
    if not _BATCH:
        fn = None
        for mod in (_nz, _import_sibling("normalize_frame", "er-text-normalization")):
            if mod is not None and hasattr(mod, "add_normalized_columns"):
                fn = mod.add_normalized_columns
                break
        _BATCH.append(fn)
    return _BATCH[0]


def _rowwise_norm(df: pl.DataFrame) -> dict:
    out = {c: [] for c in ("name_norm", "name_core", "addr_norm", "house_nums", "city_key")}
    for name, addr, cty in zip(df["business_name"].to_list(), df["business_address"].to_list(), df["country"].to_list()):
        if _nz is not None:
            v = _nz.name_variants(name, cty)
            an, ap = _nz.address_all(addr, cty)
            vals = (v["norm"], v["core"], an, list(ap.get("house_numbers") or []),
                    (ap.get("city_candidates") or [""])[-1])
        else:
            nn, core, an, hn, city = _crude_norm(name, addr)
            vals = (nn, core, an, hn, city)
        for c, x in zip(out, vals):
            out[c].append(x)
    return out


def ensure_normalized(df: pl.DataFrame) -> pl.DataFrame:
    """Add any missing CONTRACT normalized column (use on SUBSETS: ~0.3-0.5 ms per distinct string)."""
    for c in ("business_name", "business_address", "country"):
        if c not in df.columns:
            df = df.with_columns(pl.lit("").alias(c))
    df = df.with_columns([pl.col(c).cast(pl.Utf8).fill_null("") for c in ("business_name", "business_address", "country")])
    missing = [c for c in _NEED if c not in df.columns]
    if not missing:
        return df
    dtypes = {"house_nums": pl.List(pl.Utf8), "name_is_domain": pl.Boolean}
    if df.height == 0:
        return df.with_columns([pl.lit(None).cast(dtypes.get(c, pl.Utf8)).alias(c) for c in missing])
    fn = _batch_normalizer()
    if fn is not None:
        try:
            got = fn(df.select("business_name", "business_address", "country"), n_jobs=1)
            have = [c for c in missing if c in got.columns]
            if got.height == df.height and have:
                df = df.with_columns([got[c].cast(dtypes.get(c, pl.Utf8)).alias(c) for c in have])
                missing = [c for c in missing if c not in have]
        except Exception as ex:  # a changed sibling API must not kill the report
            print(f"[error_report] add_normalized_columns failed ({ex!r}); using row-wise fallback", file=sys.stderr)
    if not missing:
        return df
    out = _rowwise_norm(df) if any(c in missing for c in ("name_norm", "name_core", "name_compact", "addr_norm",
                                                           "house_nums", "city_key")) else {}
    add = []
    for c in missing:
        if c == "name_compact":
            src = out["name_core"] if "name_core" in out else df["name_core"].to_list()
            add.append(pl.Series(c, [x.replace(" ", "") for x in src], dtype=pl.Utf8))
        elif c == "script":
            add.append(pl.Series(c, [_script_of(x) for x in df["business_name"].to_list()], dtype=pl.Utf8))
        elif c == "name_is_domain":
            add.append(pl.Series(c, [bool(_DOMAIN_RE.search(x)) for x in df["business_name"].to_list()], dtype=pl.Boolean))
        else:
            add.append(pl.Series(c, out[c], dtype=dtypes.get(c, pl.Utf8)))
    return df.with_columns(add)


def generic_stats(s1_lazy, min_df_abs: int = 20, min_df_frac: float = 0.0005) -> dict:
    """{'dup_names': {(country, name_core) shared by >= 2 S1}, 'freq_tokens': {country: {token,...}}}.
    A token is frequent when its S1 document frequency >= max(min_df_abs, min_df_frac * n_S1_country)."""
    lf = _as_lazy(s1_lazy)
    cols = _cols(lf)
    if "name_core" in cols:
        base = lf.select(pl.col("country").cast(pl.Utf8).fill_null("?"), pl.col("name_core").fill_null("")).collect()
    else:
        raw = lf.select(pl.col("country").cast(pl.Utf8).fill_null("?"), pl.col("business_name").fill_null("")).collect()
        u = raw.unique()
        if u.height > 500_000:
            print(f"[error_report] computing name_core for {u.height:,} S1 names -- pass a normalized S1 frame",
                  file=sys.stderr)
        core = ensure_normalized(u.with_columns(pl.lit("").alias("business_address")))["name_core"]
        base = raw.join(u.with_columns(core.alias("name_core")), on=["country", "business_name"], how="left") \
                  .select("country", "name_core")
    dup = base.group_by(["country", "name_core"]).len().filter((pl.col("len") >= 2) & (pl.col("name_core") != ""))
    dup_names = set(dup.select("country", "name_core").iter_rows())
    ncty = dict(base.group_by("country").len().iter_rows())
    toks = (base.with_columns(pl.col("name_core").str.split(" ").list.unique().alias("t")).explode("t", empty_as_null=True)
                .filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
                .group_by(["country", "t"]).len())
    freq = {}
    for cty, tok, cnt in toks.iter_rows():
        if cnt >= max(min_df_abs, min_df_frac * ncty.get(cty, 0)):
            freq.setdefault(cty, set()).add(tok)
    return {"dup_names": dup_names, "freq_tokens": freq}


# =====================================================================================================
# error pairs + heuristic tags
# =====================================================================================================
def _sim(a: list, b: list, kind: str, workers: int = -1):
    """Element-wise similarity in [0,1] (rapidfuzz; token Jaccard fallback). Empty side -> 0."""
    try:
        from rapidfuzz import fuzz
        from rapidfuzz.process import cpdist
        scorer = {"sort": fuzz.token_sort_ratio, "set": fuzz.token_set_ratio}[kind]
        if not a:
            return []
        return [float(v) / 100.0 for v in cpdist(a, b, scorer=scorer, workers=workers)]
    except ImportError:
        out = []
        for x, y in zip(a, b):
            sx, sy = set(x.split()), set(y.split())
            if not sx or not sy:
                out.append(0.0)
            elif kind == "set":
                out.append(len(sx & sy) / min(len(sx), len(sy)))
            else:
                out.append(len(sx & sy) / len(sx | sy))
        return out


def error_pairs(scored: pl.DataFrame, links: pl.DataFrame, gt: pl.DataFrame, ids: pl.DataFrame) -> pl.DataFrame:
    """s1, cand, etype in {FP, FN_model, FN_blocking}, p (null for blocking misses), k (true size of s1)."""
    sc = scored.join(ids, on="s1", how="semi").select("s1", "cand", "p")
    lk = links.join(ids, on="s1", how="semi").select("s1", pl.col("mid").alias("cand"))
    g = gt.join(ids, on="s1", how="semi").select("s1", pl.col("mid").alias("cand"))
    fp = lk.join(g, on=["s1", "cand"], how="anti").with_columns(pl.lit("FP").alias("etype"))
    fn = g.join(lk, on=["s1", "cand"], how="anti")
    fn_m = fn.join(sc.select("s1", "cand"), on=["s1", "cand"], how="semi").with_columns(pl.lit("FN_model").alias("etype"))
    fn_b = fn.join(sc.select("s1", "cand"), on=["s1", "cand"], how="anti").with_columns(pl.lit("FN_blocking").alias("etype"))
    e = pl.concat([fp, fn_m, fn_b], how="vertical").join(sc, on=["s1", "cand"], how="left")
    k = g.group_by("s1").agg(pl.len().alias("k"))
    return e.join(k, on="s1", how="left").with_columns(pl.col("k").fill_null(0)).sort(["etype", "s1", "cand"])


def _norm_subset(lf: pl.LazyFrame, ids: pl.Series) -> pl.DataFrame:
    cols = _cols(lf)
    keep = ["entity_id"] + [c for c in _NEED if c in cols]
    sub = lf.select(keep).with_columns(pl.col("entity_id").cast(pl.Utf8)) \
            .join(pl.DataFrame({"entity_id": ids.unique()}).lazy(), on="entity_id", how="semi").collect()
    return ensure_normalized(sub.unique(subset=["entity_id"], keep="first"))


def _hn_digits(lst) -> set:
    out = set()
    for h in lst or []:
        d = _DIGITS_RE.findall(str(h))
        if d:
            out.add(d[0].lstrip("0") or "0")
    return out


def tag_error_pairs(err: pl.DataFrame, s1n, pooln, links: pl.DataFrame, gt_full: pl.DataFrame, gen: dict,
                    workers: int = -1) -> pl.DataFrame:
    """Attach display columns, similarities and one boolean column per TAG to error pairs (s1, cand, etype, p).
    Tags are heuristics, NOT exclusive; a pair can carry several or none."""
    if err.height == 0:
        return err.with_columns([pl.lit(False).alias(t) for t in TAGS] + [pl.lit("").alias("tags")])
    s1lf, plf = _as_lazy(s1n), _as_lazy(pooln)
    e = err.with_row_index("_rid")
    sib = e.select("_rid", "s1", "cand").join(links.select("s1", pl.col("mid").alias("sib")), on="s1", how="inner") \
           .filter(pl.col("sib") != pl.col("cand"))
    S = _norm_subset(s1lf, e["s1"])
    P = _norm_subset(plf, pl.concat([e["cand"], sib["sib"]]))
    pref = lambda df, p: df.rename({c: p + c for c in df.columns if c != "entity_id"})
    e = (e.join(pref(S, "s_"), left_on="s1", right_on="entity_id", how="left")
          .join(pref(P, "c_"), left_on="cand", right_on="entity_id", how="left"))
    for c in e.columns:
        if e[c].dtype == pl.Utf8:
            e = e.with_columns(pl.col(c).fill_null(""))
    sn, cn = e["s_name_core"].to_list(), e["c_name_core"].to_list()
    sa, ca = e["s_addr_norm"].to_list(), e["c_addr_norm"].to_list()
    nsort, nset, aset = _sim(sn, cn, "sort", workers), _sim(sn, cn, "set", workers), _sim(sa, ca, "set", workers)
    freq, dups = gen.get("freq_tokens", {}), gen.get("dup_names", set())
    rows = zip(e["s_country"].to_list(), sn, cn, e["s_name_norm"].to_list(), e["c_name_norm"].to_list(),
               e["s_name_compact"].to_list(), e["c_name_compact"].to_list(), sa, ca,
               e["s_house_nums"].to_list(), e["c_house_nums"].to_list(), e["s_city_key"].to_list(),
               e["c_city_key"].to_list(), e["c_business_name"].to_list(), e["c_script"].to_list(),
               e["c_name_is_domain"].to_list(), nsort, nset, aset)
    T = {t: [] for t in TAGS if t not in ("cross_duplicate", "competition_loss", "belongs_to_other_s1")}
    for (cty, s_core, c_core, s_nn, c_nn, s_cmp, c_cmp, s_ad, c_ad, s_hn, c_hn, s_city, c_city, c_raw, c_scr,
         c_dom, ns, nst, ast) in rows:
        st, ct = set(s_core.split()), set(c_core.split())
        same_city = bool(s_city) and (s_city == c_city or f" {s_city} " in f" {c_ad} ")
        generic = (cty, s_core) in dups or (bool(st) and st <= freq.get(cty, set()))
        T["generic_name_same_city"].append(ns >= 0.9 and same_city and generic)
        hs, hc = _hn_digits(s_hn), _hn_digits(c_hn)
        T["house_number_mismatch"].append(bool(hs) and bool(hc) and not (hs & hc) and (ns >= 0.8 or ast >= 0.6))
        glued = bool(s_cmp) and bool(c_cmp) and len(min(s_cmp, c_cmp, key=len)) >= 5 and \
            (s_cmp in c_cmp or c_cmp in s_cmp)
        T["alias_name"].append(ns < 0.5 and nst < 0.5 and not glued and ast >= 0.8)
        T["native_script"].append(c_scr == "indic" or bool(_INDIC_RE.search(c_raw)))
        T["domain_form"].append(bool(c_dom) or bool(_DOMAIN_RE.search(c_raw)))
        T["truncated_name"].append(bool(ct) and ct < st)
        T["empty_address"].append(not c_ad.strip())
        T["legal_form_only_diff"].append(bool(st) and st == ct and s_nn != c_nn)
    e = e.with_columns([pl.Series("name_sim", nsort), pl.Series("name_set_sim", nset), pl.Series("addr_sim", aset)]
                       + [pl.Series(t, v, dtype=pl.Boolean) for t, v in T.items()])
    # cross_duplicate: near-duplicate of another record linked to the same S1
    if sib.height:
        sb = (sib.join(P.select("entity_id", "name_core", "name_compact", "addr_norm"), left_on="sib", right_on="entity_id", how="left")
                 .join(e.select("_rid", "c_name_core", "c_name_compact", "c_addr_norm"), on="_rid", how="left")
                 .with_columns([pl.col(c).fill_null("") for c in ("name_core", "name_compact", "addr_norm",
                                                                   "c_name_core", "c_name_compact", "c_addr_norm")]))
        ns2 = _sim(sb["name_core"].to_list(), sb["c_name_core"].to_list(), "sort", workers)
        as2 = _sim(sb["addr_norm"].to_list(), sb["c_addr_norm"].to_list(), "set", workers)
        sb = sb.with_columns(pl.Series("_ns", ns2), pl.Series("_as", as2))
        dup = sb.filter(((pl.col("_ns") >= 0.9) | ((pl.col("name_compact") == pl.col("c_name_compact"))
                                                   & (pl.col("name_compact") != ""))) & (pl.col("_as") >= 0.8)) \
                .select("_rid").unique().with_columns(pl.lit(True).alias("_dup"))
        e = e.join(dup, on="_rid", how="left").with_columns(pl.col("_dup").fill_null(False).alias("cross_duplicate")) \
             .drop("_dup")
    else:
        e = e.with_columns(pl.lit(False).alias("cross_duplicate"))
    # competition_loss (FN): the true candidate was selected for a different S1
    other_link = links.select(pl.col("mid").alias("cand"), pl.col("s1").alias("_o"))
    comp = e.select("_rid", "s1", "cand").join(other_link, on="cand", how="inner") \
            .filter(pl.col("_o") != pl.col("s1")).select("_rid").unique().with_columns(pl.lit(True).alias("_c"))
    owner = gt_full.select(pl.col("mid").alias("cand"), pl.col("s1").alias("_g"))
    bel = e.select("_rid", "s1", "cand").join(owner, on="cand", how="inner") \
           .filter(pl.col("_g") != pl.col("s1")).select("_rid").unique().with_columns(pl.lit(True).alias("_b"))
    e = (e.join(comp, on="_rid", how="left").join(bel, on="_rid", how="left")
          .with_columns((pl.col("_c").fill_null(False) & pl.col("etype").str.starts_with("FN")).alias("competition_loss"),
                        (pl.col("_b").fill_null(False) & (pl.col("etype") == "FP")).alias("belongs_to_other_s1"))
          .drop("_c", "_b"))
    e = e.with_columns(pl.concat_list([pl.when(pl.col(t)).then(pl.lit(t)).otherwise(None) for t in TAGS])
                         .list.drop_nulls().list.join(",").alias("tags"))
    return e.sort("_rid").drop("_rid")


# =====================================================================================================
# the report
# =====================================================================================================
def build_report(scored, links, gt, s1, pool, s1_ids=None, n_examples: int = 4, max_tag_pairs: int = 30_000,
                 seed: int = 0, workers: int = -1, title: str = "") -> dict:
    """Compute every table of the error report. Returns a dict (see render_markdown for the layout);
    rep['per_s1'] and rep['tagged'] are frames you can slice further."""
    sc = load_scored(scored)
    lk = load_long(links)
    g = load_long(gt)
    s1lf = _as_lazy(s1)
    per = per_s1_table(sc, lk, g, s1lf, s1_ids)
    ids = per.select("s1")
    n = per.height
    rep = {"title": title, "meta": {"generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "n_s1": n,
                                    "n_scored_pairs": sc.join(ids, on="s1", how="semi").height,
                                    "n_links": lk.join(ids, on="s1", how="semi").height, "warnings": []}}
    W = rep["meta"]["warnings"]
    mf = metric.macro_f05(lk, g, ids)
    assert abs(mf - float(per["f"].mean())) < 1e-9, "per-S1 table disagrees with metric.macro_f05"
    no_cand = float((per["n_cand"] == 0).mean())
    if no_cand > 0.2:
        W.append(f"{no_cand:.1%} of evaluated S1 have no scored candidate -- if the scored frame covers one fold, "
                 f"pass s1_ids for that fold (otherwise those S1 count as empty predictions)")
    lk_ev = lk.join(ids, on="s1", how="semi")
    not_cand = lk_ev.join(sc.select("s1", pl.col("cand").alias("mid")), on=["s1", "mid"], how="anti").height
    if not_cand:
        W.append(f"{not_cand:,} links are NOT in the scored candidates (audit rule: matches must be a subset of candidates)")
    multi = lk.group_by("mid").len().filter(pl.col("len") > 1).height
    if multi:
        W.append(f"{multi:,} S2/S3 ids are linked to more than one S1 (exclusivity violated: each id has <= 1 owner)")
    if "label" in sc.columns:
        chk = sc.join(g.select("s1", pl.col("mid").alias("cand"), pl.lit(1).cast(pl.Int8).alias("_y")),
                      on=["s1", "cand"], how="left").with_columns(pl.col("_y").fill_null(0))
        bad = chk.filter(pl.col("label") != pl.col("_y")).height
        if bad:
            W.append(f"{bad:,} scored rows have label != ground truth (stale labels or wrong gt passed)")
    ksum, msum, tpsum = int(per["k"].sum()), int(per["m"].sum()), int(per["tp"].sum())
    kcsum = int(per["kc"].sum())
    single = per.filter(pl.col("k") == 0)
    rep["headline"] = {
        "macro_f": mf, "micro_p": tpsum / msum if msum else float("nan"),
        "micro_r": tpsum / ksum if ksum else float("nan"),
        "mean_true_k": ksum / max(n, 1), "mean_pred_size": msum / max(n, 1),
        "empty_pred_rate": float((per["m"] == 0).mean()), "singleton_rate": single.height / max(n, 1),
        "singleton_empty_acc": float((single["m"] == 0).mean()) if single.height else float("nan"),
        "matched_empty_rate": float((per.filter(pl.col("k") > 0)["m"] == 0).mean()) if n > single.height else float("nan"),
        "s1_no_candidate_rate": no_cand, "cands_per_s1": float(per["n_cand"].mean()),
        "blocking_pair_recall": kcsum / ksum if ksum else float("nan"),
        "blocking_ceiling_f": float(per["f_ceiling"].mean()), "oracle_cut_f": float(per["f_oracle_cut"].mean()),
    }
    h = rep["headline"]
    rep["ladder"] = pl.DataFrame([
        ("blocking (1 - ceiling)", 1 - h["blocking_ceiling_f"], "true links never generated as candidates"),
        ("ranking (ceiling - oracle cut)", h["blocking_ceiling_f"] - h["oracle_cut_f"],
         "true candidates ranked below false ones: features / model"),
        ("decision (oracle cut - actual)", h["oracle_cut_f"] - mf,
         "wrong list length given the ranking: calibration / exclusivity / expected-F rule"),
    ], schema=["layer", "points", "fix_in"], orient="row")
    rep["loss"], rep["loss_rollup"] = loss_decomposition(per)
    rep["by_country"] = _group_table(per, "country")
    rep["by_k"] = _group_table(per, "k_bucket", K_ORDER)
    rep["by_pred_size"] = _group_table(per, "m_bucket", K_ORDER)
    rep["by_country_k"] = _group_table(per, ["country", "k_bucket"])
    rep["top_buckets"] = rep["by_country_k"].sort("loss_pts", descending=True).head(10)
    # pair level
    g_ev = g.join(ids, on="s1", how="semi")
    cand_pairs = sc.join(ids, on="s1", how="semi").select("s1", pl.col("cand").alias("mid"), "p")
    gl = (g_ev.join(cand_pairs.select("s1", "mid", pl.lit(True).alias("in_cand")), on=["s1", "mid"], how="left")
              .join(lk_ev.select("s1", "mid", pl.lit(True).alias("in_link")), on=["s1", "mid"], how="left")
              .with_columns(pl.col("in_cand").fill_null(False), pl.col("in_link").fill_null(False),
                            pl.col("mid").str.slice(0, 2).alias("source"))
              .join(per.select("s1", "country"), on="s1", how="left"))
    lsrc = lk_ev.join(g_ev.select("s1", "mid", pl.lit(True).alias("ok")), on=["s1", "mid"], how="left") \
                .with_columns(pl.col("ok").fill_null(False), pl.col("mid").str.slice(0, 2).alias("source")) \
                .join(per.select("s1", "country"), on="s1", how="left")

    def pair_tab(by):
        a = gl.group_by(by).agg(pl.len().alias("true_links"), pl.col("in_cand").mean().alias("blocking_recall"),
                                pl.col("in_link").mean().alias("final_recall"))
        b = lsrc.group_by(by).agg(pl.len().alias("pred_links"), pl.col("ok").mean().alias("link_precision"))
        return a.join(b, on=by, how="full", coalesce=True).sort(by)

    rep["pairs_by_country"] = pair_tab("country")
    rep["pairs_by_source"] = pair_tab("source")
    err = error_pairs(sc, lk, g, ids)
    fp = err.filter(pl.col("etype") == "FP")
    fp_owner = fp.join(g.select(pl.col("mid").alias("cand"), pl.lit(True).alias("_own")).unique(), on="cand", how="left")
    fnm = err.filter(pl.col("etype") == "FN_model")
    fnm_c = fnm.join(lk.select(pl.col("mid").alias("cand"), pl.lit(True).alias("_oth")).unique(), on="cand", how="left")
    rep["error_counts"] = pl.DataFrame([
        ("FP", fp.height, "on singleton S1", int((fp["k"] == 0).sum()),
         "cand is a true link of another S1", int(fp_owner["_own"].fill_null(False).sum())),
        ("FN_model", fnm.height, "cand linked to another S1", int(fnm_c["_oth"].fill_null(False).sum()),
         "cand linked nowhere", int((~fnm_c["_oth"].fill_null(False)).sum())),
        ("FN_blocking", err.filter(pl.col("etype") == "FN_blocking").height, "-", 0, "-", 0),
    ], schema=["etype", "n", "split_a", "n_a", "split_b", "n_b"], orient="row")
    pb = err.filter(pl.col("etype") != "FN_blocking").with_columns(
        (pl.col("p").clip(0, 0.999999) * 10).floor().cast(pl.Int64).alias("_b"))
    grid = pl.DataFrame({"_b": list(range(10))}, schema={"_b": pl.Int64})
    for et in ("FP", "FN_model"):
        grid = grid.join(pb.filter(pl.col("etype") == et).group_by("_b").agg(pl.len().alias(et)), on="_b", how="left")
    rep["p_bins_errors"] = grid.with_columns([pl.col(et).fill_null(0) for et in ("FP", "FN_model")])                                .select((pl.col("_b") / 10).alias("p_from"), "FP", "FN_model")
    cal = (cand_pairs.join(g_ev.select("s1", "mid", pl.lit(1).alias("y")), on=["s1", "mid"], how="left")
                     .with_columns(pl.col("y").fill_null(0),
                                   (pl.col("p").clip(0, 0.999999) * 10).floor().cast(pl.Int64).alias("_b")))
    ctab = cal.group_by("_b").agg(pl.len().alias("n"), pl.col("p").mean().alias("mean_p"),
                                  pl.col("y").mean().alias("obs_rate")).sort("_b")
    ctab = ctab.with_columns((pl.col("_b") / 10).alias("p_from"), (pl.col("obs_rate") - pl.col("mean_p")).alias("gap")) \
               .select("p_from", "n", "mean_p", "obs_rate", "gap")
    rep["calibration"] = ctab
    rep["ece"] = float((ctab["n"] * ctab["gap"].abs()).sum() / max(int(ctab["n"].sum()), 1)) if ctab.height else float("nan")
    # tags (deterministic sample per error type)
    # correct links (TP) are tagged too: a tag only points at a cause when it is over-represented in errors
    tps = (lk_ev.join(g_ev, on=["s1", "mid"], how="semi").select("s1", pl.col("mid").alias("cand"))
                .join(sc.select("s1", "cand", "p"), on=["s1", "cand"], how="left")
                .join(per.select("s1", "k"), on="s1", how="left")
                .with_columns(pl.lit("TP").alias("etype")).select(err.columns).sort(["s1", "cand"]))
    parts, sampled = [], {}
    for et in ETYPES + ["TP"]:
        x = tps if et == "TP" else err.filter(pl.col("etype") == et)
        sampled[et] = (x.height, min(x.height, max_tag_pairs))
        parts.append(x.sample(n=max_tag_pairs, seed=seed) if x.height > max_tag_pairs else x)
    sub = pl.concat(parts, how="vertical_relaxed")
    gen = generic_stats(s1lf)
    tagged = tag_error_pairs(sub, s1lf, pool, lk, g, gen, workers)
    rep["tagged"] = tagged
    rows = []
    xtp = tagged.filter(pl.col("etype") == "TP")
    for t in TAGS + ["(no tag)"]:
        flag = (lambda x: x["tags"] == "") if t == "(no tag)" else (lambda x, t=t: x[t])
        tp_share = float(flag(xtp).mean()) if xtp.height else float("nan")
        row = [t, tp_share]
        for et in ETYPES:
            x = tagged.filter(pl.col("etype") == et)
            c = int(flag(x).sum()) if x.height else 0
            sh = c / x.height if x.height else float("nan")
            lift = (sh / tp_share if tp_share > 0 else (float("inf") if sh > 0 else float("nan")))                 if not math.isnan(sh) and not math.isnan(tp_share) else float("nan")
            row += [c, sh, lift]
        rows.append(row)
    rep["tag_counts"] = pl.DataFrame(rows, schema=["tag", "TP_share"] + [f"{et}{s}" for et in ETYPES
                                                                         for s in ("_n", "_share", "_lift")],
                                     orient="row")
    rep["tag_sampled"] = sampled
    ex = {}
    for et in ETYPES:
        x = tagged.filter(pl.col("etype") == et)
        if not x.height:
            continue
        order = sorted(TAGS, key=lambda t: -int(x[t].sum()))
        blocks = []
        for t in [t for t in order if int(x[t].sum()) > 0] + ["(no tag)"]:
            y = x.filter(pl.col(t)) if t != "(no tag)" else x.filter(pl.col("tags") == "")
            if y.height:
                blocks.append((t, y.sample(n=min(n_examples, y.height), seed=seed).sort(["s1", "cand"])))
        ex[et] = blocks
    rep["examples"] = ex
    rep["per_s1"] = per
    return rep


# ----------------------------------------------------------------------------------------------------- rendering
def _fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Y" if v else ""
    if isinstance(v, float):
        return "-" if math.isnan(v) else ("inf" if math.isinf(v) else f"{v:.{nd}f}")
    if isinstance(v, int):
        return f"{v:,}"
    s = str(v).replace("|", "\\|").replace("\n", " ").replace("\t", " ")
    return s


def md_table(df: pl.DataFrame, nd: int = 4) -> str:
    if df is None or df.height == 0:
        return "_(empty)_\n"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v, nd) for v in row) + " |")
    return "\n".join(lines) + "\n"


def _cut(s: str, n: int = 55) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "~"


def render_markdown(rep: dict, title: str = "") -> str:
    m, h = rep["meta"], rep["headline"]
    title = title or rep.get("title") or "validation run"
    out = [f"# Error report - {title}", "",
           f"generated {m['generated']} | evaluated S1: {m['n_s1']:,} | scored pairs: {m['n_scored_pairs']:,} | "
           f"links: {m['n_links']:,}", ""]
    for w in m["warnings"]:
        out.append(f"> **WARNING:** {w}")
    out += ["", "## 1. Headline", "",
            md_table(pl.DataFrame({"metric": list(h.keys()), "value": [float(v) for v in h.values()]}), 5),
            "## 2. Loss ladder - which stage owns the lost points", "",
            md_table(rep["ladder"], 5),
            "## 3. Loss decomposition (components sum to 1 - macro F0.5)", "",
            md_table(rep["loss"], 5), md_table(rep["loss_rollup"], 5),
            "## 4. Where the points are", "",
            "loss_pts = share x (1 - F) of the bucket = the most a perfect fix of that bucket can add; "
            "model_gap_pts = points recoverable without touching blocking; blocking_pts = points only blocking can recover.",
            "", "### Top country x true-k buckets by loss", "", md_table(rep["top_buckets"]),
            "### By country", "", md_table(rep["by_country"]),
            "### By true match count k", "", md_table(rep["by_k"]),
            "### By predicted list size", "", md_table(rep["by_pred_size"]),
            "### By country x k", "", md_table(rep["by_country_k"]),
            "## 5. Pair-level errors", "",
            md_table(rep["error_counts"]), "### Recall / precision by country", "", md_table(rep["pairs_by_country"]),
            "### By pool source", "", md_table(rep["pairs_by_source"]),
            "### p of the error pairs (FP = selected but false; FN_model = true but not selected)", "",
            md_table(rep["p_bins_errors"]),
            f"## 6. Calibration of p on all scored pairs (ECE = {rep['ece']:.4f})", "", md_table(rep["calibration"]),
            "## 7. Error tags (heuristic, non-exclusive)", ""]
    samp = ", ".join(f"{et}: {s:,} of {n_:,}" for et, (n_, s) in rep["tag_sampled"].items())
    out += [f"tagged pairs (deterministic sample): {samp}", "",
            "lift = share among these errors / share among correct links (TP). Act on tags with high share AND "
            "lift > ~1.5; a tag that is just as common among TPs is noise, not a cause.", "",
            md_table(rep["tag_counts"], 3), ""]
    out += [f"- `{t}`: {TAG_HELP[t]}" for t in TAGS] + ["", "## 8. Examples by error type and tag", ""]
    for et, blocks in rep["examples"].items():
        out += [f"### {et} - {ETYPE_HELP[et]}", ""]
        for t, df in blocks:
            show = df.select(
                pl.col("s1"), pl.concat_str([pl.col("s_business_name").map_elements(_cut, return_dtype=pl.Utf8),
                                             pl.col("s_business_address").map_elements(_cut, return_dtype=pl.Utf8)],
                                            separator=" / ").alias("S1 name / address"),
                pl.col("cand"), pl.concat_str([pl.col("c_business_name").map_elements(_cut, return_dtype=pl.Utf8),
                                               pl.col("c_business_address").map_elements(_cut, return_dtype=pl.Utf8)],
                                              separator=" / ").alias("candidate name / address"),
                pl.col("p"), pl.col("name_sim"), pl.col("addr_sim"), pl.col("tags"))
            out += [f"**{t}**", "", md_table(show, 3)]
    return "\n".join(out) + "\n"


def write_report(out_md: str, save_frames: bool = True, **kw) -> dict:
    rep = build_report(**kw)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_markdown(rep, kw.get("title", "")))
    if save_frames:
        stem = os.path.splitext(out_md)[0]
        rep["per_s1"].write_parquet(stem + "_per_s1.parquet")
        rep["tagged"].write_parquet(stem + "_tagged.parquet")
    return rep


# =====================================================================================================
# label-free checks: shift (France / test vs OOF), PSI, manual inspection sheet
# =====================================================================================================
def unlabeled_stats(scored, links, s1_meta, split: str) -> pl.DataFrame:
    """Per-country statistics that need NO labels: candidate volume, max-p distribution, predicted list
    sizes, share of uncertain links, candidate conflicts. Compare test/France rows against OOF rows."""
    sc, lk = load_scored(scored), load_long(links)
    meta = _as_lazy(s1_meta)
    idc = "entity_id" if "entity_id" in _cols(meta) else "s1"
    meta = meta.select(pl.col(idc).cast(pl.Utf8).alias("s1"), pl.col("country").cast(pl.Utf8).fill_null("?")).collect()
    per = (meta.join(sc.group_by("s1").agg(pl.len().alias("n_cand"), pl.col("p").max().alias("max_p")), on="s1", how="left")
               .join(lk.group_by("s1").agg(pl.len().alias("m")), on="s1", how="left")
               .with_columns(pl.col("n_cand").fill_null(0), pl.col("m").fill_null(0), pl.col("max_p").fill_null(0.0)))
    selp = lk.join(sc.select("s1", pl.col("cand").alias("mid"), "p"), on=["s1", "mid"], how="left") \
             .join(meta, on="s1", how="left")
    hot = sc.filter(pl.col("p") >= 0.5).join(meta, on="s1", how="left")
    conf = hot.group_by(["country", "cand"]).len().group_by("country").agg(
        (pl.col("len") > 1).mean().alias("conflict_rate"))
    a = per.group_by("country").agg(
        pl.len().alias("n_s1"), pl.col("n_cand").mean().alias("cands_per_s1"),
        pl.col("n_cand").quantile(0.9).alias("cands_p90"), (pl.col("n_cand") == 0).mean().alias("no_cand_rate"),
        pl.col("max_p").mean().alias("max_p_mean"), pl.col("max_p").median().alias("max_p_p50"),
        (pl.col("max_p") >= 0.5).mean().alias("maxp_ge_0.5"), (pl.col("max_p") >= 0.9).mean().alias("maxp_ge_0.9"),
        pl.col("m").mean().alias("links_per_s1"), (pl.col("m") == 0).mean().alias("empty_rate"),
        (pl.col("m") >= 5).mean().alias("m_ge_5_rate"))
    b = selp.group_by("country").agg(pl.col("p").mean().alias("sel_p_mean"),
                                     (pl.col("p") < 0.7).mean().alias("sel_p_lt_0.7"))
    return (a.join(b, on="country", how="left").join(conf, on="country", how="left")
             .with_columns(pl.lit(split).alias("split")).sort("country")
             .select(["split", "country"] + [c for c in a.columns if c != "country"] + ["sel_p_mean", "sel_p_lt_0.7",
                                                                                         "conflict_rate"]))


def psi(ref_values, cur_values, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index with reference-quantile bins. <0.1 stable, 0.1-0.25 moderate, >0.25 shift."""
    import numpy as np
    r = np.asarray(ref_values, dtype=np.float64)
    c = np.asarray(cur_values, dtype=np.float64)
    r, c = r[np.isfinite(r)], c[np.isfinite(c)]
    if len(r) == 0 or len(c) == 0:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)[1:-1]))
    rb = np.bincount(np.searchsorted(edges, r, side="right"), minlength=len(edges) + 1) / len(r)
    cb = np.bincount(np.searchsorted(edges, c, side="right"), minlength=len(edges) + 1) / len(c)
    rb, cb = np.maximum(rb, eps), np.maximum(cb, eps)
    return float(((cb - rb) * np.log(cb / rb)).sum())


def _id_sample(lf: pl.LazyFrame, frac: float) -> pl.LazyFrame:
    # version-stable deterministic sample by the numeric id suffix (polars hash is not stable across versions)
    if frac >= 1:
        return lf
    q = max(1, int(round(frac * 1000)))
    return lf.filter(pl.col("s1").cast(pl.Utf8).str.extract(r"(\d+)\s*$", 1).cast(pl.UInt64, strict=False)
                     .fill_null(0) % 1000 < q)


def feature_psi(features_ref, features_cur, s1_meta_cur, feature_cols=None, frac: float = 0.1,
                s1_meta_ref=None) -> pl.DataFrame:
    """PSI of every feature: reference (all rows, sampled) vs each country of the current frame."""
    fr, fc = _as_lazy(features_ref), _as_lazy(features_cur)
    cols = feature_cols or [c for c in _cols(fc) if c not in ("s1", "cand", "label", "p", "fold")]
    R = _id_sample(fr, frac).select(cols).collect()
    mc = _as_lazy(s1_meta_cur)
    idc = "entity_id" if "entity_id" in _cols(mc) else "s1"
    C = _id_sample(fc, frac).select(["s1"] + cols).collect() \
        .join(mc.select(pl.col(idc).cast(pl.Utf8).alias("s1"), pl.col("country").cast(pl.Utf8)).collect(),
              on="s1", how="left")
    rows = []
    for cty in sorted(C["country"].fill_null("?").unique().to_list()):
        sub = C.filter(pl.col("country").fill_null("?") == cty)
        for col in cols:
            rows.append((cty, col, psi(R[col].to_numpy(), sub[col].to_numpy()),
                         float(R[col].mean()), float(sub[col].mean())))
    return pl.DataFrame(rows, schema=["country", "feature", "psi", "ref_mean", "cur_mean"], orient="row") \
             .sort(["country", "psi"], descending=[False, True])


def shift_report(ref: dict, cur: dict, features_ref=None, features_cur=None, feature_cols=None,
                 frac: float = 0.1, rel_tol: float = 0.15) -> tuple:
    """ref/cur: {'scored':..., 'links':..., 's1':...}. Returns (markdown, {'stats', 'flags', 'psi'})."""
    st = pl.concat([unlabeled_stats(ref["scored"], ref["links"], ref["s1"], "ref"),
                    unlabeled_stats(cur["scored"], cur["links"], cur["s1"], "cur")], how="vertical_relaxed")
    watch = ["cands_per_s1", "max_p_mean", "maxp_ge_0.5", "links_per_s1", "empty_rate", "sel_p_lt_0.7", "conflict_rate"]
    refm = st.filter(pl.col("split") == "ref")
    pooled = {c: float((refm[c].fill_null(0) * refm["n_s1"]).sum() / refm["n_s1"].sum()) for c in watch}
    by_cty = {r["country"]: r for r in refm.iter_rows(named=True)}
    flags = []
    for row in st.filter(pl.col("split") == "cur").iter_rows(named=True):
        # a country seen in ref is compared with itself; an unseen one (France) with the pooled ref
        basis = "same country" if row["country"] in by_cty else "pooled ref"
        for c in watch:
            b = by_cty[row["country"]][c] if basis == "same country" else pooled[c]
            v = row[c]
            if v is None or b is None or (isinstance(v, float) and math.isnan(v)):
                continue
            rel = (v - b) / b if b else (float("inf") if v else 0.0)
            if abs(rel) > rel_tol:
                flags.append((row["country"], c, basis, float(b), float(v), rel))
    fl = pl.DataFrame(flags, schema=["country", "stat", "basis", "ref", "cur", "rel_diff"], orient="row")
    ps = None
    if features_ref is not None and features_cur is not None:
        ps = feature_psi(features_ref, features_cur, cur["s1"], feature_cols, frac)
    out = ["# Shift report (no labels needed)", "",
           f"generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}; ref = labelled validation run (OOF), "
           f"cur = run to check (test / France). Flag = |relative diff| > {rel_tol:.0%} vs the same country in ref, "
           f"or vs the pooled ref for a country ref does not have.", "",
           "## Per-country statistics", "", md_table(st), "## Flags", "", md_table(fl)]
    if ps is not None:
        out += ["## Feature PSI (top 15 per country; > 0.25 = strong shift)", "",
                md_table(ps.group_by("country", maintain_order=True).head(15))]
    return "\n".join(out) + "\n", {"stats": st, "flags": fl, "psi": ps}


def inspection_sheet(scored, links, s1, pool, country: str = "France", n: int = 50, top: int = 3,
                     seed: int = 0, strata=(("random", 0.5), ("uncertain", 0.25), ("empty", 0.25))) -> str:
    """Markdown sheet for MANUAL review of `n` S1 of one country with their top candidates. Strata:
    random S1, uncertain (max p in [0.3, 0.7)), empty prediction. Fill the verdict column by hand."""
    sc, lk = load_scored(scored), load_long(links)
    S = _as_lazy(s1)
    idc = "entity_id" if "entity_id" in _cols(S) else "s1"
    S = S.select(pl.col(idc).cast(pl.Utf8).alias("s1"), pl.col("business_name").fill_null(""),
                 pl.col("business_address").fill_null(""), pl.col("country").cast(pl.Utf8)) \
         .filter(pl.col("country") == country).collect().sort("s1")
    mx = sc.group_by("s1").agg(pl.col("p").max().alias("max_p"))
    ms = lk.group_by("s1").agg(pl.len().alias("m"))
    S = S.join(mx, on="s1", how="left").join(ms, on="s1", how="left") \
         .with_columns(pl.col("max_p").fill_null(0.0), pl.col("m").fill_null(0))
    picked, chosen = [], set()

    def _rest(df):
        return df.join(pl.DataFrame({"s1": sorted(chosen)}, schema={"s1": pl.Utf8}), on="s1", how="anti")

    for name, share in list(strata) + [("random", 1.0)]:          # the last pass tops up to n
        pool_s = _rest({"random": S, "uncertain": S.filter((pl.col("max_p") >= 0.3) & (pl.col("max_p") < 0.7)),
                        "empty": S.filter(pl.col("m") == 0)}[name])
        want = min(pool_s.height, max(1, int(round(n * share))), n - len(chosen))
        if want > 0:
            x = pool_s.sample(n=want, seed=seed).with_columns(pl.lit(name).alias("stratum"))
            chosen |= set(x["s1"].to_list())
            picked.append(x)
    if not picked:
        return f"# Inspection sheet - {country}\n\n_(no S1 for this country)_\n"
    P = pl.concat(picked, how="vertical").head(n)
    top_c = (sc.join(P.select("s1"), on="s1", how="semi").sort(["s1", "p"], descending=[False, True])
               .group_by("s1", maintain_order=True).head(top))
    sel = lk.select("s1", pl.col("mid").alias("cand"), pl.lit(True).alias("selected"))
    top_c = top_c.join(sel, on=["s1", "cand"], how="left").with_columns(pl.col("selected").fill_null(False))
    # selected links outside the top-k are shown too: the reviewer must see every predicted link
    extra = sel.join(P.select("s1"), on="s1", how="semi").join(top_c.select("s1", "cand"), on=["s1", "cand"], how="anti") \
               .join(sc, on=["s1", "cand"], how="left")
    rows = pl.concat([top_c.select("s1", "cand", "p", "selected"), extra.select("s1", "cand", "p", "selected")],
                     how="vertical_relaxed")
    PL = _as_lazy(pool)
    pc = PL.select(pl.col("entity_id").cast(pl.Utf8).alias("cand"), pl.col("business_name").fill_null(""),
                   pl.col("business_address").fill_null("")) \
           .join(rows.select("cand").unique().lazy(), on="cand", how="semi").collect()
    rows = rows.join(pc, on="cand", how="left")
    out = [f"# Inspection sheet - {country} ({P.height} S1, top {top} candidates each)", "",
           "Mark each candidate: OK (same business) / FP (selected but different) / FN (not selected but same) / ?.",
           "Count FP and FN per stratum; a France FP rate clearly above the OOF FP rate => raise empty_bias / "
           "threshold for all countries or fix the France-specific normalization gap you see.", ""]
    for r in P.iter_rows(named=True):
        out.append(f"### {r['s1']} [{r['stratum']}] max_p={r['max_p']:.3f} predicted={r['m']}")
        out.append(f"S1: **{_fmt(r['business_name'])}** / {_fmt(r['business_address'])}")
        out.append("")
        sub = rows.filter(pl.col("s1") == r["s1"]).sort("p", descending=True, nulls_last=True)
        show = sub.select("cand", "p", "selected", pl.col("business_name").alias("name"),
                          pl.col("business_address").alias("address"), pl.lit("").alias("verdict"))
        out.append(md_table(show, 3) if show.height else "_(no candidates)_\n")
    return "\n".join(out) + "\n"


# =====================================================================================================
# CLI
# =====================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Error analysis for the ER pipeline (report / shift / inspect)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("report", help="labelled error report on a validation run")
    for k in ("scored", "links", "gt", "s1", "pool", "out"):
        a.add_argument(f"--{k}", required=True)
    a.add_argument("--ids", help="file with evaluated S1 ids (validation fold); default: all S1 in --s1")
    a.add_argument("--title", default="")
    a.add_argument("--examples", type=int, default=4)
    a.add_argument("--max-tag-pairs", type=int, default=30_000)
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("shift", help="label-free comparison of a run (test/France) against a reference run")
    for k in ("ref-scored", "ref-links", "ref-s1", "cur-scored", "cur-links", "cur-s1", "out"):
        b.add_argument(f"--{k}", required=True)
    b.add_argument("--ref-features")
    b.add_argument("--cur-features")
    b.add_argument("--frac", type=float, default=0.1)
    c = sub.add_parser("inspect", help="manual inspection sheet for one country")
    for k in ("scored", "links", "s1", "pool", "out"):
        c.add_argument(f"--{k}", required=True)
    c.add_argument("--country", default="France")
    c.add_argument("--n", type=int, default=50)
    c.add_argument("--top", type=int, default=3)
    c.add_argument("--seed", type=int, default=0)
    x = ap.parse_args(argv)
    if x.cmd == "report":
        rep = write_report(x.out, scored=x.scored, links=x.links, gt=x.gt, s1=x.s1, pool=x.pool, s1_ids=x.ids,
                           n_examples=x.examples, max_tag_pairs=x.max_tag_pairs, seed=x.seed, title=x.title)
        h = rep["headline"]
        print(f"macro F0.5={h['macro_f']:.5f} ceiling={h['blocking_ceiling_f']:.5f} oracle_cut={h['oracle_cut_f']:.5f}"
              f" -> {x.out}")
    elif x.cmd == "shift":
        md, _ = shift_report({"scored": x.ref_scored, "links": x.ref_links, "s1": x.ref_s1},
                             {"scored": x.cur_scored, "links": x.cur_links, "s1": x.cur_s1},
                             x.ref_features, x.cur_features, frac=x.frac)
        _write(x.out, md)
    else:
        _write(x.out, inspection_sheet(x.scored, x.links, x.s1, x.pool, x.country, x.n, x.top, x.seed))
    return 0


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
