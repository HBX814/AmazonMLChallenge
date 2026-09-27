#!/usr/bin/env python
"""
metric.py -- Reference implementation of the Amazon ML Challenge 2026 metric
("Business Entity Resolution"): per-Source-1 F_beta (beta = 0.5), macro-averaged
over ALL evaluated S1 entities (singletons included).

Per S1 entity with truth set T and predicted set P:
    |T| = 0 and |P| = 0            -> 1.0
    |P ∩ T| = 0 (any other case)   -> 0.0   (covers: T empty & P non-empty,
                                              T non-empty & P empty,
                                              both non-empty but disjoint)
    otherwise                      -> (1+b^2) * tp / (b^2 * |T| + |P|)
which is algebraically identical to the problem statement's
    F_0.5 = 1.25 * Prec * Rec / (0.25 * Prec + Rec).

Sets, not lists: duplicate IDs inside one list are collapsed (the official
scorer REJECTS such files -- we only warn, so you can still score drafts).
S1 ids in the evaluation set that are missing from the predictions are scored
as an EMPTY prediction (the official scorer rejects a file with missing rows --
we warn). Prediction rows for S1 ids outside the evaluation set are ignored.

Two engines that give identical numbers (tested):
    * engine="polars"  -- vectorised; ~2.2M S1 in a few seconds
    * engine="python"  -- plain dict/set loop; simplest to audit

CLI examples
------------
  python metric.py --pred out/val_pred.tsv --gt dataset/train/train_ground_truth.tsv
  python metric.py --pred p.tsv --gt gt.tsv --ids val_ids.txt \
         --s1 dataset/train/train_source1.tsv --by-country --by-k --by-pred-size \
         --json scores.json
  python metric.py --selftest

--ids file: one S1 id per line (a header line 'source1_entity_id' is skipped),
or any TSV whose FIRST column holds S1 ids.

Library use
-----------
  from metric import f_beta, score_dicts, score_files
  f_beta({"a","b","c"}, {"a","c"})            # 0.7142857...
  res = score_files("pred.tsv", "gt.tsv", s1_path="train_source1.tsv",
                    by_country=True, by_k=True)
  res["macro_f"], res["by_country"], res["by_k"]

Contract API (CONTRACT.md names; long frames with columns s1, mid):
  macro_f05(pred_long, gt_long, s1_ids) -> float
  f05_breakdown(pred_long, gt_long, s1_meta) -> polars frame (ALL / country / k / country|k rows)
  per_s1_scores(pred_long, gt_long, s1_ids) -> frame s1, k, m, tp, f
  fold_of(s1_ids, n_folds=5, seed=0) -> numpy fold ids (stable GroupKFold-by-S1 assignment)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

BETA = 0.5
K_BUCKETS = ["0", "1", "2", "3", "4", "5", "6", "7+"]


# ----------------------------------------------------------------------------
# core metric
# ----------------------------------------------------------------------------
def f_beta_counts(tp: int, n_true: int, n_pred: int, beta: float = BETA) -> float:
    """F_beta for one entity from counts, with the challenge's edge cases."""
    if n_true == 0 and n_pred == 0:
        return 1.0
    if tp == 0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * tp / (b2 * n_true + n_pred)


def f_beta(pred: Iterable[str], truth: Iterable[str], beta: float = BETA) -> float:
    """F_beta for one S1 entity. `pred` and `truth` are iterables of S2/S3 ids."""
    p, t = set(pred), set(truth)
    return f_beta_counts(len(p & t), len(t), len(p), beta)


def k_bucket(k: int) -> str:
    return str(k) if k < 7 else "7+"


def pred_bucket(m: int) -> str:
    return str(m) if m < 7 else "7+"


# ----------------------------------------------------------------------------
# I/O helpers (pure python)
# ----------------------------------------------------------------------------
def parse_id_list(field: str) -> List[str]:
    field = field.strip()
    if not field:
        return []
    return [x.strip() for x in field.split(",") if x.strip()]


def load_id_list_tsv(path: str, strict_dup_rows: bool = True) -> Dict[str, List[str]]:
    """Load a 2-column TSV (source1_entity_id <TAB> comma-separated ids).

    Header is skipped. Quoting is NOT interpreted. Returns {s1: [ids...]} with
    list order preserved (duplicates kept so callers can count them).
    """
    out: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        if "\t" not in header:
            raise ValueError(f"{path}: header has no TAB -- not a TSV? header={header!r}")
        for ln, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            s1, _, rest = line.partition("\t")
            s1 = s1.strip()
            if s1 in out:
                if strict_dup_rows:
                    raise ValueError(f"{path}: duplicate S1 row {s1!r} at line {ln} (scorer rejects this)")
                out[s1].extend(parse_id_list(rest))
            else:
                out[s1] = parse_id_list(rest)
    return out


def load_ids_file(path: str) -> List[str]:
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\r\n").split("\t", 1)[0].strip()
            if not tok or tok in ("source1_entity_id", "entity_id", "s1"):
                continue
            ids.append(tok)
    return ids


def load_country_map(s1_path: str) -> Dict[str, str]:
    """entity_id -> country from a source1 TSV (columns entity_id, business_name,
    business_address, country). Reads only the first and last field."""
    cmap = {}
    with open(s1_path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        ci = header.index("country")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > ci:
                cmap[parts[0].strip()] = parts[ci].strip()
    return cmap


# ----------------------------------------------------------------------------
# scoring (pure python engine)
# ----------------------------------------------------------------------------
LOSS_KEYS = ("singleton_false_merge", "nonsingleton_empty_pred", "nonsingleton_all_wrong", "partial")


def _finish(n, fsum, tp_sum, k_sum, m_sum, n_empty_pred, n_single, n_single_ok,
            n_nonsingle_empty, loss):
    """Build the summary dict from aggregate counts (shared by both engines)."""
    nan = float("nan")
    tot_loss = sum(loss.values())
    return {
        "n": n,
        "macro_f": fsum / n if n else nan,
        "micro_precision": tp_sum / m_sum if m_sum else nan,
        "micro_recall": tp_sum / k_sum if k_sum else nan,
        "mean_true_k": k_sum / n if n else nan,
        "mean_pred_size": m_sum / n if n else nan,
        "empty_pred_rate": n_empty_pred / n if n else nan,
        "singleton_rate": n_single / n if n else nan,
        "singleton_empty_acc": n_single_ok / n_single if n_single else nan,
        "nonsingleton_empty_rate": n_nonsingle_empty / (n - n_single) if n > n_single else nan,
        # (1 - macro_f) split by failure mode: loss_points sum to 1 - macro_f
        "loss_share": {kk: (loss[kk] / tot_loss if tot_loss else 0.0) for kk in LOSS_KEYS},
        "loss_points": {kk: (loss[kk] / n if n else 0.0) for kk in LOSS_KEYS},
    }


def _summarise(rows, beta):
    """rows: iterable of (s1, k, m, tp, f, country). Returns summary dict."""
    n = 0
    fsum = 0.0
    tp_sum = k_sum = m_sum = 0
    n_empty_pred = n_single = n_single_ok = n_nonsingle_empty = 0
    loss = dict.fromkeys(LOSS_KEYS, 0.0)
    for (_, k, m, tp, f, _) in rows:
        n += 1
        fsum += f
        tp_sum += tp
        k_sum += k
        m_sum += m
        if m == 0:
            n_empty_pred += 1
        if k == 0:
            n_single += 1
            if m == 0:
                n_single_ok += 1
            else:
                loss["singleton_false_merge"] += 1.0
        else:
            if m == 0:
                n_nonsingle_empty += 1
                loss["nonsingleton_empty_pred"] += 1.0
            elif tp == 0:
                loss["nonsingleton_all_wrong"] += 1.0
            else:
                loss["partial"] += 1.0 - f
    return _finish(n, fsum, tp_sum, k_sum, m_sum, n_empty_pred, n_single, n_single_ok,
                   n_nonsingle_empty, loss)


def _pl_agg_exprs():
    import polars as pl
    k, m, tp, f = pl.col("k"), pl.col("m"), pl.col("tp"), pl.col("f")
    return [pl.len().alias("n"), f.sum().alias("fsum"), tp.sum().alias("tp_sum"), k.sum().alias("k_sum"),
            m.sum().alias("m_sum"), (m == 0).sum().alias("n_empty_pred"), (k == 0).sum().alias("n_single"),
            ((k == 0) & (m == 0)).sum().alias("n_single_ok"), ((k > 0) & (m == 0)).sum().alias("n_nonsingle_empty"),
            ((k == 0) & (m > 0)).sum().cast(pl.Float64).alias("singleton_false_merge"),
            ((k > 0) & (m == 0)).sum().cast(pl.Float64).alias("nonsingleton_empty_pred"),
            ((k > 0) & (m > 0) & (tp == 0)).sum().cast(pl.Float64).alias("nonsingleton_all_wrong"),
            pl.when(tp > 0).then(1.0 - f).otherwise(0.0).sum().alias("partial")]


def _finish_row(r):
    return _finish(r["n"], r["fsum"], r["tp_sum"], r["k_sum"], r["m_sum"], r["n_empty_pred"], r["n_single"],
                   r["n_single_ok"], r["n_nonsingle_empty"], {kk: r[kk] for kk in LOSS_KEYS})


def _pl_summary(fr, by=None, order=None):
    """Vectorised summary of a per-S1 frame (columns k, m, tp, f [, by])."""
    if by is None:
        return _finish_row(fr.select(_pl_agg_exprs()).row(0, named=True))
    g = fr.group_by(by).agg(_pl_agg_exprs())
    out = {str(r[by]): _finish_row(r) for r in g.iter_rows(named=True)}
    keys = order if order is not None else sorted(out)
    return {kk: out[kk] for kk in keys if kk in out}


def score_dicts(pred: Dict[str, Iterable[str]], gt: Dict[str, Iterable[str]],
                ids: Optional[Iterable[str]] = None, beta: float = BETA,
                country: Optional[Dict[str, str]] = None, by_k: bool = False,
                by_pred_size: bool = False, warn=print) -> dict:
    """Score predictions against ground truth (pure python).

    pred, gt : {s1_id: iterable of S2/S3 ids}
    ids      : optional subset of S1 ids to evaluate (default: all gt keys)
    country  : optional {s1_id: country} -> adds 'by_country'
    """
    eval_ids = list(gt.keys()) if ids is None else list(dict.fromkeys(ids))
    missing_gt = [s for s in eval_ids if s not in gt]
    if missing_gt:
        warn(f"[metric] {len(missing_gt)} requested ids not in ground truth -> dropped (e.g. {missing_gt[:3]})")
        eval_ids = [s for s in eval_ids if s in gt]
    n_missing_pred = 0
    n_dup_in_list = 0
    rows = []
    for s1 in eval_ids:
        t = set(gt[s1])
        if s1 in pred:
            plist = list(pred[s1])
            pset = set(plist)
            if len(pset) != len(plist):
                n_dup_in_list += 1
        else:
            pset = set()
            n_missing_pred += 1
        tp = len(pset & t)
        f = f_beta_counts(tp, len(t), len(pset), beta)
        rows.append((s1, len(t), len(pset), tp, f, (country or {}).get(s1, "?")))
    if n_missing_pred:
        warn(f"[metric] WARNING {n_missing_pred} evaluated S1 have no prediction row -> scored as empty "
             f"(the official scorer would REJECT the file)")
    if n_dup_in_list:
        warn(f"[metric] WARNING {n_dup_in_list} prediction lists contain duplicate ids -> deduplicated "
             f"(the official scorer would REJECT the file)")
    extra = len(set(pred) - set(eval_ids))
    res = _summarise(rows, beta)
    res.update({"beta": beta, "n_pred_rows_ignored": extra, "n_missing_pred_rows": n_missing_pred,
                "n_lists_with_duplicates": n_dup_in_list})
    if country is not None:
        groups = defaultdict(list)
        for r in rows:
            groups[r[5]].append(r)
        res["by_country"] = {c: _summarise(g, beta) for c, g in sorted(groups.items())}
    if by_k:
        groups = defaultdict(list)
        for r in rows:
            groups[k_bucket(r[1])].append(r)
        res["by_k"] = {b: _summarise(groups[b], beta) for b in K_BUCKETS if b in groups}
    if by_pred_size:
        groups = defaultdict(list)
        for r in rows:
            groups[pred_bucket(r[2])].append(r)
        res["by_pred_size"] = {b: _summarise(groups[b], beta) for b in K_BUCKETS if b in groups}
    res["_rows"] = rows
    return res


# ----------------------------------------------------------------------------
# scoring (polars engine, vectorised)
# ----------------------------------------------------------------------------
def _pl_read_idlist(path):
    import polars as pl
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     empty_string_is_null=False, has_header=True)
    if df.width < 2:
        raise ValueError(f"{path}: expected 2 tab-separated columns, got {df.columns}")
    df = df.select(pl.col(df.columns[0]).str.strip_chars().alias("s1"),
                   pl.col(df.columns[1]).fill_null("").alias("ids"))
    dup = df.height - df["s1"].n_unique()
    if dup:
        raise ValueError(f"{path}: {dup} duplicate S1 rows (scorer rejects this)")
    return df


def _pl_long(df):
    import polars as pl
    return (df.select("s1", pl.col("ids").str.split(",").alias("mid"))
              .explode("mid", empty_as_null=True)
              .with_columns(pl.col("mid").str.strip_chars())
              .filter(pl.col("mid").is_not_null() & (pl.col("mid") != "")))


def score_frames_polars(pred_df, gt_df, ids=None, beta=BETA, country_df=None,
                        by_k=False, by_pred_size=False, warn=print, return_rows=False):
    """pred_df / gt_df: polars frames with columns s1 (str) and ids (comma list)."""
    import polars as pl
    b2 = beta * beta
    if ids is not None:
        ev = pl.DataFrame({"s1": list(dict.fromkeys(ids))})
        n_req = ev.height
        ev = ev.join(gt_df.select("s1"), on="s1", how="semi")
        if ev.height < n_req:
            warn(f"[metric] {n_req - ev.height} requested ids not in ground truth -> dropped")
    else:
        ev = gt_df.select("s1")
    gl = _pl_long(gt_df.join(ev, on="s1", how="semi"))
    pl_raw = _pl_long(pred_df.join(ev, on="s1", how="semi"))
    raw_counts = pl_raw.group_by("s1").agg(pl.len().alias("m_raw"), pl.col("mid").n_unique().alias("m"))
    n_dup_lists = raw_counts.filter(pl.col("m_raw") != pl.col("m")).height
    pll = pl_raw.unique()
    k = gl.group_by("s1").agg(pl.len().alias("k"))
    m = raw_counts.select("s1", "m")
    tp = gl.join(pll, on=["s1", "mid"], how="inner").group_by("s1").agg(pl.len().alias("tp"))
    have_pred = pred_df.select("s1").join(ev, on="s1", how="semi")
    n_missing_pred = ev.height - have_pred.height
    rows = (ev.join(k, on="s1", how="left").join(m, on="s1", how="left").join(tp, on="s1", how="left")
              .with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in ("k", "m", "tp")]))
    rows = rows.with_columns(
        pl.when((pl.col("k") == 0) & (pl.col("m") == 0)).then(1.0)
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("k") + pl.col("m"))).alias("f"))
    if country_df is not None:
        rows = rows.join(country_df, on="s1", how="left").with_columns(pl.col("country").fill_null("?"))
    else:
        rows = rows.with_columns(pl.lit("?").alias("country"))
    if n_missing_pred:
        warn(f"[metric] WARNING {n_missing_pred} evaluated S1 have no prediction row -> scored as empty "
             f"(the official scorer would REJECT the file)")
    if n_dup_lists:
        warn(f"[metric] WARNING {n_dup_lists} prediction lists contain duplicate ids -> deduplicated "
             f"(the official scorer would REJECT the file)")
    n_extra = pred_df.height - have_pred.height

    res = _pl_summary(rows)
    res.update({"beta": beta, "n_pred_rows_ignored": n_extra, "n_missing_pred_rows": n_missing_pred,
                "n_lists_with_duplicates": n_dup_lists})
    if country_df is not None:
        res["by_country"] = _pl_summary(rows, by="country")
    if by_k:
        rk = rows.with_columns(pl.when(pl.col("k") >= 7).then(pl.lit("7+"))
                                 .otherwise(pl.col("k").cast(pl.Utf8)).alias("kb"))
        res["by_k"] = _pl_summary(rk, by="kb", order=K_BUCKETS)
    if by_pred_size:
        rm = rows.with_columns(pl.when(pl.col("m") >= 7).then(pl.lit("7+"))
                                 .otherwise(pl.col("m").cast(pl.Utf8)).alias("mb"))
        res["by_pred_size"] = _pl_summary(rm, by="mb", order=K_BUCKETS)
    if return_rows:
        res["_rows_df"] = rows
    return res


def score_files(pred_path: str, gt_path: str, ids_path: Optional[str] = None,
                s1_path: Optional[str] = None, by_country: bool = False, by_k: bool = False,
                by_pred_size: bool = False, beta: float = BETA, engine: str = "polars",
                warn=print) -> dict:
    ids = load_ids_file(ids_path) if ids_path else None
    if engine == "python":
        pred = load_id_list_tsv(pred_path)
        gt = load_id_list_tsv(gt_path)
        country = load_country_map(s1_path) if (s1_path and by_country) else None
        res = score_dicts(pred, gt, ids=ids, beta=beta, country=country, by_k=by_k,
                          by_pred_size=by_pred_size, warn=warn)
        res.pop("_rows", None)
        return res
    import polars as pl
    pred_df = _pl_read_idlist(pred_path)
    gt_df = _pl_read_idlist(gt_path)
    cdf = None
    if s1_path and by_country:
        s1 = pl.read_csv(s1_path, separator="\t", quote_char=None, infer_schema=False,
                         empty_string_is_null=False, columns=["entity_id", "country"])
        cdf = s1.select(pl.col("entity_id").str.strip_chars().alias("s1"),
                        pl.col("country").str.strip_chars())
    return score_frames_polars(pred_df, gt_df, ids=ids, beta=beta, country_df=cdf,
                               by_k=by_k, by_pred_size=by_pred_size, warn=warn)


# ----------------------------------------------------------------------------
# contract API (long-format frames: s1, mid)  -- names fixed by CONTRACT.md
# ----------------------------------------------------------------------------
def per_s1_scores(pred, gt, s1_ids, beta: float = BETA):
    """pred, gt: polars frames in LONG form (s1, mid) -- one row per link; singletons have no rows.
    s1_ids: the evaluated S1 ids (list / Series / frame with column s1).  Returns a polars frame
    s1, k (true size), m (pred size), tp, f  -- one row per evaluated S1 (missing pred = empty)."""
    import polars as pl
    b2 = beta * beta
    if isinstance(s1_ids, pl.DataFrame):
        ev = s1_ids.select(pl.col("s1").cast(pl.Utf8)).unique(maintain_order=True)
    else:
        ev = pl.DataFrame({"s1": pl.Series(list(s1_ids), dtype=pl.Utf8)}).unique(maintain_order=True)
    g = gt.select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8)).unique().join(ev, on="s1", how="semi")
    p = pred.select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8)).unique().join(ev, on="s1", how="semi")
    k = g.group_by("s1").agg(pl.len().alias("k"))
    m = p.group_by("s1").agg(pl.len().alias("m"))
    tp = g.join(p, on=["s1", "mid"], how="inner").group_by("s1").agg(pl.len().alias("tp"))
    r = (ev.join(k, on="s1", how="left").join(m, on="s1", how="left").join(tp, on="s1", how="left")
           .with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in ("k", "m", "tp")]))
    return r.with_columns(
        pl.when((pl.col("k") == 0) & (pl.col("m") == 0)).then(1.0)
          .when(pl.col("tp") == 0).then(0.0)
          .otherwise((1 + b2) * pl.col("tp") / (b2 * pl.col("k") + pl.col("m"))).alias("f"))


def macro_f05(pred, gt, s1_ids, beta: float = BETA) -> float:
    """CONTRACT: exact macro F0.5 over s1_ids. pred/gt long frames (s1, mid)."""
    return float(per_s1_scores(pred, gt, s1_ids, beta)["f"].mean())


def f05_breakdown(pred, gt, s1_meta, beta: float = BETA):
    """CONTRACT: breakdown table. s1_meta: polars frame with the evaluated S1s and a `country`
    column (id column `s1` or `entity_id`). Returns rows for ALL, each country, each true-k bucket
    and each (country, k bucket): group_type, group, n, macro_f, mean_k, mean_m, micro_p, micro_r,
    empty_pred_rate, share_of_loss (fraction of the total 1-F lost in this group)."""
    import polars as pl
    meta = s1_meta.rename({"entity_id": "s1"}) if "entity_id" in s1_meta.columns else s1_meta
    meta = meta.select(pl.col("s1").cast(pl.Utf8), pl.col("country").cast(pl.Utf8).fill_null("?"))
    r = per_s1_scores(pred, gt, meta.select("s1"), beta).join(meta, on="s1", how="left")
    r = r.with_columns(pl.when(pl.col("k") >= 7).then(pl.lit("7+")).otherwise(pl.col("k").cast(pl.Utf8)).alias("kb"),
                       pl.lit("ALL").alias("all"))
    tot_loss = float((1 - r["f"]).sum()) or 1.0
    aggs = [pl.len().alias("n"), pl.col("f").mean().alias("macro_f"), pl.col("k").mean().alias("mean_k"),
            pl.col("m").mean().alias("mean_m"),
            (pl.col("tp").sum() / pl.col("m").sum()).alias("micro_p"),
            (pl.col("tp").sum() / pl.col("k").sum()).alias("micro_r"),
            (pl.col("m") == 0).mean().alias("empty_pred_rate"),
            ((1 - pl.col("f")).sum() / tot_loss).alias("share_of_loss")]
    parts = []
    for gtype, cols in (("all", ["all"]), ("country", ["country"]), ("k", ["kb"]), ("country_k", ["country", "kb"])):
        a = r.group_by(cols).agg(aggs)
        a = a.with_columns(pl.concat_str([pl.col(c) for c in cols], separator="|").alias("group")).drop(cols)
        parts.append(a.with_columns(pl.lit(gtype).alias("group_type")).sort("group"))
    return pl.concat(parts, how="vertical_relaxed").select(
        "group_type", "group", "n", "macro_f", "mean_k", "mean_m", "micro_p", "micro_r", "empty_pred_rate",
        "share_of_loss")


def fold_of(s1_ids, n_folds: int = 5, seed: int = 0):
    """Deterministic, version-stable fold id per S1 (splitmix64 of the numeric id suffix; zlib.crc32 for
    ids without digits). Use for GroupKFold-by-S1: every pair (s1, cand) inherits the fold of its s1.
    Returns a numpy int64 array aligned with s1_ids."""
    import zlib
    import numpy as np
    import polars as pl
    s = pl.Series(list(s1_ids) if not isinstance(s1_ids, pl.Series) else s1_ids).cast(pl.Utf8)
    num = s.str.extract(r"(\d+)\s*$", 1).cast(pl.UInt64, strict=False)
    x = num.fill_null(0).to_numpy().astype(np.uint64)
    miss = num.is_null().to_numpy()
    if miss.any():
        x[miss] = np.array([zlib.crc32(v.encode()) for v in s.filter(pl.Series(miss)).to_list()], dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15) * np.uint64(seed + 1)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(n_folds)).astype(np.int64)


# ----------------------------------------------------------------------------
# pretty printing / CLI
# ----------------------------------------------------------------------------
def _fmt_block(name, d):
    return (f"{name:<10} n={d['n']:>9,}  F0.5={d['macro_f']:.5f}  microP={d['micro_precision']:.4f} "
            f"microR={d['micro_recall']:.4f}  predSize={d['mean_pred_size']:.2f} trueK={d['mean_true_k']:.2f} "
            f"emptyPred={d['empty_pred_rate']:.4f} singletonAcc={d['singleton_empty_acc']:.4f} "
            f"nonSingleEmpty={d['nonsingleton_empty_rate']:.4f}")


def print_report(res):
    print(_fmt_block("ALL", res))
    print("  loss points (sum = 1 - F):", {k: round(v, 5) for k, v in res.get("loss_points", {}).items()})
    for sec in ("by_country", "by_k", "by_pred_size"):
        if sec in res:
            print(f"-- {sec}")
            for k, d in res[sec].items():
                print("  " + _fmt_block(k, d))


def _strip_private(res):
    return {k: (_strip_private(v) if isinstance(v, dict) else v) for k, v in res.items() if not k.startswith("_")}


def _peak_mem_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 1e9  # peak_wset exists on Windows
    except Exception:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB on Linux
        except Exception:
            return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-S1 macro F_beta (beta=0.5) scorer for ML Challenge 2026")
    ap.add_argument("--pred", help="predictions TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--gt", help="ground-truth TSV (source1_entity_id, matched_entity_ids)")
    ap.add_argument("--ids", help="optional file with S1 ids to restrict evaluation to")
    ap.add_argument("--s1", help="source1 TSV (for --by-country)")
    ap.add_argument("--by-country", action="store_true")
    ap.add_argument("--by-k", action="store_true", help="breakdown by true match count bucket")
    ap.add_argument("--by-pred-size", action="store_true", help="breakdown by predicted list size bucket")
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--engine", choices=["polars", "python"], default="polars")
    ap.add_argument("--json", help="write full result dict as JSON here")
    ap.add_argument("--selftest", action="store_true", help="run built-in sanity checks and exit")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not (a.pred and a.gt):
        ap.error("--pred and --gt are required (or use --selftest)")
    if a.by_country and not a.s1:
        ap.error("--by-country needs --s1 <source1.tsv>")
    t0 = time.time()
    res = score_files(a.pred, a.gt, ids_path=a.ids, s1_path=a.s1, by_country=a.by_country,
                      by_k=a.by_k, by_pred_size=a.by_pred_size, beta=a.beta, engine=a.engine,
                      warn=lambda s: print(s, file=sys.stderr))
    print_report(res)
    print(f"[metric] engine={a.engine} time={time.time() - t0:.1f}s peak_mem={_peak_mem_gb():.2f}GB",
          file=sys.stderr)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_strip_private(res), f, indent=2)
    return 0


def _selftest():
    ex = f_beta(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert abs(ex - 0.7142857142857143) < 1e-12, ex
    assert round(ex, 3) == 0.714
    assert f_beta([], []) == 1.0
    assert f_beta(["S2-1"], []) == 0.0
    assert f_beta([], ["S2-1"]) == 0.0
    assert f_beta(["S2-2"], ["S2-1"]) == 0.0
    assert f_beta(["S2-1", "S2-1"], ["S2-1"]) == 1.0
    # equivalence with 1.25PR/(0.25P+R)
    for tp, k, m in [(1, 1, 1), (2, 4, 2), (3, 4, 5), (1, 7, 1), (4, 4, 6)]:
        P, R = tp / m, tp / k
        assert abs(f_beta_counts(tp, k, m) - 1.25 * P * R / (0.25 * P + R)) < 1e-12
    print("selftest OK: example=%.6f" % ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
