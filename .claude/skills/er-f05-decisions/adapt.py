# -*- coding: utf-8 -*-
"""adapt.py -- label-free prior-shift adaptation of calibrated pair probabilities (Master Bolt ER).
Reference copy: .claude/skills/er-f05-decisions/adapt.py (keep both files identical).

Why: test pools carry ~1.9x the distractors per S1 of train, concentrated in HARD house-number relations (same
name / street / city, number shifted by 3-100 or one digit changed by 3/4/5/7/9; train P_true 0.02-0.13), while the
copy-type relations (exact, letter change, digit added / dropped) are unchanged (work/reports/diag_results.md).
A model calibrated on train then over-states P(match) inside the shifted classes. This module re-estimates the
match prior of every (country, house-number relation class) on the unlabeled target and rescales the odds:
    p' = r p / (r p + 1 - p),   r = odds(prior_target) / odds(prior_source)
which is the Bayes-optimal correction under label shift inside a class (P(x | y, class) unchanged).

Relation class (first house number of the S1 vs the candidate; copied from work/infra/diag_struct_k.py and made
vectorised): exact | letter_change | digit_added | digit_dropped | subst_d1_2 | subst_d3_9 (one digit changed by
3/4/5/7/9) | subst_d6_8 (by 6/8: copy-like in US, P_true 0.67-0.80) | subst_d10_99 | subst_d100plus (a higher digit
changed) | diff_1_2 | diff_3_10 | diff_11_100 | diff_100plus | one_missing | none (no number on either side) | other.
HARD = subst_d3_9, diff_3_10, diff_11_100; SMALL = subst_d1_2, diff_1_2.

Estimators (per class, on the rows with p >= p_min)
    em       Saerens-Latinne-Decaestecker EM: pi <- mean_i q_i, q_i = a p_i / (a p_i + b (1 - p_i)),
             a = pi / pi_s, b = (1 - pi) / (1 - pi_s)  (maximum-likelihood prior under label shift; run on a
             histogram of logit p, exact to ~1e-6)
    density  positives per S1 in the class are invariant (the copy generator did not change), so
             pi_t = pi_s * (pairs per S1 in the source) / (pairs per S1 in the target)
Source prior pi_s = mean p of the class on the OOF frame (source_stat="p": EM is then exactly neutral on the source
frame) or its mean label (source_stat="label").
Safeguards: classes with < min_rows target rows keep pi_s; the estimate is shrunk towards pi_s with shrink_n0
pseudo-rows; priors only go DOWN unless allow_raise and 2*logLR >= raise_llr; EM changes need 2*logLR >= min_llr;
the odds factor is clipped to [1/max_factor, max_factor]. Countries without a source (France) use the pooled prior.

Contract
--------
hn_relation(a, b) -> str                 scalar reference (first house numbers as str / None)
hn_relation_expr(a="h1", b="h2") -> pl.Expr   vectorised version (identical results, tested)
tag_pairs(scored, s1_norm, pool_norm, p_min=0.0) -> scored + hn_rel (null for rows with p < p_min)
    s1_norm / pool_norm: DataFrame | LazyFrame | parquet path(s) with entity_id, house_nums (List[str])
AdaptConfig                              estimator + safeguards
fit_source(tagged_oof, cfg, country_col="country") -> source dict (priors per country and pooled "*")
save_source(source, path) / load_source(path)
estimate(tagged, source, cfg, country) -> pl.DataFrame  per class: n, pairs_per_s1, prior_s, prior_raw, llr2,
                                         prior_t, factor, reason
apply(tagged, estimates, p_min) -> tagged with p rescaled (p_unadapted keeps the input p)
adapt(tagged, source, cfg, country) -> (adapted, estimates)
targeted_rule(links, tagged, classes=HARD_CLASSES, p_max=0.99, require_exact=True) -> links
    drop a selected HARD link when the same S1 has another selected link with the exact house number and p < p_max
    (diag_struct_k: -0.00032 on OOF; drops 0.044 / 0.015 / 0.018 links per S1 on test US / India / France)
class_profile(tagged, links, n_s1) -> per class: pairs / sum p / links per S1 (label-free shift monitor)
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

REL_CLASSES = ("exact", "letter_change", "digit_added", "digit_dropped", "subst_d1_2", "subst_d3_9", "subst_d6_8",
               "subst_d10_99", "subst_d100plus", "diff_1_2", "diff_3_10", "diff_11_100", "diff_100plus",
               "one_missing", "none", "other")
HARD_CLASSES = ("subst_d3_9", "diff_3_10", "diff_11_100")
SMALL_CLASSES = ("subst_d1_2", "diff_1_2")
COPY_CLASSES = ("exact", "letter_change", "digit_added", "digit_dropped")
POOLED = "*"
_HAM_LEN = 16
_NONDIGIT = re.compile(r"\D")


# ------------------------------------------------------------------------------------------------ relation class
def hn_relation(a: Optional[str], b: Optional[str]) -> str:
    """Relation between the S1's first house number a and the candidate's first house number b."""
    a, b = a or "", b or ""
    if not a and not b:
        return "none"
    if not a or not b:
        return "one_missing"
    if a == b:
        return "exact"
    da, db = _NONDIGIT.sub("", a), _NONDIGIT.sub("", b)
    if da == db and da:
        return "letter_change"
    if not da or not db:
        return "other"
    if len(db) < len(da) and db in da:
        return "digit_dropped"
    if len(da) < len(db) and da in db:
        return "digit_added"
    d = abs(int(da[:12]) - int(db[:12]))
    if len(da) == len(db) and sum(x != y for x, y in zip(da[:_HAM_LEN], db[:_HAM_LEN])) == 1:
        if d < 10:
            return "subst_d1_2" if d <= 2 else ("subst_d6_8" if d in (6, 8) else "subst_d3_9")
        return "subst_d10_99" if d < 100 else "subst_d100plus"
    if d <= 2:
        return "diff_1_2"
    if d <= 10:
        return "diff_3_10"
    if d <= 100:
        return "diff_11_100"
    return "diff_100plus"


def hn_relation_expr(a: str = "h1", b: str = "h2") -> pl.Expr:
    """Vectorised hn_relation on two Utf8 columns (null / "" = no house number). Returns a Utf8 expression."""
    A, B = pl.col(a).fill_null(""), pl.col(b).fill_null("")
    da, db = A.str.replace_all(r"\D", ""), B.str.replace_all(r"\D", "")
    la, lb = da.str.len_chars(), db.str.len_chars()
    d = (da.str.slice(0, 12).cast(pl.Int64, strict=False) - db.str.slice(0, 12).cast(pl.Int64, strict=False)).abs()
    ham = pl.sum_horizontal([(da.str.slice(k, 1) != db.str.slice(k, 1)).cast(pl.Int32) for k in range(_HAM_LEN)])
    sub1 = (la == lb) & (ham == 1)
    lit = pl.lit
    return (pl.when((A == "") & (B == "")).then(lit("none"))
              .when((A == "") | (B == "")).then(lit("one_missing"))
              .when(A == B).then(lit("exact"))
              .when((da == db) & (la > 0)).then(lit("letter_change"))
              .when((la == 0) | (lb == 0)).then(lit("other"))
              .when((lb < la) & da.str.contains(db, literal=True)).then(lit("digit_dropped"))
              .when((la < lb) & db.str.contains(da, literal=True)).then(lit("digit_added"))
              .when(d.is_null()).then(lit("other"))
              .when(sub1 & (d <= 2)).then(lit("subst_d1_2"))
              .when(sub1 & ((d == 6) | (d == 8))).then(lit("subst_d6_8"))
              .when(sub1 & (d < 10)).then(lit("subst_d3_9"))
              .when(sub1 & (d < 100)).then(lit("subst_d10_99"))
              .when(sub1).then(lit("subst_d100plus"))
              .when(d <= 2).then(lit("diff_1_2"))
              .when(d <= 10).then(lit("diff_3_10"))
              .when(d <= 100).then(lit("diff_11_100"))
              .otherwise(lit("diff_100plus")))


def _paths(src) -> Optional[List[str]]:
    if isinstance(src, (str, os.PathLike)):
        return [str(src)]
    if isinstance(src, (list, tuple)) and src and all(isinstance(s, (str, os.PathLike)) for s in src):
        return [str(s) for s in src]
    return None


def _first_hn(norm, ids: pl.DataFrame) -> pl.DataFrame:
    """entity_id, h (first house number, "" when none) for the ids in `ids` (column entity_id)."""
    ps = _paths(norm)
    lf = pl.scan_parquet(ps) if ps is not None else (norm.lazy() if isinstance(norm, pl.DataFrame) else norm)
    sch = lf.collect_schema()
    ids = ids.with_columns(pl.col("entity_id").cast(sch["entity_id"]))
    hn = pl.col("house_nums")
    first = (hn.list.first() if isinstance(sch["house_nums"], pl.List) else hn).cast(pl.Utf8).fill_null("")
    return (lf.select("entity_id", first.alias("h")).join(ids.lazy(), on="entity_id", how="semi").collect()
              .unique(subset=["entity_id"], keep="first", maintain_order=True))


def tag_pairs(scored: pl.DataFrame, s1_norm, pool_norm, p_min: float = 0.0, p: str = "p") -> pl.DataFrame:
    """scored (s1, cand, p, ...) + hn_rel (Utf8; null for rows with p < p_min). Row order is kept."""
    x = scored.with_row_index("_ri")
    sub = x.filter(pl.col(p) >= p_min).select("_ri", "s1", "cand") if p_min > 0 else x.select("_ri", "s1", "cand")
    h1 = _first_hn(s1_norm, sub.select(pl.col("s1").alias("entity_id")).unique())
    h2 = _first_hn(pool_norm, sub.select(pl.col("cand").alias("entity_id")).unique())
    sub = (sub.join(h1.rename({"entity_id": "s1", "h": "h1"}).with_columns(pl.col("s1").cast(sub.schema["s1"])),
                    on="s1", how="left")
              .join(h2.rename({"entity_id": "cand", "h": "h2"}).with_columns(pl.col("cand").cast(sub.schema["cand"])),
                    on="cand", how="left")
              .select("_ri", hn_relation_expr("h1", "h2").alias("hn_rel")))
    return x.join(sub, on="_ri", how="left", maintain_order="left").drop("_ri")


# ------------------------------------------------------------------------------------------------ estimation
@dataclass
class AdaptConfig:
    method: str = "em"                  # em | density | none
    p_min: float = 1e-3                 # rows with p < p_min are neither used nor changed
    classes: Optional[List[str]] = None  # classes that may be adapted (None = every class)
    source_stat: str = "p"              # source prior = mean p ("p") or mean label ("label") of the class on OOF
    shrink_n0: float = 2000.0           # pseudo-rows pulling every estimate towards the source prior
    min_rows: int = 200                 # classes with fewer target rows keep the source prior
    allow_raise: bool = False           # priors only go down unless allow_raise and 2*logLR >= raise_llr
    raise_llr: float = 100.0
    min_llr: float = 0.0                # em only: 2*logLR below this -> no change
    max_factor: float = 4.0             # odds factor clipped to [1/max_factor, max_factor]
    max_iter: int = 5000
    tol: float = 1e-10
    n_bins: int = 4096                  # EM runs on a histogram of logit(p) per class
    by_country: bool = True             # per-country source priors; unseen countries use the pooled "*" prior


def _hist(p: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """(counts, mean p) over n_bins bins of logit(p) on [-18, 18] (empty bins removed)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p))
    k = np.clip(((z + 18.0) / 36.0 * n_bins).astype(np.int64), 0, n_bins - 1)
    w = np.bincount(k, minlength=n_bins).astype(np.float64)
    s = np.bincount(k, weights=p, minlength=n_bins)
    m = w > 0
    return w[m], s[m] / w[m]


def em_prior(p: np.ndarray, prior_s: float, w: Optional[np.ndarray] = None, max_iter: int = 5000,
             tol: float = 1e-10) -> Tuple[float, int]:
    """Saerens-Latinne-Decaestecker EM for the target prior of calibrated posteriors p (source prior prior_s)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    ps = float(np.clip(prior_s, 1e-9, 1 - 1e-9))
    pi, W = ps, w.sum()
    for it in range(1, max_iter + 1):
        a, b = pi / ps, (1 - pi) / (1 - ps)
        q = a * p / (a * p + b * (1 - p))
        new = float((w * q).sum() / W)
        if abs(new - pi) < tol:
            return new, it
        pi = new
    return pi, max_iter


def loglr2(p: np.ndarray, prior_s: float, prior_t: float, w: Optional[np.ndarray] = None) -> float:
    """2 x log-likelihood ratio of target prior prior_t vs prior_s given calibrated posteriors p."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    w = np.ones_like(p) if w is None else np.asarray(w, dtype=np.float64)
    ps, pt = (float(np.clip(v, 1e-9, 1 - 1e-9)) for v in (prior_s, prior_t))
    return float(2.0 * (w * np.log(pt / ps * p + (1 - pt) / (1 - ps) * (1 - p))).sum())


def _odds(x: float) -> float:
    x = min(max(x, 1e-12), 1 - 1e-12)
    return x / (1 - x)


def fit_source(tagged: pl.DataFrame, cfg: Optional[AdaptConfig] = None, country_col: Optional[str] = "country",
               p: str = "p", label: str = "label") -> Dict:
    """Source priors from the OOF frame (s1, cand, p, hn_rel [, label] [, country]); rows with p < p_min are not
    counted. Per country (when by_country and the column exists) and pooled ('*')."""
    cfg = cfg or AdaptConfig()
    has_c = bool(country_col) and country_col in tagged.columns
    has_l = label in tagged.columns
    if cfg.source_stat == "label" and not has_l:
        raise ValueError("source_stat='label' needs a label column")
    x = tagged.with_columns(pl.lit(POOLED).alias("_c") if not has_c else pl.col(country_col).cast(pl.Utf8).alias("_c"))
    groups = [x]
    if has_c:
        groups.append(x.with_columns(pl.lit(POOLED).alias("_c")))
    out: Dict = {"cfg": asdict(cfg), "priors": {}, "n_s1": {}}
    for g in groups:
        ns1 = g.group_by("_c").agg(pl.col("s1").n_unique().alias("n_s1"))
        for c, n in ns1.rows():
            out["n_s1"][c] = int(n)
        aggs = [pl.len().alias("n"), pl.col(p).cast(pl.Float64).mean().alias("mean_p"),
                pl.col(p).cast(pl.Float64).sum().alias("sum_p")]
        if has_l:
            aggs.append(pl.col(label).cast(pl.Float64).mean().alias("mean_label"))
        t = (g.filter((pl.col(p) >= cfg.p_min) & pl.col("hn_rel").is_not_null())
              .group_by("_c", "hn_rel").agg(aggs).sort("_c", "hn_rel"))
        for r in t.iter_rows(named=True):
            c = r["_c"]
            n_s1 = out["n_s1"][c]
            prior = r["mean_label"] if cfg.source_stat == "label" else r["mean_p"]
            out["priors"].setdefault(c, {})[r["hn_rel"]] = {
                "n": int(r["n"]), "pairs_per_s1": r["n"] / max(n_s1, 1), "prior": float(prior),
                "mean_p": float(r["mean_p"]), "mean_label": float(r["mean_label"]) if has_l else None}
    return out


def save_source(source: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(source, f, indent=1)


def load_source(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _source_for(source: Dict, country: Optional[str], cfg: AdaptConfig) -> Tuple[str, Dict]:
    pri = source["priors"]
    if cfg.by_country and country is not None and str(country) in pri:
        return str(country), pri[str(country)]
    return POOLED, pri[POOLED]


def estimate(tagged: pl.DataFrame, source: Dict, cfg: Optional[AdaptConfig] = None,
             country: Optional[str] = None, p: str = "p", n_s1: Optional[int] = None) -> pl.DataFrame:
    """Target prior per relation class for ONE country frame (s1, cand, p, hn_rel). n_s1 = number of S1 of the
    frame (default: distinct s1 in `tagged`). Columns: hn_rel, n, pairs_per_s1, pairs_per_s1_src, prior_s,
    prior_raw, llr2, iters, prior_t, factor, reason, src."""
    cfg = cfg or AdaptConfig()
    src_key, src = _source_for(source, country, cfg)
    n_s1 = int(n_s1 if n_s1 is not None else tagged.select(pl.col("s1").n_unique()).item())
    x = tagged.filter((pl.col(p) >= cfg.p_min) & pl.col("hn_rel").is_not_null()).select("hn_rel", pl.col(p).cast(pl.Float64))
    rows = []
    for (cls,), g in sorted(x.group_by(["hn_rel"]), key=lambda kv: str(kv[0][0])):
        pv = g[p].to_numpy()
        n = len(pv)
        s = src.get(cls)
        row = {"hn_rel": cls, "n": n, "pairs_per_s1": n / max(n_s1, 1), "src": src_key,
               "pairs_per_s1_src": s["pairs_per_s1"] if s else None, "prior_s": s["prior"] if s else None,
               "mean_p": float(pv.mean()) if n else None, "prior_raw": None, "llr2": None, "iters": 0,
               "prior_t": s["prior"] if s else None, "factor": 1.0, "reason": ""}
        eligible = cfg.classes is None or cls in cfg.classes
        if s is None or not (0.0 < s["prior"] < 1.0):
            row["reason"] = "no source prior"
        elif cfg.method == "none" or not eligible:
            row["reason"] = "not adapted"
        elif n < cfg.min_rows:
            row["reason"] = f"n < {cfg.min_rows}"
        else:
            ps = float(s["prior"])
            w, pm = _hist(pv, cfg.n_bins)
            if cfg.method == "em":
                raw, row["iters"] = em_prior(pm, ps, w, cfg.max_iter, cfg.tol)
            elif cfg.method == "density":
                raw = float(np.clip(ps * s["pairs_per_s1"] / max(row["pairs_per_s1"], 1e-12), 1e-9, 1 - 1e-9))
            else:
                raise ValueError(f"method must be em, density or none, got {cfg.method!r}")
            llr = loglr2(pm, ps, raw, w)
            row["prior_raw"], row["llr2"] = raw, llr
            wt = n / (n + cfg.shrink_n0)
            pt = wt * raw + (1 - wt) * ps
            reason = "adapted"
            if pt > ps and not (cfg.allow_raise and llr >= cfg.raise_llr):
                pt, reason = ps, "raise blocked"
            elif cfg.method == "em" and llr < cfg.min_llr:
                pt, reason = ps, f"2logLR < {cfg.min_llr}"
            r = _odds(pt) / _odds(ps)
            hi = cfg.max_factor if (cfg.allow_raise and pt > ps) else 1.0
            r_c = float(min(max(r, 1.0 / cfg.max_factor), max(hi, 1.0)))
            if r_c != r:
                reason += " (capped)"
                pt = _odds(ps) * r_c / (1 + _odds(ps) * r_c)
            row["prior_t"], row["factor"], row["reason"] = float(pt), r_c, reason
        rows.append(row)
    schema = {"hn_rel": pl.Utf8, "n": pl.Int64, "pairs_per_s1": pl.Float64, "pairs_per_s1_src": pl.Float64,
              "prior_s": pl.Float64, "mean_p": pl.Float64, "prior_raw": pl.Float64, "llr2": pl.Float64,
              "iters": pl.Int64, "prior_t": pl.Float64, "factor": pl.Float64, "reason": pl.Utf8, "src": pl.Utf8}
    return pl.DataFrame(rows, schema=schema).select(list(schema))


def apply(tagged: pl.DataFrame, estimates: pl.DataFrame, p_min: float = 1e-3, p: str = "p",
          keep_orig: bool = True) -> pl.DataFrame:
    """Rescale p by the class odds factor (rows with p >= p_min and a class). Row order kept."""
    f = estimates.filter(pl.col("factor") != 1.0).select("hn_rel", pl.col("factor").alias("_r"))
    x = tagged.join(f, on="hn_rel", how="left", maintain_order="left")
    pc = pl.col(p).cast(pl.Float64).clip(0.0, 1.0)
    r = pl.when(pc >= p_min).then(pl.col("_r").fill_null(1.0)).otherwise(1.0)
    new = (r * pc / (r * pc + (1 - pc))).fill_nan(0.0)
    if keep_orig and "p_unadapted" not in x.columns:
        x = x.with_columns(pl.col(p).alias("p_unadapted"))
    return x.with_columns(new.cast(tagged.schema[p]).alias(p)).drop("_r")


def adapt(tagged: pl.DataFrame, source: Dict, cfg: Optional[AdaptConfig] = None, country: Optional[str] = None,
          p: str = "p", n_s1: Optional[int] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """estimate + apply for one country frame -> (adapted frame, estimates)."""
    cfg = cfg or AdaptConfig()
    est = estimate(tagged, source, cfg, country, p=p, n_s1=n_s1)
    return apply(tagged, est, cfg.p_min, p=p), est


# ------------------------------------------------------------------------------------------------ rule + monitor
def targeted_rule(links: pl.DataFrame, tagged: pl.DataFrame, classes: Sequence[str] = HARD_CLASSES,
                  p_max: float = 0.99, require_exact: bool = True, p: str = "p") -> pl.DataFrame:
    """Drop selected links in `classes` with p < p_max when the same S1 also has a selected exact-house-number
    link (diag_struct_k 'drop HARD if S1 has exact & p<0.99'). links: s1, mid; tagged: s1, cand, p, hn_rel."""
    t = tagged.select("s1", pl.col("cand").alias("mid"), pl.col(p).alias("_p"), "hn_rel")
    x = links.select("s1", "mid").join(t, on=["s1", "mid"], how="left", maintain_order="left")
    x = x.with_columns((pl.col("hn_rel") == "exact").fill_null(False).any().over("s1").alias("_ex"))
    drop = pl.col("hn_rel").is_in(list(classes)).fill_null(False) & (pl.col("_p") < p_max).fill_null(False)
    if require_exact:
        drop = drop & pl.col("_ex")
    return x.filter(~drop).select("s1", "mid")


def class_profile(tagged: pl.DataFrame, links: Optional[pl.DataFrame] = None, n_s1: Optional[int] = None,
                  p_min: float = 1e-3, p: str = "p") -> pl.DataFrame:
    """Label-free monitor per relation class: pairs / sum p / selected links per S1 (+ a HARD total row)."""
    n_s1 = int(n_s1 if n_s1 is not None else tagged.select(pl.col("s1").n_unique()).item())
    x = tagged.filter(pl.col(p) >= p_min).select("s1", "cand", pl.col(p).cast(pl.Float64), "hn_rel")
    if links is not None:
        x = x.join(links.select("s1", pl.col("mid").alias("cand"), pl.lit(1).alias("_sel")), on=["s1", "cand"],
                   how="left").with_columns(pl.col("_sel").fill_null(0))
    else:
        x = x.with_columns(pl.lit(0).alias("_sel"))
    x = x.with_columns(pl.col("hn_rel").fill_null("untagged"))
    x = pl.concat([x, x.filter(pl.col("hn_rel").is_in(list(HARD_CLASSES))).with_columns(pl.lit("HARD").alias("hn_rel"))])
    return (x.group_by("hn_rel").agg((pl.len() / n_s1).alias("pairs_per_s1"), (pl.col(p).sum() / n_s1).alias("sum_p_per_s1"),
                                     (pl.col("_sel").sum() / n_s1).alias("links_per_s1"))
             .sort("hn_rel"))
