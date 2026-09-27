# -*- coding: utf-8 -*-
"""diag_stage2_lib.py -- stage-2 "collective" re-scoring on top of stage-1 calibrated p (prototype; written so it
can be moved into the pipeline as ber/stage2.py).

Every stage-2 feature is LABEL-FREE: a function of the stage-1 calibrated probability p1 of all candidate pairs,
the candidate lists and the normalized names/addresses. Nothing uses the country value.

Frames (ids may be str or int64 codes; only equality is used)
    edges : s1, cand, p1 [, p1raw]            one row per (S1, candidate) pair of the population to re-score
    allp  : s1, cand, p1                      every S1 that can claim the records (competitors; superset of edges)
    names : cand, nc (name_core), ad (addr_norm), nk (name_key)

Feature groups
    W_FEATS  within-S1 : p1, logit(raw), rank, gap to top / next / previous, #p1>0.5 / >0.2, sum p1 (expected #links),
                         sum excluding self, list length, top-1 / top-2 p1 of the S1
    D_FEATS  decision  : stage-1 expected-F decision on p1 alone (selected?, rank - m*, m*, P(empty))
    C_FEATS  competition (exclusivity: a record belongs to at most one S1): #other claimants, max p1 of the other S1,
                         margin to it, #other S1 with p1>0.3, p1 / sum_s p1(s, record), soft posterior o/(1+sum o),
                         sum over the others, is-argmax
    S_FEATS  siblings  : for candidate c of S1 s, over the anchors d = top-8 other candidates of s with p1(d) >= 0.05:
                         max sim(c,d)*p1(d) (name token_set on name_core), p1-weighted mean sim, sim / p1 of the best
                         other anchor, #anchors with sim>=0.9 & p1>=0.5, max p1 of anchors with the SAME name key and
                         their count, max addr-sim*p1 and max (name+addr)/2*p1 when both addresses are non-empty

Decision helpers
    select_fast(...)  = decide.resolve(expected-F, exact) on edges pruned at p < prune (their mass folded into the
                        per-S1 background rate) -> same links as decide.select_links, ~5x faster on 20M edges
    score(links, gt, meta) -> {"ALL": f, <country>: f, ...}
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import polars as pl

EPS = 1e-6

W_FEATS = ["p1", "p1_logit", "w_rank", "w_gap_top", "w_gap_next", "w_gap_prev", "w_n05", "w_n02", "w_sum",
           "w_sum_other", "w_ncand", "w_top1", "w_top2"]
D_FEATS = ["w_sel1", "w_rank_m", "w_mstar", "w_pempty"]
C_BASE = ["c_n_other", "c_other_max", "c_margin", "c_n_other03", "c_share", "c_soft", "c_sum_other", "c_is_best"]
S_FEATS = ["sib_n", "sib_simp_max", "sib_wsim", "sib_sim_max", "sib_top_sim", "sib_top_p", "sib_n09", "sib_key_maxp",
           "sib_key_n05", "sib_asimp_max", "sib_nasimp_max"]


def c_feats(suffix: str = "") -> List[str]:
    return [f"{c}{suffix}" for c in C_BASE]


def _log(*a):
    print(time.strftime("%H:%M:%S"), "[stage2]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ ids
def id_code(col: str = "entity_id") -> pl.Expr:
    """'S2-166376419' -> 2*1e10 + 166376419 (int64). Train/test ids are 'S<k>-<digits>' with < 10 digits."""
    c = pl.col(col).cast(pl.Utf8)
    return c.str.slice(1, 1).cast(pl.Int64) * 10_000_000_000 + c.str.slice(3).cast(pl.Int64)


# ------------------------------------------------------------------------------------------------ within-S1
def within_s1_feats(e: pl.DataFrame, p: str = "p1", praw: Optional[str] = None) -> pl.DataFrame:
    """e: s1, cand, p1 [, p1raw] with ALL candidates of each S1 (the features describe the whole list)."""
    lg = pl.col(praw if praw and praw in e.columns else p).clip(EPS, 1 - EPS)
    e = e.sort(["s1", p, "cand"], descending=[False, True, False])
    f32 = pl.Float32
    e = e.with_columns(
        (lg / (1 - lg)).log().cast(f32).alias("p1_logit"),
        pl.int_range(pl.len()).over("s1").cast(f32).alias("w_rank"),
        (pl.col(p) - pl.col(p).max().over("s1")).cast(f32).alias("w_gap_top"),
        (pl.col(p) - pl.col(p).shift(-1).over("s1").fill_null(0.0)).cast(f32).alias("w_gap_next"),
        (pl.col(p).shift(1).over("s1").fill_null(1.0) - pl.col(p)).cast(f32).alias("w_gap_prev"),
        (pl.col(p) > 0.5).sum().over("s1").cast(f32).alias("w_n05"),
        (pl.col(p) > 0.2).sum().over("s1").cast(f32).alias("w_n02"),
        pl.col(p).sum().over("s1").cast(f32).alias("w_sum"),
        pl.len().over("s1").cast(f32).alias("w_ncand"),
        pl.col(p).first().over("s1").cast(f32).alias("w_top1"),
        pl.col(p).slice(1, 1).first().over("s1").fill_null(0.0).cast(f32).alias("w_top2"),
    )
    return e.with_columns((pl.col("w_sum") - pl.col(p)).cast(f32).alias("w_sum_other"))


def decision_feats(e: pl.DataFrame, lam: float, empty_bias: float, p: str = "p1") -> pl.DataFrame:
    """Stage-1 expected-F decision (no exclusivity) as features: selected?, rank - m*, m*, P(truth empty)."""
    from ber import decide as D
    d, tab = D.decide_exact(e.select("s1", "cand", pl.col(p).alias("p")), lam_missing=lam, empty_bias=empty_bias,
                            return_s1_table=True)
    out = (d.select("s1", "cand", "rank", pl.col("sel").cast(pl.Float32).alias("w_sel1"))
            .join(tab.select("s1", "m_star", "p_empty"), on="s1", how="left")
            .select("s1", "cand", "w_sel1",
                    (pl.col("rank") - pl.col("m_star")).cast(pl.Float32).alias("w_rank_m"),
                    pl.col("m_star").cast(pl.Float32).alias("w_mstar"),
                    pl.col("p_empty").cast(pl.Float32).alias("w_pempty")))
    return e.join(out, on=["s1", "cand"], how="left")


# ------------------------------------------------------------------------------------------------ competition
def competition_aggregates(allp: pl.DataFrame, cands: Optional[pl.DataFrame] = None, p: str = "p1") -> pl.DataFrame:
    """Per record: #claimants, sum p, sum odds, #p>0.3, top-1 p (+ its S1), top-2 p over ALL claiming S1."""
    a = allp.select("s1", "cand", pl.col(p).cast(pl.Float64).alias("_p"))
    if cands is not None:
        a = a.join(cands.select("cand").unique(), on="cand", how="semi")
    pc = pl.col("_p").clip(0.0, 1 - EPS)
    return a.group_by("cand").agg(
        pl.len().alias("_n"), pl.col("_p").sum().alias("_sum"), (pc / (1 - pc)).sum().alias("_osum"),
        (pl.col("_p") > 0.3).sum().alias("_n03"), pl.col("_p").max().alias("_top1"),
        pl.col("s1").sort_by("_p", descending=True).first().alias("_top1_s1"),
        pl.col("_p").top_k(2).min().alias("_top2"))


def competition_feats(e: pl.DataFrame, allp: Optional[pl.DataFrame] = None, p: str = "p1", suffix: str = "",
                      agg: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Competition over every S1 that claims the record (allp must contain e's own rows)."""
    if agg is None:
        agg = competition_aggregates(allp, e, p)
    x = e.join(agg, on="cand", how="left")
    pp = pl.col(p).clip(0.0, 1 - EPS)
    other = (pl.when(pl.col("s1") == pl.col("_top1_s1"))
               .then(pl.when(pl.col("_n") > 1).then(pl.col("_top2")).otherwise(0.0))
               .otherwise(pl.col("_top1")))
    f32 = pl.Float32
    x = x.with_columns(other.fill_null(0.0).alias("_other")).with_columns(
        (pl.col("_n").fill_null(1) - 1).cast(f32).alias(f"c_n_other{suffix}"),
        pl.col("_other").cast(f32).alias(f"c_other_max{suffix}"),
        (pl.col(p) - pl.col("_other")).cast(f32).alias(f"c_margin{suffix}"),
        (pl.col("_n03").fill_null(0) - (pl.col(p) > 0.3).cast(pl.Int64)).clip(0, None).cast(f32).alias(f"c_n_other03{suffix}"),
        (pl.col(p) / pl.col("_sum").fill_null(0.0).clip(EPS, None)).clip(0.0, 1.0).cast(f32).alias(f"c_share{suffix}"),
        ((pp / (1 - pp)) / (1 + pl.col("_osum").fill_null(0.0))).cast(f32).alias(f"c_soft{suffix}"),
        (pl.col("_sum").fill_null(0.0) - pl.col(p)).clip(0.0, None).cast(f32).alias(f"c_sum_other{suffix}"),
        (pl.col("s1") == pl.col("_top1_s1")).fill_null(True).cast(f32).alias(f"c_is_best{suffix}"),
    )
    return x.drop(["_n", "_sum", "_osum", "_n03", "_top1", "_top1_s1", "_top2", "_other"])


# ------------------------------------------------------------------------------------------------ siblings
def _cpdist(a: List[str], b: List[str], scorer, workers: int) -> np.ndarray:
    from rapidfuzz import process
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32) / 100.0


def sibling_feats(e: pl.DataFrame, names: pl.DataFrame, p: str = "p1", top: int = 8, pmin: float = 0.05,
                  workers: int = -1, chunk: int = 4_000_000, verbose: bool = True) -> pl.DataFrame:
    """S2/S3 hold several duplicates of the same entity -> a true candidate usually resembles the S1's other
    high-p candidates. e must contain every candidate of an S1 with p1 >= pmin (the anchors)."""
    from rapidfuzz import fuzz
    t0 = time.time()
    base = (e.select("s1", "cand", pl.col(p).alias("_p")).with_row_index("_i")
             .join(names.select("cand", "nc", "ad", "nk"), on="cand", how="left", maintain_order="left")
             .with_columns([pl.col(c).fill_null("") for c in ("nc", "ad", "nk")]))
    anc = (base.filter(pl.col("_p") >= pmin).sort(["s1", "_p", "cand"], descending=[False, True, False])
               .group_by("s1", maintain_order=True).head(top)
               .with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
               .select("s1", pl.col("_i").alias("_j"), pl.col("_p").alias("_pd"), "_r"))
    pairs = base.select("_i", "s1").join(anc, on="s1").filter(pl.col("_i") != pl.col("_j")).drop("s1")
    nc, ad, nk = base["nc"], base["ad"], base["nk"]
    sn, sa = np.empty(pairs.height, np.float32), np.empty(pairs.height, np.float32)
    ii, jj = pairs["_i"], pairs["_j"]
    for lo in range(0, pairs.height, chunk):
        i, j = ii.slice(lo, chunk), jj.slice(lo, chunk)
        sn[lo:lo + len(i)] = _cpdist(nc.gather(i).to_list(), nc.gather(j).to_list(), fuzz.token_set_ratio, workers)
        sa[lo:lo + len(i)] = _cpdist(ad.gather(i).to_list(), ad.gather(j).to_list(), fuzz.token_set_ratio, workers)
    both_ad = (ad.gather(ii) != "") & (ad.gather(jj) != "")
    same_key = (nk.gather(ii) == nk.gather(jj)) & (nk.gather(ii) != "")
    pairs = pairs.with_columns(pl.Series("sn", sn), pl.Series("sa", sa), both_ad.alias("ba"), same_key.alias("sk"))
    pd_ = pl.col("_pd")
    agg = pairs.group_by("_i").agg(
        pl.len().cast(pl.Float32).alias("sib_n"),
        (pl.col("sn") * pd_).max().alias("sib_simp_max"),
        ((pl.col("sn") * pd_).sum() / pd_.sum().clip(EPS, None)).alias("sib_wsim"),
        pl.col("sn").max().alias("sib_sim_max"),
        pl.col("sn").sort_by("_r").first().alias("sib_top_sim"),
        pd_.sort_by("_r").first().alias("sib_top_p"),
        ((pl.col("sn") >= 0.9) & (pd_ >= 0.5)).sum().cast(pl.Float32).alias("sib_n09"),
        pd_.filter(pl.col("sk")).max().alias("sib_key_maxp"),
        (pl.col("sk") & (pd_ >= 0.5)).sum().cast(pl.Float32).alias("sib_key_n05"),
        (pl.col("sa") * pd_).filter(pl.col("ba")).max().alias("sib_asimp_max"),
        (0.5 * (pl.col("sn") + pl.col("sa")) * pd_).filter(pl.col("ba")).max().alias("sib_nasimp_max"),
    )
    fill = {"sib_n": 0.0, "sib_simp_max": 0.0, "sib_wsim": -1.0, "sib_sim_max": -1.0, "sib_top_sim": -1.0,
            "sib_top_p": 0.0, "sib_n09": 0.0, "sib_key_maxp": 0.0, "sib_key_n05": 0.0, "sib_asimp_max": -1.0,
            "sib_nasimp_max": -1.0}
    out = (base.select("_i").join(agg, on="_i", how="left").sort("_i")
               .select([pl.col(c).fill_null(v).cast(pl.Float32) for c, v in fill.items()]))
    if verbose:
        _log(f"siblings: {e.height:,} rows, {pairs.height:,} sibling pairs, {time.time() - t0:.1f}s")
    return e.hstack(out.get_columns())


# ------------------------------------------------------------------------------------------------ decisions / scoring
def select_fast(scored: pl.DataFrame, exclusivity: str = "soft", lam: float = 0.05303, empty_bias: float = 2.0,
                prune: float = 1e-3, hard_margin: float = 0.0, osum_ext: Optional[pl.DataFrame] = None,
                L: int = 16, p: str = "p") -> pl.DataFrame:
    """Links (s1, mid) with the pipeline's expected-F (exact) rule. Edges with p < prune (after exclusivity) are
    removed and their p mass is added to the S1's background rate (what decide_exact does with the tail anyway).
    exclusivity: none | soft | hard | soft_ext (osum_ext: cand, _ext = sum of stage-1 odds of the claiming S1 that
    are OUTSIDE `scored`, see external_osum; p' = o / (1 + sum_in_frame o + _ext) -- test-like competition)."""
    from ber import decide as D
    e = scored.select("s1", "cand", pl.col(p).alias("p"))
    if exclusivity == "soft":
        e = D.exclusivity_soft(e)
    elif exclusivity == "soft_ext":
        pc = pl.col("p").clip(0.0, 1 - EPS)
        o = pc / (1 - pc)
        e = e.join(osum_ext.select("cand", "_ext"), on="cand", how="left").with_columns(
            (o / (1 + o.sum().over("cand") + pl.col("_ext").fill_null(0.0).clip(0.0, None))).alias("p")).drop("_ext")
    elif exclusivity == "hard":
        e = D.exclusivity_hard(e, margin=hard_margin)
    elif exclusivity != "none":
        raise ValueError(exclusivity)
    tail = e.filter(pl.col("p") < prune).group_by("s1").agg(pl.col("p").sum().alias("_t"))
    k = (e.filter(pl.col("p") >= prune).join(tail, on="s1", how="left")
          .with_columns((lam + pl.col("_t").fill_null(0.0)).alias("_lam")).drop("_t"))
    r = D.resolve(k, method="exact", exclusivity="none", lam_missing="_lam", empty_bias=empty_bias, L=L)
    return r.filter(pl.col("sel")).select("s1", pl.col("cand").alias("mid"))


def external_osum(osum_full: pl.DataFrame, frame: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    """osum_full: cand, _osum (sum of odds over ALL S1 of the split). frame: s1, cand, p1 of the evaluated population.
    -> cand, _ext = odds mass of the S1 outside the frame."""
    pc = pl.col(p).cast(pl.Float64).clip(0.0, 1 - EPS)
    inn = frame.group_by("cand").agg((pc / (1 - pc)).sum().alias("_in"))
    return (osum_full.select("cand", "_osum").join(inn, on="cand", how="left")
                     .select("cand", (pl.col("_osum") - pl.col("_in").fill_null(0.0)).clip(0.0, None).alias("_ext")))


def select_threshold(scored: pl.DataFrame, t: float, p: str = "p") -> pl.DataFrame:
    from ber import decide as D
    e = scored.select("s1", "cand", pl.col(p).alias("p")).filter(pl.col("p") >= min(t, 0.05))
    return D.select_links(e, "threshold", exclusivity="none", t=t)


def score(links: pl.DataFrame, gt: pl.DataFrame, meta: pl.DataFrame) -> Dict[str, float]:
    """Macro F0.5 overall and per country. meta: s1, country (the evaluated S1 population)."""
    from ber import metric as M
    r = M.per_s1_scores(links, gt, meta.select("s1")).join(
        meta.select(pl.col("s1").cast(pl.Utf8), "country"), on="s1", how="left")
    out = {"ALL": float(r["f"].mean())}
    for c, f in r.group_by("country").agg(pl.col("f").mean()).sort("country").rows():
        out[str(c)] = float(f)
    return out


def ladder(scored: pl.DataFrame, links: pl.DataFrame, gt: pl.DataFrame, meta: pl.DataFrame, p: str = "p") -> Dict:
    """Blocking ceiling / oracle cut / actual + loss components (error_report.per_s1_table)."""
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("/root/proj/.claude/skills/er-error-analysis",
              os.path.join(here, "..", "..", ".claude", "skills", "er-error-analysis")):
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    import error_report as ER
    u = pl.Utf8
    per = ER.per_s1_table(scored.select(pl.col("s1").cast(u), pl.col("cand").cast(u), pl.col(p).alias("p")),
                          links.select(pl.col("s1").cast(u), pl.col("mid").cast(u)),
                          gt.select(pl.col("s1").cast(u), pl.col("mid").cast(u)),
                          meta.select(pl.col("s1").cast(u), "country"))
    comp, roll = ER.loss_decomposition(per)
    f, ceil, oc = float(per["f"].mean()), float(per["f_ceiling"].mean()), float(per["f_oracle_cut"].mean())
    return {"macro_f": f, "ceiling": ceil, "oracle_cut": oc, "blocking": 1 - ceil, "ranking": ceil - oc,
            "decision": oc - f, "components": {r[0]: round(r[3], 6) for r in comp.rows()},
            "buckets": {r[0]: round(r[1], 6) for r in roll.rows()}}


# ------------------------------------------------------------------------------------------------ stage-2 model
def fit_oof(frame: pl.DataFrame, cols: List[str], folds: np.ndarray, params: Dict, max_rounds: int = 3000,
            early_stopping: int = 100, inner_frac: float = 0.1, n_folds: int = 5):
    """GroupKFold OOF with the frozen folds (fold id per row, from metric.fold_of on the S1 id), early stopping on
    an inner S1-group split of each training fold (ber.model._fit). Returns raw OOF, best iters, summed gains."""
    from ber import model as MD
    cfg = MD.ModelConfig(params=dict(params), max_rounds=max_rounds, early_stopping=early_stopping,
                         inner_frac=inner_frac, feature_columns=list(cols), verbose=False)
    raw = np.zeros(frame.height, dtype=np.float64)
    iters, gain = [], np.zeros(len(cols))
    t0 = time.time()
    for k in range(n_folds):
        m = folds == k
        b, it = MD._fit(frame.filter(pl.Series(~m)), list(cols), cfg, seed=int(params.get("seed", 0)) + k)
        raw[m] = b.predict(frame.filter(pl.Series(m)).select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy(),
                           num_iteration=it)
        iters.append(int(it))
        gain += np.asarray(b.feature_importance("gain", iteration=it), dtype=np.float64)
        _log(f"  fold {k}: best_iter {it} ({time.time() - t0:.0f}s)")
    return raw, iters, dict(zip(cols, gain.tolist()))


def fit_final(frame: pl.DataFrame, cols: List[str], params: Dict, rounds: int):
    from ber import model as MD
    cfg = MD.ModelConfig(params=dict(params), feature_columns=list(cols), verbose=False)
    b, _ = MD._fit(frame, list(cols), cfg, seed=int(params.get("seed", 0)) + 99, rounds=rounds)
    return b


def isotonic(raw: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    return np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_)


def merge_p(all_edges: pl.DataFrame, rescored: pl.DataFrame, p_new: str = "p2") -> pl.DataFrame:
    """all_edges: s1, cand, p1 (every candidate); rescored: s1, cand, p2 (rows >= floor). -> s1, cand, p."""
    return (all_edges.select("s1", "cand", "p1").join(rescored.select("s1", "cand", p_new), on=["s1", "cand"], how="left")
                     .select("s1", "cand", pl.coalesce(pl.col(p_new), pl.col("p1")).alias("p")))
