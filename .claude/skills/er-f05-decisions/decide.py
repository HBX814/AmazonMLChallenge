#!/usr/bin/env python
"""
decide.py -- decision rules that turn calibrated per-pair match probabilities
into per-S1 match lists, optimised for the challenge metric (per-S1 macro F_0.5).

Input everywhere: a polars DataFrame of candidate edges
    s1   : S1 id (str or int)       -- the Source-1 entity
    cand : S2/S3 id (str or int)    -- a candidate record
    p    : float in [0,1]           -- CALIBRATED P(cand is a true match of s1)

Rules
-----
(a) decide_threshold(edges, t, fallback_t=None)
      select p >= t  [optionally: if nothing passes, take the argmax if p_max >= fallback_t]
(b) decide_exact(edges, lam_missing, ...)          <- exact expected-F maximiser
      per S1: sort p desc, choose m in 0..L maximising E[F_beta(top-m)] under independent
      Bernoulli truths; truth may also contain matches OUTSIDE the candidate list
      (blocking misses ~ Poisson(lam_missing)) and the low-p tail beyond rank L is folded
      into the same background count (Poisson(sum tail p)).
    decide_fast(edges, lam_missing, kind="series"|"plugin")   <- O(n) approximations
(c) exclusivity_soft(edges)   -> p' = o/(1+sum_cand o),  o = p/(1-p)
      (posterior under "each S2/S3 record belongs to at most one S1", independent priors)
    exclusivity_hard(edges, margin)   -> keep only the best S1 per record
    resolve(edges, ...)       -> soft/hard exclusivity -> per-S1 decision -> conflict
                                 repair loop (losers removed, re-decide) -> hard guarantee

Output helpers
--------------
    write_id_list_tsv(sel_edges, all_s1, path, header2="matched_entity_ids")
    (use header2="candidate_entity_ids" and all edges for candidate_pairs.tsv)

Everything is vectorised (polars window functions + numpy batches); tested on
10M edges (see R5_metric_decision.md for timings).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

BETA = 0.5


# =============================================================================
# numpy cores (padded, per-S1 batches)
# =============================================================================
def poisson_pmf_trunc(lam: np.ndarray, R: int) -> np.ndarray:
    """(B,) rates -> (B,R) Poisson pmf; the last bucket absorbs the tail mass."""
    lam = np.maximum(np.asarray(lam, dtype=np.float64), 0.0)
    k = np.arange(R, dtype=np.float64)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, R)))])
    with np.errstate(divide="ignore"):
        loglam = np.where(lam > 0, np.log(np.maximum(lam, 1e-300)), -np.inf)
    with np.errstate(invalid="ignore"):  # 0 * log(0) in column 0 -> fixed on the next line
        logp = -lam[:, None] + k[None, :] * loglam[:, None] - logfact[None, :]
    logp[:, 0] = -lam  # 0 * -inf guard
    pmf = np.exp(logp)
    pmf[:, -1] += np.clip(1.0 - pmf.sum(1), 0.0, None)
    return pmf


def expected_f_topm(P: np.ndarray, bg: np.ndarray, beta: float = BETA) -> np.ndarray:
    """EXACT E[F_beta] of predicting the top-m candidates, for m = 0..L.

    P  : (B, L) calibrated probabilities, each row sorted DESCENDING, zero padded.
    bg : (B, R) pmf of the number of true matches that are NOT among the L listed
         candidates (blocking misses + low-p tail). bg[:,0] = P(no such match).
    Returns EF : (B, L+1).  EF[:,0] = P(truth empty) = prod(1-p) * bg[:,0].

    Model: Z_j ~ Bernoulli(p_j) independent, background count Q ~ bg independent,
    |T| = sum Z + Q, tp(m) = sum_{j<=m} Z_j,  F = (1+b2) tp / (b2 |T| + m)  (m>=1).
    Joint pmf of (A = tp, C = rest) = prefix_m (x) (suffix_m * bg); cost O(B L^2 (L+R)).
    """
    P = np.clip(np.asarray(P, dtype=np.float64), 0.0, 1.0)
    bg = np.asarray(bg, dtype=np.float64)
    B, L = P.shape
    R = bg.shape[1]
    b2 = beta * beta
    pre = np.zeros((L + 1, B, L + 1))
    pre[0, :, 0] = 1.0
    for j in range(L):
        pj = P[:, j:j + 1]
        pre[j + 1] = pre[j] * (1.0 - pj)
        pre[j + 1][:, 1:] += pre[j][:, :-1] * pj
    sb = np.zeros((L + 1, B, L + R))
    sb[L, :, :R] = bg
    for j in range(L - 1, -1, -1):
        pj = P[:, j:j + 1]
        sb[j] = sb[j + 1] * (1.0 - pj)
        sb[j][:, 1:] += sb[j + 1][:, :-1] * pj
    EF = np.empty((B, L + 1))
    EF[:, 0] = pre[L][:, 0] * bg[:, 0]
    a = np.arange(L + 1, dtype=np.float64)[:, None]
    c = np.arange(L + R, dtype=np.float64)[None, :]
    for m in range(1, L + 1):
        G = (1.0 + b2) * a / (b2 * (a + c) + m)          # (L+1, L+R)
        EF[:, m] = ((pre[m] @ G) * sb[m]).sum(1)
    return EF


def expected_f_approx(P: np.ndarray, lam_bg: np.ndarray, beta: float = BETA,
                      kind: str = "series") -> np.ndarray:
    """O(L) approximations of expected_f_topm (same shapes; lam_bg = E[background]).

    kind="plugin": (1+b2) S1_m / (m + b2 mu)                 (ratio of expectations)
    kind="series": leave-one-out + 2nd-order delta method:
        (1+b2) [S1/c + b2 S2/c^2 + b2^2 (V S1 - S2 + 2 S3)/c^3],  c = m + b2 (1 + mu)
      with S_r = sum_{j<=m} p_j^r, mu = sum p + lam_bg, V = sum p(1-p) + lam_bg.
    EF[:,0] = prod(1-p) * exp(-lam_bg) in both.
    """
    P = np.clip(np.asarray(P, dtype=np.float64), 0.0, 1.0)
    lam_bg = np.broadcast_to(np.asarray(lam_bg, dtype=np.float64), (P.shape[0],))
    B, L = P.shape
    b2 = beta * beta
    m = np.arange(1, L + 1, dtype=np.float64)[None, :]
    S1 = np.cumsum(P, 1)
    mu = (P.sum(1) + lam_bg)[:, None]
    EF = np.empty((B, L + 1))
    with np.errstate(divide="ignore"):
        EF[:, 0] = np.exp(np.log1p(-np.minimum(P, 1 - 1e-15)).sum(1) - lam_bg)
    EF[:, 0] = np.where((P >= 1.0).any(1), 0.0, EF[:, 0])
    if kind == "plugin":
        EF[:, 1:] = (1 + b2) * S1 / (m + b2 * mu)
    elif kind == "series":
        S2 = np.cumsum(P * P, 1)
        S3 = np.cumsum(P ** 3, 1)
        V = ((P * (1 - P)).sum(1) + lam_bg)[:, None]
        c = m + b2 * (1 + mu)
        EF[:, 1:] = (1 + b2) * (S1 / c + b2 * S2 / c ** 2 + b2 * b2 * (V * S1 - S2 + 2 * S3) / c ** 3)
    else:
        raise ValueError(kind)
    return EF


def best_m(EF: np.ndarray, empty_bias: float = 1.0, n_valid: Optional[np.ndarray] = None) -> np.ndarray:
    """argmax_m EF (ties -> smaller m). empty_bias multiplies EF[:,0] (>1 favours empty)."""
    E = EF.copy()
    E[:, 0] *= empty_bias
    if n_valid is not None:  # never pick padded slots
        L = E.shape[1] - 1
        E[np.arange(L + 1)[None, :] > n_valid[:, None]] = -np.inf
    return E.argmax(1)


def brute_force_expected_f(p: Sequence[float], S: Sequence[int], bg: Sequence[float] = (1.0,),
                           beta: float = BETA) -> float:
    """Enumerate all 2^n truth vectors (n <= ~12). For tests only."""
    import itertools
    p = list(p)
    S = set(S)
    b2 = beta * beta
    tot = 0.0
    for z in itertools.product([0, 1], repeat=len(p)):
        pz = 1.0
        for zi, pi in zip(z, p):
            pz *= pi if zi else 1 - pi
        if pz == 0:
            continue
        tp = sum(z[j] for j in S)
        kc = sum(z)
        for q, pq in enumerate(bg):
            if pq == 0:
                continue
            k = kc + q
            if not S:
                f = 1.0 if k == 0 else 0.0
            else:
                f = 0.0 if tp == 0 else (1 + b2) * tp / (b2 * k + len(S))
            tot += pz * pq * f
    return tot


# =============================================================================
# polars wrappers (long format edges)
# =============================================================================
def _pl():
    import polars as pl
    return pl


def add_rank(edges, s1="s1", p="p", cand="cand"):
    """Sort by s1, p desc (ties by cand) and add 0-based `rank` and dense `code` per s1."""
    pl = _pl()
    e = edges.sort([s1, p, cand], descending=[False, True, False])
    return e.with_columns(
        pl.int_range(pl.len()).over(s1).alias("rank"),
        ((pl.col(s1) != pl.col(s1).shift(1)).fill_null(True).cum_sum() - 1).alias("code"),
    )


def decide_threshold(edges, t: float, fallback_t: Optional[float] = None, p="p", s1="s1"):
    """Rule (a): global threshold, optional best-candidate fallback. Adds bool `sel`."""
    pl = _pl()
    sel = pl.col(p) >= t
    if fallback_t is not None:
        # at most one fallback per S1: the rank-0 candidate (ties broken by cand id)
        if "rank" not in edges.columns:
            edges = add_rank(edges, s1, p, "cand" if "cand" in edges.columns else p)
        none_pass = (pl.col(p).max().over(s1) < t)
        sel = sel | (none_pass & (pl.col("rank") == 0) & (pl.col(p) >= fallback_t))
    return edges.with_columns(sel.alias("sel"))


def _padded(e, L, p="p"):
    code = e["code"].to_numpy()
    rank = e["rank"].to_numpy()
    pv = e[p].to_numpy().astype(np.float64)
    B = int(code.max()) + 1 if len(code) else 0
    n = np.bincount(code, minlength=B)
    mask = rank < L
    P = np.zeros((B, L))
    P[code[mask], rank[mask]] = pv[mask]
    tail = np.bincount(code[~mask], weights=pv[~mask], minlength=B)
    return P, tail, np.minimum(n, L), code, rank


def _lam_per_s1(e, lam_missing, B):
    if isinstance(lam_missing, str):  # column name, constant within S1
        pl = _pl()
        v = e.group_by("code").agg(pl.col(lam_missing).first()).sort("code")[lam_missing].to_numpy()
        return v.astype(np.float64)
    return np.full(B, float(lam_missing))


def decide_exact(edges, lam_missing=0.0, L: int = 16, R: int = 12, beta: float = BETA,
                 empty_bias: float = 1.0, chunk: int = 50_000, p="p", s1="s1", cand="cand",
                 return_s1_table: bool = False):
    """Rule (b), exact. lam_missing: float or column name (expected #true matches of this S1
    that are NOT in its candidate list, i.e. blocking misses; estimate on validation)."""
    pl = _pl()
    e = add_rank(edges, s1, p, cand)
    if e.height == 0:
        return e.with_columns(pl.lit(False).alias("sel"))
    P, tail, nval, code, rank = _padded(e, L, p)
    B = P.shape[0]
    lam = _lam_per_s1(e, lam_missing, B) + tail
    mstar = np.empty(B, dtype=np.int64)
    ef_best = np.empty(B)
    ef_empty = np.empty(B)
    for s in range(0, B, chunk):
        sl = slice(s, min(B, s + chunk))
        EF = expected_f_topm(P[sl], poisson_pmf_trunc(lam[sl], R), beta)
        mm = best_m(EF, empty_bias, nval[sl])
        mstar[sl] = mm
        ef_best[sl] = EF[np.arange(len(mm)), mm]
        ef_empty[sl] = EF[:, 0]
    out = e.with_columns(pl.Series("sel", rank < mstar[code]))
    if return_s1_table:
        first = e.filter(pl.col("rank") == 0).select(s1, "code").sort("code")
        tab = first.with_columns(pl.Series("m_star", mstar), pl.Series("ef_best", ef_best),
                                 pl.Series("p_empty", ef_empty))
        return out, tab
    return out


def decide_fast(edges, lam_missing=0.0, kind: str = "series", beta: float = BETA,
                empty_bias: float = 1.0, p="p", s1="s1", cand="cand"):
    """Rule (b), O(n) approximation, pure polars window functions (no padding, no L cap)."""
    pl = _pl()
    b2 = beta * beta
    e = add_rank(edges, s1, p, cand)
    lam = pl.col(lam_missing) if isinstance(lam_missing, str) else pl.lit(float(lam_missing))
    pc = pl.col(p).clip(0.0, 1.0)
    m = (pl.col("rank") + 1).cast(pl.Float64)
    e = e.with_columns(
        pc.cum_sum().over(s1).alias("_S1"),
        (pc * pc).cum_sum().over(s1).alias("_S2"),
        (pc ** 3).cum_sum().over(s1).alias("_S3"),
        (pc.sum().over(s1) + lam).alias("_mu"),
        ((pc * (1 - pc)).sum().over(s1) + lam).alias("_V"),
        ((1 - pc).clip(1e-300, 1.0).log().sum().over(s1) - lam).exp().alias("_e0"),
    )
    if kind == "plugin":
        ef = (1 + b2) * pl.col("_S1") / (m + b2 * pl.col("_mu"))
    else:
        c = m + b2 * (1 + pl.col("_mu"))
        ef = (1 + b2) * (pl.col("_S1") / c + b2 * pl.col("_S2") / c ** 2
                         + b2 * b2 * (pl.col("_V") * pl.col("_S1") - pl.col("_S2") + 2 * pl.col("_S3")) / c ** 3)
    e = e.with_columns(ef.alias("_ef"))
    # best m per S1 among m>=1 (first max -> smallest m), compare with empty
    e = e.with_columns(pl.col("_ef").max().over(s1).alias("_efmax"))
    e = e.with_columns(
        pl.when(pl.col("_ef") == pl.col("_efmax")).then(pl.col("rank")).otherwise(None)
          .min().over(s1).alias("_argmax"))
    choose_empty = (pl.col("_e0") * empty_bias) >= pl.col("_efmax")
    e = e.with_columns((~choose_empty & (pl.col("rank") <= pl.col("_argmax"))).alias("sel"))
    return e.drop(["_S1", "_S2", "_S3", "_mu", "_V", "_e0", "_ef", "_efmax", "_argmax"])


# ----------------------------------------------------------------------------- exclusivity
def exclusivity_soft(edges, p="p", cand="cand", out="p", eps=1e-6):
    """At-most-one-owner posterior:  p'_ij = o_ij / (1 + sum_l o_lj),  o = p/(1-p).
    Leaves records with a single candidate S1 almost unchanged: p' = p (exactly)."""
    pl = _pl()
    pc = pl.col(p).clip(0.0, 1.0 - eps)
    o = pc / (1 - pc)
    # p' = o / (1 + sum_{l != i} o_l + o_i)  -> for a lone edge: o/(1+o) = p
    return edges.with_columns((o / (1 + o.sum().over(cand))).alias(out))


def exclusivity_hard(edges, margin: float = 0.0, p="p", cand="cand", s1="s1", drop=True):
    """Keep, for every record, only its highest-p S1 edge; if margin>0 also drop the winner
    when (best - second best) < margin. drop=False keeps rows with p set to 0."""
    pl = _pl()
    e = edges.sort([cand, p, s1], descending=[False, True, False]).with_columns(
        pl.int_range(pl.len()).over(cand).alias("_cr"))
    second = pl.col(p).filter(pl.col("_cr") == 1).first().over(cand).fill_null(0.0)
    keep = (pl.col("_cr") == 0) & ((pl.col(p) - second) >= margin)
    if drop:
        return e.filter(keep).drop("_cr")
    return e.with_columns(pl.when(keep).then(pl.col(p)).otherwise(0.0).alias(p)).drop("_cr")


def decide(edges, method="exact", **kw):
    if method == "exact":
        return decide_exact(edges, **kw)
    if method in ("series", "plugin"):
        return decide_fast(edges, kind=method, **kw)
    if method == "threshold":
        return decide_threshold(edges, **kw)
    raise ValueError(method)


def resolve(edges, method="exact", exclusivity="soft", repair_iters: int = 2, hard_margin=0.0,
            s1="s1", cand="cand", p="p", **kw):
    """Full decision stage: exclusivity -> per-S1 decision -> conflict repair -> guarantee.
    Returns edges (with possibly modified p) and boolean `sel`; at most one selected S1 per cand."""
    pl = _pl()
    extra = [kw["lam_missing"]] if isinstance(kw.get("lam_missing"), str) else []
    e = edges.select(s1, cand, p, *extra)
    if exclusivity == "soft":
        e = exclusivity_soft(e, p=p, cand=cand)
    elif exclusivity == "hard":
        e = exclusivity_hard(e, margin=hard_margin, p=p, cand=cand, s1=s1)
    elif exclusivity not in (None, "none"):
        raise ValueError(exclusivity)
    d = decide(e, method, **kw).select(s1, cand, p, *extra, "sel")
    for _ in range(repair_iters):
        nsel = pl.col("sel").cast(pl.Int32).sum().over(cand)
        best_sel = pl.col(p).filter(pl.col("sel")).max().over(cand)
        d = d.with_columns((pl.col("sel") & (nsel > 1) & (pl.col(p) < best_sel)).alias("_loser"))
        affected = d.filter(pl.col("_loser")).select(s1).unique()
        if affected.height == 0:
            d = d.drop("_loser")
            break
        # losing edges are removed; only the affected S1s are re-decided
        keep = d.join(affected, on=s1, how="anti").drop("_loser")
        redo = d.join(affected, on=s1, how="semi").filter(~pl.col("_loser")).select(s1, cand, p, *extra)
        d = pl.concat([keep, decide(redo, method, **kw).select(s1, cand, p, *extra, "sel")], how="vertical_relaxed")
    # hard guarantee (ties): keep one selected S1 per cand
    ds =d.filter(pl.col("sel")).sort([cand, p, s1], descending=[False, True, False]) \
          .unique(subset=[cand], keep="first", maintain_order=True)
    return d.drop("sel").join(ds.select(s1, cand).with_columns(pl.lit(True).alias("sel")),
                              on=[s1, cand], how="left").with_columns(pl.col("sel").fill_null(False))


# ----------------------------------------------------------------------------- contract API
def assign_exclusive(scored, margin: float = 0.0, mode: str = "hard"):
    """CONTRACT: exclusivity step on scored pairs (s1, cand, p).
    mode="hard": keep, for each cand, only its best s1 (dropped if best-second < margin).
    mode="soft": keep all rows, replace p by the at-most-one-owner posterior o/(1+sum o)."""
    if mode == "hard":
        return exclusivity_hard(scored, margin=margin)
    if mode == "soft":
        return exclusivity_soft(scored)
    raise ValueError(mode)


_METHOD_ALIASES = {"expected_f": "exact", "expected_f_exact": "exact", "exact": "exact",
                   "expected_f_fast": "series", "series": "series", "plugin": "plugin",
                   "threshold": "threshold"}


def select_links(scored, method: str = "expected_f", exclusivity: str = "soft", **kw):
    """CONTRACT: scored pairs (s1, cand, p) -> final links (s1, mid), each mid used at most once.
    method: expected_f (exact, default) | expected_f_fast (series approx) | plugin | threshold
    kw: lam_missing, empty_bias, L, R (expected-F) or t, fallback_t (threshold); repair_iters.
    NOTE: for method="threshold" exclusivity is still enforced by the final repair/guarantee."""
    pl = _pl()
    m = _METHOD_ALIASES[method]
    r = resolve(scored.select("s1", "cand", "p"), method=m, exclusivity=exclusivity, **kw)
    return r.filter(pl.col("sel")).select("s1", pl.col("cand").alias("mid"))


# ----------------------------------------------------------------------------- output
def write_id_lists(s1_ids, pairs, path: str, list_col: str = "matched_entity_ids"):
    """CONTRACT: write one row per S1 in s1_ids (empty list allowed). pairs: (s1, mid) or (s1, cand[, p])."""
    pl = _pl()
    e = pairs.rename({"mid": "cand"}) if "mid" in pairs.columns else pairs
    if "p" not in e.columns:
        e = e.with_columns(pl.lit(0.0).alias("p"))
    write_id_list_tsv(e, list(s1_ids), path, header2=list_col, only_selected=False)


def write_id_list_tsv(sel_edges, all_s1: Sequence[str], path: str, header2="matched_entity_ids",
                      s1="s1", cand="cand", p="p", only_selected=True):
    """Write the submission-format TSV: one row per S1 in `all_s1` (empty list allowed),
    ids ordered by p desc, no duplicates, tab separated, no quoting."""
    pl = _pl()
    e = sel_edges.filter(pl.col("sel")) if (only_selected and "sel" in sel_edges.columns) else sel_edges
    e = e.unique(subset=[s1, cand])
    agg = (e.sort([s1, p], descending=[False, True])
             .group_by(s1, maintain_order=True)
             .agg(pl.col(cand).cast(pl.Utf8).str.join(",").alias(header2)))
    base = pl.DataFrame({"source1_entity_id": pl.Series(list(all_s1), dtype=pl.Utf8)})
    out = base.join(agg.with_columns(pl.col(s1).cast(pl.Utf8)), left_on="source1_entity_id",
                    right_on=s1, how="left").with_columns(pl.col(header2).fill_null(""))
    out.select("source1_entity_id", header2).write_csv(path, separator="\t", quote_style="never")
    return out


if __name__ == "__main__":
    # tiny demo
    import polars as pl
    ed = pl.DataFrame({"s1": ["A"] * 4 + ["B"] * 2 + ["C"],
                       "cand": ["x1", "x2", "x3", "x4", "x2", "y1", "z1"],
                       "p": [0.97, 0.90, 0.55, 0.05, 0.60, 0.30, 0.02]})
    print(resolve(ed, method="exact", exclusivity="soft", lam_missing=0.1).sort(["s1", "p"], descending=[False, True]))
