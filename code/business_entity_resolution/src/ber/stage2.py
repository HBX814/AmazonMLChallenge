# -*- coding: utf-8 -*-
"""stage2.py -- stage-2 "collective" re-scoring on top of the stage-1 calibrated p (Master Bolt ER).
Reference copy: .claude/skills/er-matcher-training/stage2.py (keep both files identical).

Why: stage 1 scores every (S1, candidate) pair from pair features. Stage 2 re-scores the uncertain part of the
graph with features computed FROM the stage-1 p of the whole country frame: how the pair ranks inside its S1's
list, what the stage-1 expected-F decision would do, how strongly OTHER S1 claim the same record (exclusivity: a
record belongs to at most one S1) and whether the candidate resembles the S1's other confident candidates (S2/S3
hold several copies of one entity). Measured (work/reports/diag_results.md, prototype work/infra/diag_stage2_*.py):
OOF macro F0.5 0.98152 -> 0.98411 (guard G1 0.98361), honest holdout HB 0.98189 -> 0.98452 (G1 0.98397).

Every stage-2 feature is LABEL-FREE and never uses the country value.

Contract
--------
Stage2Config                        all settings (floor, sibling anchors, stage-1 decision params, feature groups,
                                    guard, LightGBM params, folds)
feature_columns(cfg) -> list        the stage-2 model's columns (orig stage-1 features + W + D + C + S groups)
build_frame(scored, feats, pool_norm, cfg, population=None, stage1_bundle=None) -> frame
    scored    : s1, cand, p [, p_raw] [, label]   stage-1 CALIBRATED p for EVERY S1 of the split/country (= all
                claimants of the records; at train time OOF p for the sampled S1 + final-model p for the others,
                at test time the final-model p of all test S1 of the country).
    population: S1 ids whose pairs are re-scored (None = every S1 in `scored`); their FULL lists are needed.
    feats     : DataFrame | LazyFrame | parquet path(s) with s1, cand + the stage-1 feature columns
                (only the rows of the re-scored pairs are read; paths are streamed in batches).
    pool_norm : DataFrame | LazyFrame | parquet path(s) of the normalized pool (entity_id, name_core, addr_norm,
                [name_key]).
    stage1_bundle: ber.model bundle, used only when `scored` has no p_raw column (p1_logit = logit(raw score)).
    -> one row per re-scored pair (population rows with p >= cfg.floor): s1, cand, p1 [, label] + features.
fit(frame, cfg) -> (bundle, oof)    5-fold OOF on the frozen folds metric.fold_of(s1, n_folds, fold_seed)
                                    (early stopping on an inner S1-group split, rows sorted by (s1, cand)),
                                    isotonic calibration on the OOF raw score, final booster with
                                    final_rounds_mult x mean(best_iter) rounds.
                                    oof: s1, cand, label, p1, p2_raw, p2, p  (p = guarded stage-2 probability)
predict(bundle, frame, scored=None, guard=None)
                                    scored=None -> re-scored rows s1, cand, p1, p2, p (p = guarded)
                                    scored given -> merge(scored, ...) = full scored frame with p replaced on the
                                    re-scored rows (rows below the floor keep p1), same row order as `scored`.
merge(scored, rescored) -> scored   rows not re-scored keep their stage-1 p
apply_guard(rescored, guard)        G0: p = p2 | G1: house-number-mismatch pairs (numbers on both sides, none
                                    equal, no common base) get p = min(p1, p2) | G2: those pairs keep p1
save_bundle(bundle, dir) / load_bundle(dir)      model_s2.txt (LightGBM text) + stage2.json
stage1_predict_stream(stage1_bundle, feats, skip_s1=None) -> s1, cand, p, p_raw [, label]
                                    stage-1 final-model p for every train pair (the competition frame at train time)

Feature groups (names kept identical to the prototype so its saved model stays applicable)
    W_FEATS  within-S1 : p1, logit(raw), rank, gap to top / next / previous, #p1>0.5 / >0.2, sum p1, sum excluding
                         self, list length, top-1 / top-2 p1 of the S1
    D_FEATS  decision  : stage-1 expected-F decision on p1 alone (selected?, rank - m*, m*, P(empty))
    C_FEATS  competition over EVERY S1 of the frame claiming the record: #other claimants, max p1 of the others,
                         margin, #others with p1>0.3, share of the record's p mass, soft posterior o/(1+sum o),
                         sum over the others, is-argmax (ties -> smallest S1 id)
    S_FEATS  siblings  : vs the S1's top-8 other candidates with p1 >= 0.05: max name-sim*p1, p1-weighted mean sim,
                         best sim, sim / p1 of the top anchor, #anchors sim>=0.9 & p1>=0.5, max p1 / count of anchors
                         with the same name key, max addr-sim*p1 and (name+addr)/2*p1 when both addresses exist

Scale (prototype, Modal 16 CPU): test India 48.1M pairs / 4.21M re-scored in 112 s, 20 GB peak; train frame for
19.5M OOF pairs ~22 s; 5-fold training on 1.55M x 108 features 133 s. Ids of the form S<d>-<digits> are encoded
to int64 internally (faster joins, half the memory) and decoded on output; any other id form is kept as is.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

try:
    import lightgbm as lgb
except ImportError as e:            # pragma: no cover
    raise ImportError("pip install lightgbm==4.7.0 (MIT)") from e

try:                                                    # inside the ber package
    from . import decide as _D                          # type: ignore
    from . import metric as _M                          # type: ignore
    from . import model as _MD                          # type: ignore
except ImportError:                                     # reference layout (.claude/skills/<skill>/)
    import importlib.util as _ilu
    import sys as _sys
    _here = os.path.dirname(os.path.abspath(__file__))

    def _load(name, rel):
        if name in _sys.modules:
            return _sys.modules[name]
        for p in (os.path.join(_here, f"{name}.py"), os.path.join(_here, "..", rel, f"{name}.py")):
            if os.path.exists(p):
                spec = _ilu.spec_from_file_location(name, p)
                mod = _ilu.module_from_spec(spec)
                _sys.modules[name] = mod
                spec.loader.exec_module(mod)  # type: ignore
                return mod
        raise ImportError(name)
    _M = _load("metric", "er-f05-decisions")
    _D = _load("decide", "er-f05-decisions")
    _MD = _load("model", "er-matcher-training")

EPS = 1e-6

W_FEATS = ["p1", "p1_logit", "w_rank", "w_gap_top", "w_gap_next", "w_gap_prev", "w_n05", "w_n02", "w_sum",
           "w_sum_other", "w_ncand", "w_top1", "w_top2"]
D_FEATS = ["w_sel1", "w_rank_m", "w_mstar", "w_pempty"]
C_FEATS = ["c_n_other", "c_other_max", "c_margin", "c_n_other03", "c_share", "c_soft", "c_sum_other", "c_is_best"]
S_FEATS = ["sib_n", "sib_simp_max", "sib_wsim", "sib_sim_max", "sib_top_sim", "sib_top_p", "sib_n09", "sib_key_maxp",
           "sib_key_n05", "sib_asimp_max", "sib_nasimp_max"]
GUARD_COLS = ["hn_both", "hn_exact", "hn_base"]          # stage-1 features the guard reads
GROUPS = ("orig", "within", "decision", "competition", "sibling")

S2_PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
             "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
             "verbose": -1, "num_threads": 0, "seed": 0, "deterministic": True, "force_row_wise": True}


@dataclass
class Stage2Config:
    floor: float = 1e-3                      # population pairs with stage-1 p >= floor are re-scored
    sib_top: int = 8                         # sibling anchors per S1 (top by p1, excluding the pair itself)
    sib_pmin: float = 0.05                   # anchors need p1 >= sib_pmin
    lam_missing: float = 0.05303060038344832  # stage-1 decision (decision features): decision.json of stage 1
    empty_bias: float = 2.0
    orig_features: Optional[List[str]] = None  # stage-1 features fed to stage 2 (None = features.FEATURE_COLUMNS);
                                               # pass the stage-1 bundle's feature_columns
    groups: List[str] = field(default_factory=lambda: list(GROUPS))
    guard: str = "G1"                        # G0 | G1 | G2 (see apply_guard)
    n_folds: int = 5
    fold_seed: int = 0                       # frozen folds = metric.fold_of(s1, n_folds, fold_seed)
    params: Dict = field(default_factory=lambda: dict(S2_PARAMS))
    max_rounds: int = 3000
    early_stopping: int = 100
    inner_frac: float = 0.10
    final_rounds_mult: float = 1.1
    workers: int = -1                        # rapidfuzz threads for the sibling similarities
    sib_chunk: int = 4_000_000
    feats_batch: int = 2_000_000             # parquet streaming batch (rows)
    verbose: bool = True


def _log(cfg: Optional[Stage2Config], *a):
    if cfg is None or cfg.verbose:
        print(time.strftime("%H:%M:%S"), "[stage2]", *a, flush=True)


def feature_columns(cfg: Optional[Stage2Config] = None) -> List[str]:
    cfg = cfg or Stage2Config()
    bad = [g for g in cfg.groups if g not in GROUPS]
    if bad:
        raise ValueError(f"unknown stage-2 feature groups {bad}; valid: {GROUPS}")
    orig = list(cfg.orig_features) if cfg.orig_features is not None else list(_MD._FC)
    cols: List[str] = []
    for g, c in (("orig", orig), ("within", W_FEATS), ("decision", D_FEATS), ("competition", C_FEATS),
                 ("sibling", S_FEATS)):
        if g in cfg.groups:
            cols += [x for x in c if x not in cols]
    return cols


# ------------------------------------------------------------------------------------------------ ids
_ID_RE = r"^S\d-(0|[1-9]\d{0,9})$"          # round-trips exactly through the int64 code


def _encodable(*series: pl.Series) -> bool:
    for s in series:
        if s.dtype.is_integer():
            continue
        if s.dtype not in (pl.Utf8, pl.String) or s.null_count() or not bool(s.str.contains(_ID_RE).all()):
            return False
    return True


def _enc(col: str) -> pl.Expr:
    """'S2-166376419' -> 2*1e10 + 166376419 (int64); integer columns pass through."""
    c = pl.col(col)
    return (c.str.slice(1, 1).cast(pl.Int64) * 10_000_000_000 + c.str.slice(3).cast(pl.Int64)).alias(col)


def _enc_if(col: str, dtype) -> pl.Expr:
    return pl.col(col).cast(pl.Int64) if dtype.is_integer() else _enc(col)


def _dec(col: str) -> pl.Expr:
    c = pl.col(col)
    return pl.format("S{}-{}", c // 10_000_000_000, c % 10_000_000_000).alias(col)


def _as_ids(x, name: str = "s1") -> pl.DataFrame:
    if isinstance(x, pl.DataFrame):
        return x.select(pl.col(name if name in x.columns else x.columns[0]).alias(name)).unique()
    if isinstance(x, pl.Series):
        return x.rename(name).to_frame().unique()
    return pl.DataFrame({name: list(x)}).unique()


# ------------------------------------------------------------------------------------------------ within-S1
def within_s1_feats(e: pl.DataFrame, p: str = "p1", praw: Optional[str] = None) -> pl.DataFrame:
    """e: s1, cand, p1 [, p1raw] with ALL candidates of each S1 (the features describe the whole list).
    Returns e sorted by (s1, p1 desc, cand) with W_FEATS added (p1_logit from the raw score when given)."""
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
    d, tab = _D.decide_exact(e.select("s1", "cand", pl.col(p).alias("p")), lam_missing=lam, empty_bias=empty_bias,
                             return_s1_table=True)
    out = (d.select("s1", "cand", "rank", pl.col("sel").cast(pl.Float32).alias("w_sel1"))
            .join(tab.select("s1", "m_star", "p_empty"), on="s1", how="left")
            .select("s1", "cand", "w_sel1",
                    (pl.col("rank") - pl.col("m_star")).cast(pl.Float32).alias("w_rank_m"),
                    pl.col("m_star").cast(pl.Float32).alias("w_mstar"),
                    pl.col("p_empty").cast(pl.Float32).alias("w_pempty")))
    return e.join(out, on=["s1", "cand"], how="left", maintain_order="left")


# ------------------------------------------------------------------------------------------------ competition
def competition_aggregates(allp: pl.DataFrame, cands: Optional[pl.DataFrame] = None, p: str = "p1") -> pl.DataFrame:
    """Per record over ALL claiming S1: #claimants, sum p, sum odds, #p>0.3, top-1 p (+ its S1), top-2 p.
    Ties for the top claimant go to the smallest S1 id (deterministic)."""
    a = allp.select("s1", "cand", pl.col(p).cast(pl.Float64).alias("_p"))
    if cands is not None:
        a = a.join(cands.select("cand").unique(), on="cand", how="semi")
    pc = pl.col("_p").clip(0.0, 1 - EPS)
    return a.group_by("cand").agg(
        pl.len().alias("_n"), pl.col("_p").sum().alias("_sum"), (pc / (1 - pc)).sum().alias("_osum"),
        (pl.col("_p") > 0.3).sum().alias("_n03"), pl.col("_p").max().alias("_top1"),
        pl.col("s1").sort_by(["_p", "s1"], descending=[True, False]).first().alias("_top1_s1"),
        pl.col("_p").top_k(2).min().alias("_top2"))


def competition_feats(e: pl.DataFrame, agg: pl.DataFrame, p: str = "p1", suffix: str = "") -> pl.DataFrame:
    """C_FEATS for the rows of e from competition_aggregates over the whole frame (which must contain e's rows)."""
    x = e.join(agg, on="cand", how="left", maintain_order="left")
    pp = pl.col(p).clip(0.0, 1 - EPS)
    other = (pl.when(pl.col("s1") == pl.col("_top1_s1"))
               .then(pl.when(pl.col("_n") > 1).then(pl.col("_top2")).otherwise(0.0))
               .otherwise(pl.col("_top1")))
    f32 = pl.Float32
    x = x.with_columns(other.fill_null(0.0).alias("_other")).with_columns(
        (pl.col("_n").fill_null(1) - 1).cast(f32).alias(f"c_n_other{suffix}"),
        pl.col("_other").cast(f32).alias(f"c_other_max{suffix}"),
        (pl.col(p) - pl.col("_other")).cast(f32).alias(f"c_margin{suffix}"),
        (pl.col("_n03").fill_null(0) - (pl.col(p) > 0.3).cast(pl.Int64)).clip(0, None).cast(f32)
        .alias(f"c_n_other03{suffix}"),
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
                  workers: int = -1, chunk: int = 4_000_000, cfg: Optional[Stage2Config] = None) -> pl.DataFrame:
    """S_FEATS. e must contain every candidate of an S1 with p1 >= pmin (the anchors).
    names: cand, nc (name_core), ad (addr_norm), nk (name_key)."""
    from rapidfuzz import fuzz
    t0 = time.time()
    base = (e.select("s1", "cand", pl.col(p).alias("_p")).with_row_index("_i")
             .join(names.select("cand", "nc", "ad", "nk"), on="cand", how="left", maintain_order="left")
             .with_columns([pl.col(c).fill_null("") for c in ("nc", "ad", "nk")]))
    anc = (base.filter(pl.col("_p") >= pmin).sort(["s1", "_p", "cand"], descending=[False, True, False])
               .group_by("s1", maintain_order=True).head(top)
               .with_columns(pl.int_range(pl.len()).over("s1").alias("_r"))
               .select("s1", pl.col("_i").alias("_j"), pl.col("_p").alias("_pd"), "_r"))
    pairs = (base.select("_i", "s1").join(anc, on="s1").filter(pl.col("_i") != pl.col("_j")).drop("s1")
                 .sort(["_i", "_r"]))
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
    _log(cfg, f"siblings: {e.height:,} rows, {pairs.height:,} sibling pairs, {time.time() - t0:.1f}s")
    return e.hstack(out.get_columns())


# ------------------------------------------------------------------------------------------------ inputs
def _paths(src) -> Optional[List[str]]:
    if isinstance(src, (str, os.PathLike)):
        return [str(src)]
    if isinstance(src, (list, tuple)) and src and all(isinstance(s, (str, os.PathLike)) for s in src):
        return [str(s) for s in src]
    return None


def _schema_names(src) -> List[str]:
    ps = _paths(src)
    if ps is not None:
        return pl.scan_parquet(ps[0]).collect_schema().names()
    if isinstance(src, pl.LazyFrame):
        return src.collect_schema().names()
    return list(src.columns)


def _read_rows(src, keys: pl.DataFrame, cols: List[str], enc: bool, key_cols=("s1", "cand"),
               batch: int = 2_000_000) -> pl.DataFrame:
    """Rows of `src` whose key columns are in `keys` (already encoded when enc), only `cols` (+ keys).
    Parquet paths are streamed with pyarrow in batches (memory ~ one batch + the kept rows)."""
    key_cols = list(key_cols)
    want = key_cols + [c for c in cols if c not in key_cols]
    ps = _paths(src)
    parts = []
    if ps is not None:
        import pyarrow.parquet as pq
        for path in ps:
            pf = pq.ParquetFile(path)
            for rb in pf.iter_batches(batch_size=batch, columns=want):
                b = pl.from_arrow(rb)
                if enc:
                    b = b.with_columns([_enc_if(k, b.schema[k]) for k in key_cols])
                parts.append(b.join(keys, on=key_cols, how="semi"))
    else:
        lf = src.lazy() if isinstance(src, pl.DataFrame) else src
        lf = lf.select(want)
        if enc:
            sch = lf.collect_schema()
            lf = lf.with_columns([_enc_if(k, sch[k]) for k in key_cols])
        parts.append(lf.join(keys.lazy(), on=key_cols, how="semi").collect())
    return pl.concat(parts, how="vertical_relaxed") if parts else pl.DataFrame()


def _names(pool_norm, cands: pl.DataFrame, enc: bool, batch: int) -> pl.DataFrame:
    have = _schema_names(pool_norm)
    cols = [c for c in ("name_core", "addr_norm", "name_key") if c in have]
    r = _read_rows(pool_norm, cands.select(pl.col("cand").alias("entity_id")).unique(), cols, enc,
                   key_cols=("entity_id",), batch=batch)
    out = r.select(pl.col("entity_id").alias("cand"),
                   *[pl.col(src).fill_null("").alias(dst) if src in r.columns else pl.lit("").alias(dst)
                     for src, dst in (("name_core", "nc"), ("addr_norm", "ad"), ("name_key", "nk"))])
    return out.unique(subset=["cand"], keep="first", maintain_order=True)


def _stage1_raw(bundle1, frame: pl.DataFrame) -> np.ndarray:
    cols = bundle1["feature_columns"]
    X = frame.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    return np.mean([b.predict(X) for b in bundle1["boosters"]], axis=0)


# ------------------------------------------------------------------------------------------------ frame
def build_frame(scored: pl.DataFrame, feats=None, pool_norm=None, cfg: Optional[Stage2Config] = None, *,
                population=None, stage1_bundle=None) -> pl.DataFrame:
    """Stage-2 frame for the re-scored pairs (see module docstring). Deterministic: rows sorted by
    (s1, p1 desc, cand); competition ties go to the smallest S1 id."""
    cfg = cfg or Stage2Config()
    t0 = time.time()
    cols = feature_columns(cfg)
    need_orig = [c for c in cols if c not in W_FEATS + D_FEATS + C_FEATS + S_FEATS]
    if cfg.guard != "G0":
        need_orig += [c for c in GUARD_COLS if c not in need_orig]
    has_raw, has_label = "p_raw" in scored.columns, "label" in scored.columns
    if not has_raw and stage1_bundle is not None:
        need_orig += [c for c in stage1_bundle["feature_columns"] if c not in need_orig]
    if need_orig and feats is None:
        raise ValueError(f"feats is required for the stage-1 feature columns {need_orig[:5]}...")
    if "sibling" in cfg.groups and pool_norm is None:
        raise ValueError("pool_norm is required for the sibling features (or drop 'sibling' from cfg.groups)")
    feat_names = _schema_names(feats) if feats is not None else []
    miss = [c for c in need_orig if c not in feat_names]
    if miss:
        raise ValueError(f"feats lacks columns {miss[:8]}")
    want_label = (not has_label) and "label" in feat_names

    base_cols = ["s1", "cand", "p"] + (["p_raw"] if has_raw else []) + (["label"] if has_label else [])
    allp = scored.select(base_cols)
    id_dtype = (allp.schema["s1"], allp.schema["cand"])
    enc = _encodable(allp["s1"], allp["cand"])
    if enc:
        allp = allp.with_columns(_enc_if("s1", id_dtype[0]), _enc_if("cand", id_dtype[1]))
    allp = allp.rename({"p": "p1"}).with_columns(pl.col("p1").cast(pl.Float32))
    if has_raw:
        allp = allp.rename({"p_raw": "p1raw"}).with_columns(pl.col("p1raw").cast(pl.Float32))
    if has_label:
        allp = allp.with_columns(pl.col("label").cast(pl.Int8))
    if population is not None:
        pop = _as_ids(population, "s1")
        if enc:
            pop = pop.with_columns(_enc_if("s1", pop.schema["s1"]))
        e = allp.join(pop, on="s1", how="semi", maintain_order="left")
    else:
        e = allp
    _log(cfg, f"frame: {allp.height:,} pairs in the competition frame, {e.height:,} population pairs "
              f"({e['s1'].n_unique():,} S1), int ids {enc}")
    # within + stage-1 decision on the FULL lists of the population
    t = time.time()
    e = within_s1_feats(e, "p1", "p1raw" if has_raw else None)
    e = decision_feats(e, cfg.lam_missing, cfg.empty_bias, "p1")
    fr = e.filter(pl.col("p1") >= cfg.floor)
    del e
    _log(cfg, f"frame: within+decision {time.time() - t:.1f}s -> {fr.height:,} re-scored rows (p1 >= {cfg.floor})")
    # stage-1 feature rows of the re-scored pairs
    t = time.time()
    extra = pl.DataFrame()
    if need_orig or want_label:
        rcols = need_orig + (["label"] if want_label else [])
        extra = _read_rows(feats, fr.select("s1", "cand"), rcols, enc, batch=cfg.feats_batch)
        extra = extra.unique(subset=["s1", "cand"], keep="first", maintain_order=True)
        if extra.height != fr.height:
            raise ValueError(f"feats cover {extra.height:,} of {fr.height:,} re-scored pairs")
        fr = fr.join(extra, on=["s1", "cand"], how="left", maintain_order="left")
        if want_label:
            fr = fr.with_columns(pl.col("label").cast(pl.Int8))
        if not has_raw and stage1_bundle is not None:
            raw = _stage1_raw(stage1_bundle, fr).astype(np.float32)
            lg = pl.Series("_raw", raw).clip(EPS, 1 - EPS)
            fr = fr.with_columns((lg / (1 - lg)).log().cast(pl.Float32).alias("p1_logit"))
        _log(cfg, f"frame: stage-1 feature rows {extra.height:,} in {time.time() - t:.1f}s")
    del extra
    # competition over the WHOLE frame
    if "competition" in cfg.groups:
        t = time.time()
        fr = competition_feats(fr, competition_aggregates(allp.select("s1", "cand", "p1"), fr.select("cand")))
        _log(cfg, f"frame: competition {time.time() - t:.1f}s")
    del allp
    if "sibling" in cfg.groups:
        names = _names(pool_norm, fr.select("cand"), enc, cfg.feats_batch)
        fr = sibling_feats(fr, names, "p1", cfg.sib_top, cfg.sib_pmin, cfg.workers, cfg.sib_chunk, cfg)
    keep = ["s1", "cand", "p1"] + (["label"] if "label" in fr.columns else [])
    for c in cols + GUARD_COLS:
        if c in fr.columns and c not in keep:
            keep.append(c)
    fr = fr.select(keep)
    if enc:
        fr = fr.with_columns([_dec(k) for k, dt in zip(("s1", "cand"), id_dtype) if not dt.is_integer()])
    _log(cfg, f"frame: done {fr.height:,} x {fr.width} in {time.time() - t0:.1f}s")
    return fr


# ------------------------------------------------------------------------------------------------ model
def _isotonic(raw: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    return iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()


def _auc_ll(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import log_loss, roc_auc_score
    if not 0 < y.sum() < len(y):
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)))


def _mm_expr() -> pl.Expr:
    return (pl.col("hn_both") > 0) & (pl.col("hn_exact") == 0) & (pl.col("hn_base") == 0)


def apply_guard(rescored: pl.DataFrame, guard: str = "G1") -> pl.DataFrame:
    """rescored: s1, cand, p1, p2 [+ GUARD_COLS or a boolean 'mm'] -> adds p (guarded stage-2 probability).
    G0 p = p2; G1 house-number-mismatch pairs p = min(p1, p2); G2 mismatch pairs p = p1."""
    if guard == "G0":
        return rescored.with_columns(pl.col("p2").alias("p"))
    mm = pl.col("mm") if "mm" in rescored.columns else _mm_expr()
    if guard == "G1":
        e = pl.when(mm).then(pl.min_horizontal("p1", "p2")).otherwise(pl.col("p2"))
    elif guard == "G2":
        e = pl.when(mm).then(pl.col("p1")).otherwise(pl.col("p2"))
    else:
        raise ValueError(f"guard must be G0, G1 or G2, got {guard!r}")
    return rescored.with_columns(e.fill_null(pl.col("p2")).cast(pl.Float32).alias("p"))


def fit(frame: pl.DataFrame, cfg: Optional[Stage2Config] = None):
    """Frozen-fold OOF + isotonic + final booster. frame from build_frame on the labelled OOF population.
    Returns (bundle, oof) with oof: s1, cand, label, p1, p2_raw, p2, p (guarded, cfg.guard)."""
    cfg = cfg or Stage2Config()
    if "label" not in frame.columns:
        raise ValueError("fit needs a 'label' column (train frame)")
    cols = feature_columns(cfg)
    miss = [c for c in cols if c not in frame.columns]
    if miss:
        raise ValueError(f"frame lacks feature columns {miss[:8]}")
    t0 = time.time()
    fr = frame.sort(["s1", "cand"])                    # canonical order -> deterministic LightGBM
    y = fr["label"].cast(pl.Int8).to_numpy()
    folds = _M.fold_of(fr["s1"], n_folds=cfg.n_folds, seed=cfg.fold_seed)
    mcfg = _MD.ModelConfig(n_folds=cfg.n_folds, fold_seed=cfg.fold_seed, params=dict(cfg.params),
                           max_rounds=cfg.max_rounds, early_stopping=cfg.early_stopping, inner_frac=cfg.inner_frac,
                           feature_columns=list(cols), verbose=False)
    X = fr.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    raw = np.zeros(fr.height, dtype=np.float64)
    iters, gain = [], np.zeros(len(cols))
    seed = int(cfg.params.get("seed", 0))
    for k in range(cfg.n_folds):
        m = folds == k
        b, it = _MD._fit(fr.filter(pl.Series(~m)), list(cols), mcfg, seed=seed + k)
        raw[m] = b.predict(X[m], num_iteration=it)
        iters.append(int(it))
        gain += np.asarray(b.feature_importance("gain", iteration=it), dtype=np.float64)
        _log(cfg, f"fit: fold {k} best_iter {it} ({time.time() - t0:.0f}s)")
    cx, cy = _isotonic(raw, y)
    p2 = np.interp(raw, cx, cy)
    n_final = int(round(cfg.final_rounds_mult * float(np.mean(iters))))
    final, _ = _MD._fit(fr, list(cols), mcfg, seed=seed + 99, rounds=n_final)
    auc2, ll2 = _auc_ll(y, p2)
    auc1, ll1 = _auc_ll(y, fr["p1"].to_numpy())
    report = {"rows": fr.height, "positives": int(y.sum()), "best_iters": iters, "final_rounds": n_final,
              "auc_ll_stage2": [auc2, ll2], "auc_ll_stage1_same_rows": [auc1, ll1],
              "seconds": round(time.time() - t0, 1),
              "top_gain": sorted(((c, round(float(g))) for c, g in zip(cols, gain)), key=lambda kv: -kv[1])[:30]}
    _log(cfg, f"fit: AUC/logloss stage 2 {auc2:.5f}/{ll2:.5f} vs stage 1 {auc1:.5f}/{ll1:.5f}; final {n_final} "
              f"rounds ({report['seconds']}s)")
    bundle = {"booster": final, "feature_columns": list(cols), "calibrator": [cx, cy], "guard": cfg.guard,
              "floor": cfg.floor, "cfg": {k: v for k, v in asdict(cfg).items()}, "best_iters": iters,
              "final_rounds": n_final, "report": report}
    keep = ["s1", "cand", "label", "p1"] + [c for c in GUARD_COLS if c in fr.columns]
    oof = fr.select(keep).with_columns(pl.Series("p2_raw", raw.astype(np.float32)),
                                       pl.Series("p2", p2.astype(np.float32)))
    if cfg.guard != "G0" and not all(c in oof.columns for c in GUARD_COLS):
        raise ValueError(f"guard {cfg.guard} needs {GUARD_COLS} in the frame")
    oof = apply_guard(oof, cfg.guard).select("s1", "cand", "label", "p1", "p2_raw", "p2", "p")
    return bundle, oof


def predict(bundle, frame: pl.DataFrame, scored: Optional[pl.DataFrame] = None,
            guard: Optional[str] = None) -> pl.DataFrame:
    """Stage-2 probabilities for the rows of `frame` (build_frame output), guard applied (default: the bundle's).
    scored=None -> s1, cand, p1, p2, p ; scored given -> merge(scored, ...) (full frame, p replaced)."""
    guard = guard or bundle.get("guard", "G1")
    cols = bundle["feature_columns"]
    X = frame.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
    raw = bundle["booster"].predict(X)
    cx, cy = (np.asarray(v, dtype=np.float64) for v in bundle["calibrator"])
    p2 = np.interp(raw, cx, cy).astype(np.float32)
    keep = ["s1", "cand", "p1"] + ([c for c in GUARD_COLS if c in frame.columns] if guard != "G0" else [])
    res = apply_guard(frame.select(keep).with_columns(pl.Series("p2", p2)), guard).select("s1", "cand", "p1", "p2", "p")
    return res if scored is None else merge(scored, res)


def merge(scored: pl.DataFrame, rescored: pl.DataFrame, p_new: str = "p") -> pl.DataFrame:
    """scored: s1, cand, p (+ any columns); rescored: s1, cand, <p_new>. Rows not re-scored keep their stage-1 p;
    row order of `scored` is kept."""
    r = rescored.select("s1", "cand", pl.col(p_new).alias("_p2"))
    if r.schema["s1"] != scored.schema["s1"] or r.schema["cand"] != scored.schema["cand"]:
        r = r.with_columns(pl.col("s1").cast(scored.schema["s1"]), pl.col("cand").cast(scored.schema["cand"]))
    out = scored.join(r, on=["s1", "cand"], how="left", maintain_order="left")
    return out.with_columns(pl.coalesce(pl.col("_p2"), pl.col("p").cast(pl.Float32)).alias("p")).drop("_p2")


def save_bundle(bundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    bundle["booster"].save_model(os.path.join(out_dir, "model_s2.txt"))
    meta = {k: v for k, v in bundle.items() if k != "booster"}
    with open(os.path.join(out_dir, "stage2.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)


def load_bundle(out_dir: str):
    with open(os.path.join(out_dir, "stage2.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["booster"] = lgb.Booster(model_file=os.path.join(out_dir, "model_s2.txt"))
    return meta


def stage2_config_from_bundle(bundle) -> Stage2Config:
    """The Stage2Config the bundle was trained with (use it for build_frame at test time)."""
    c = dict(bundle.get("cfg", {}))
    known = {f for f in Stage2Config.__dataclass_fields__}
    return Stage2Config(**{k: v for k, v in c.items() if k in known})


# ------------------------------------------------------------------------------------------------ stage-1 helper
def stage1_predict_stream(stage1_bundle, feats, skip_s1=None, batch: int = 2_000_000,
                          keep_label: bool = True) -> pl.DataFrame:
    """Stage-1 final-model p (calibrated) and raw score for every pair of `feats` (parquet path(s) streamed, or a
    frame), skipping the S1 in skip_s1 (e.g. the OOF sample, whose OOF p is used instead).
    -> s1, cand, p, p_raw [, label] (Float32)."""
    cols = stage1_bundle["feature_columns"]
    cal = stage1_bundle.get("calibrator")
    have = _schema_names(feats)
    lab = ["label"] if keep_label and "label" in have else []
    skip = _as_ids(skip_s1, "s1") if skip_s1 is not None else None
    out = []

    def _one(b: pl.DataFrame):
        if skip is not None:
            b = b.join(skip.with_columns(pl.col("s1").cast(b.schema["s1"])), on="s1", how="anti", maintain_order="left")
        if b.height == 0:
            return
        raw = _stage1_raw(stage1_bundle, b)
        p = np.interp(raw, cal[0], cal[1]) if cal else raw
        out.append(b.select("s1", "cand", *lab).with_columns(pl.Series("p", p.astype(np.float32)),
                                                            pl.Series("p_raw", raw.astype(np.float32))))
    ps = _paths(feats)
    if ps is not None:
        import pyarrow.parquet as pq
        for path in ps:
            for rb in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=["s1", "cand"] + lab + cols):
                _one(pl.from_arrow(rb))
    else:
        df = feats.collect() if isinstance(feats, pl.LazyFrame) else feats
        for lo in range(0, df.height, batch):
            _one(df.slice(lo, batch))
    return pl.concat(out) if out else pl.DataFrame(schema={"s1": pl.Utf8, "cand": pl.Utf8, "p": pl.Float32,
                                                           "p_raw": pl.Float32})
