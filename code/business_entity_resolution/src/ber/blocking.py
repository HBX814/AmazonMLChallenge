# -*- coding: utf-8 -*-
"""blocking.py -- candidate generation (multi-pass UNION) + recall measurement for Business Entity Resolution
(Amazon ML Challenge 2026, team Master Bolt). Copy into code/business_entity_resolution/src/ber/blocking.py.

Blocking sets the recall ceiling of the whole solution: a true link that is not a candidate can never be predicted.
One pass is never enough (name-only char-3gram TF-IDF top-40 = 0.71 pair recall on Karnataka), so this module
unions cheap exact-key passes (polars joins with document-frequency caps) with TF-IDF char_wb 3-gram top-K passes
run inside geographic blocks in BOTH directions (S1 -> pool top-K and pool -> S1 top-k; the reverse direction
exploits exclusivity: every S2/S3 record belongs to at most one S1, so its best S1 is usually the right one).

Contract functions
------------------
generate_candidates(s1, pool, cfg=None, stats=None) -> pl.DataFrame
    s1, pool: NORMALIZED frames (er-text-normalization columns: entity_id, country, name_norm, name_core,
    name_compact, addr_norm, house_nums, city_key, state_key; missing optional columns are derived).
    Works per country (hard partition; true links never cross countries).  Returns one row per (s1, cand):
      s1, cand                         Utf8 ids
      p_key_<k> (Boolean)              exact-key pass hits, k in KEY_PASSES
      p_<t>_score (Float32)            best cosine of TF-IDF pass t (either direction), null if not hit
      p_<t>_rank  (Int16)              rank of cand in the S1 -> pool top-K list (1 = best), null if absent
      p_<t>_rrank (Int16)              rank of s1 in the pool -> S1 top-k list of cand, null if absent
      p_key_hit (Int8)                 number of key passes that hit
      n_passes (Int8)                  number of passes (key + TF-IDF, any direction) that hit
      prio (Float32)                   priority used by the per-S1 cap (higher = kept first)
candidate_recall(cands, gt, s1_ids=None) -> dict
    pair recall, per-S1 full-recall rate, candidates/S1 mean/p50/p95/max, oracle macro-F0.5 ceiling
    (via metric.py: predict exactly cands & truth), plus 'by_country' when s1_ids is a frame with `country`.
cap_candidates(cands, cap, reserve=None, n_reserve=0) -> pl.DataFrame
    keep the `cap` highest-priority candidates per S1, plus up to `n_reserve` of the cut rows flagged by `reserve`
reserve_expr(cfg) -> pl.Expr                        Int8 number of cfg.reserve_passes that hit (for cap_candidates)
street_words(min_len=2) -> pl.Expr                  sorted street words of addr_norm (akey / stname passes)
prio_unknown_passes(model_features, cfg)            key passes a (legacy) priority model does not know
geo_shards(s1, pool, cfg) -> iterator               (country, block, s1_part, pool_part) for sharded laptop runs

Same-address passes (key_akey, key_stname) and the reserve
-----------------------------------------------------------
Huge geo blocks (France: 3 regions, 15 cities) break the TF-IDF passes for 'city + generic word' names: the name
top-K ties with hundreds of block members and the address top-K fills up with the ~6 same-address neighbours of
every S1, so exact same-address duplicates were never candidates (diag_fr_e proxy 0.0131/S1 on test France).
  akey    base house number | sorted street words | city_key       (street words = addr_norm tokens without digits,
                                                                    city/state tokens, street types, unit words,
                                                                    directionals and stop words)
  stname  sorted street words | city_key | glued name             (ignores the house number; name = glued sorted
                                                                    name_core tokens and name_compact)
Keys are only built when city_key and the street words are non-empty; a key joins country-wide when <= street_cap
pool rows carry it, else inside the geo block when <= street_block_cap there. Their hits survive the learned
priority cap through `reserve_slots` extra per-S1 slots (only rows the cap would otherwise cut; ordered by #reserve
passes hit, then prio). A prio model trained before these passes existed keeps working: its features are computed
exactly as before (the new flags are left out of p_key_hit / n_passes / prio_hand / n_cand) and rows that only the
new passes found get prio 0, so they enter only through free slots or the reserve.

Usage
-----
    from ber.blocking import BlockingConfig, generate_candidates, candidate_recall
    cfg = BlockingConfig()                          # measured defaults (SKILL.md table)
    stats = {}
    cands = generate_candidates(s1_norm, pool_norm, cfg, stats=stats)
    print(candidate_recall(cands, gt_long, s1_norm.select("entity_id", "country")))
    # candidate_pairs.tsv = exactly these (s1, cand) pairs, one row per test S1 (empty list allowed)

Memory: everything is keyed by Int32 row indices inside a country; string ids are attached only at the end.
TF-IDF uses HashingVectorizer(2**20, alternate_sign=False) + own IDF (sklearn TfidfVectorizer(char 2-4) hit
MemoryError on a 350k-row slice) and sparse_dot_topn (Apache-2.0) with query-side df pruning (9.5x faster).
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp

try:                                    # inside the ber package
    from . import normalize as _N       # type: ignore
except ImportError:                     # reference-code layout: sibling skill folder
    try:
        import normalize as _N          # type: ignore
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "er-text-normalization"))
        import normalize as _N          # type: ignore

KEY_PASSES = ("name", "tok", "hn", "skel", "compact", "addr", "akey", "stname")
TFIDF_FIELDS = ("name", "addr", "both", "skel")
STREET_PASSES = ("akey", "stname")      # same-address / same-street exact-key passes (street_words based)


def _street_stop_words() -> frozenset:
    """Address tokens that are not street-NAME words, for every country (hand-written domain knowledge from
    normalize.py: canonical street types, unit / floor words, designators, directionals, fillers). The country is
    an open set, so the union of all lists is used everywhere."""
    s = set()
    for v in _N.STREET_TYPE_CANON.values():
        s |= set(v)
    s -= {"main", "cross", "nagar", "colony", "sector", "layout"}     # street-NAME words ('Main St', 'Gandhi Nagar')
    s |= set(_N.UNIT_WORDS) | set(_N.DESIGNATORS) | set(_N.FR_ADDR_STOP) | set(_N.COUNTRY_WORDS)
    s |= set(getattr(_N, "_FLOOR_WORDS", ()))
    s |= {"n", "s", "e", "w", "ne", "nw", "se", "sw", "north", "south", "east", "west",
          "the", "and", "of", "at", "in", "on", "to", "near", "opp", "behind", "po", "dist", "tal", "vill", "ps",
          "co", "so", "wo", "floor", "box", "city", "county", "town", "township", "village", "no", "hno", "dno",
          "plot", "flat", "shop", "apt", "appt", "appart", "appartement", "etage", "bat", "pmb", "suite", "ste",
          "cdp", "twp", "cedex", "bp", "cs", "lieu", "dit", "unit", "room", "rm", "bldg", "building"}
    return frozenset(x for x in s if x)


STREET_STOP = _street_stop_words()


@dataclass
class BlockingConfig:
    # ---- exact-key passes (country-wide polars joins) --------------------------------------------------
    key_passes: Tuple[str, ...] = KEY_PASSES
    key_cap: int = 30                 # keep a key only if <= key_cap pool records carry it (country-wide) ...
    key_block_cap: int = 150          # ... else retry it inside the geo block if <= key_block_cap there
    tok_min_len: int = 3              # rare-name-token pass: tokens shorter than this are ignored
    tok_per_record: int = 3           # S1 side: only its N rarest name tokens are joined
    hn_nums: int = 3                  # house-number pass: N rarest numbers x M rarest address words per record
    hn_words: int = 6
    hn_max_digits: int = 6            # numbers longer than this (phones) are ignored
    # same-address passes (key_akey / key_stname, see module docstring)
    street_cap: int = 20              # keep an akey / stname key if <= street_cap pool rows carry it (country-wide) ...
    street_block_cap: int = 20        # ... else retry it inside the geo block (same city name in two US states)
    street_min_len: int = 2           # street words shorter than this are ignored
    akey_max_nums: int = 2            # akey: first N distinct base house numbers of a record (house_nums order)
    # per-S1 cap reserve: up to reserve_slots EXTRA candidates beyond max_cands_per_s1 for rows hit by one of
    # reserve_passes that the (learned) priority would cut; 0 disables
    reserve_passes: Tuple[str, ...] = STREET_PASSES
    reserve_slots: int = 5
    # ---- TF-IDF top-K passes (inside geo blocks) --------------------------------------------------------
    tfidf_passes: Tuple[str, ...] = ("name", "addr", "both")
    k_fwd: Dict[str, int] = field(default_factory=lambda: {"name": 20, "addr": 20, "both": 20, "skel": 10})
    k_rev: Dict[str, int] = field(default_factory=lambda: {"name": 3, "addr": 3, "both": 3, "skel": 2})
    min_score: float = 0.05           # drop TF-IDF hits below this (pruned) cosine
    ngram: int = 3                    # char_wb n-gram size
    n_features: int = 2 ** 20
    max_df: float = 0.01              # query-side pruning of n-grams present in > max_df of the block's docs
    max_df_min_docs: int = 200        # ... but never prune an n-gram present in <= this many docs (small blocks)
    block_col: str = "state_key"      # geo block; "" = unknown -> fallback block queried by every S1
    fallback_block: bool = True
    fallback_k_fwd: int = 5           # S1 -> unknown-geo pool top-K per TF-IDF pass
    # label-free alternate blocks: a pool record whose city is (per the S1 side of the SAME split) dominated by
    # another state also joins that state's TF-IDF block (e.g. "HYDERABAD, Andhra Pradesh" -> telangana, the
    # post-2014 state; region-less "..., DUNKERQUE" -> hauts de france). Measured: 1.2% of India train links sit
    # in a different state block than their S1, 97% of them Hyderabad records labelled Andhra Pradesh.
    city_alt_blocks: bool = False
    city_alt_min_n: int = 20          # S1 with that city_key needed before the city votes
    city_alt_min_share: float = 0.8   # share of those S1 in the dominant state
    # learned candidate priority: LightGBM text model over prio_features() (fit on TRAIN labels by
    # train_prio_model); replaces the hand-made `prio` before the per-S1 cap (hand value kept as prio_hand).
    # Measured on slices (same top-80): recall@30 US_IL 0.9824 -> 0.9900, IN_KA 0.9723 -> 0.9788.
    prio_model: Optional[str] = None
    # ---- union ------------------------------------------------------------------------------------------
    max_cands_per_s1: Optional[int] = 80   # None = no cap (measurement mode)
    n_threads: int = 8
    verbose: bool = True


# =====================================================================================================
# helpers
# =====================================================================================================
def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:            # psutil is optional
        return float("nan")


def _log(cfg, *a):
    if cfg.verbose:
        print(time.strftime("%H:%M:%S"), f"[rss {_rss_gb():.2f}GB]", *a, flush=True)


def _col(df: pl.DataFrame, name: str, default=None) -> pl.Expr:
    if name in df.columns:
        return pl.col(name).cast(pl.Utf8).fill_null("")
    return pl.lit("") if default is None else default


def _prep(df: pl.DataFrame, idx: str) -> pl.DataFrame:
    """Select / derive the columns blocking needs; tolerant to a partially normalized frame."""
    out = df.select(
        pl.int_range(0, pl.len(), dtype=pl.Int32).alias(idx),
        pl.col("entity_id").cast(pl.Utf8).alias("id"),
        _col(df, "name_core").alias("name_core"),
        _col(df, "name_norm", _col(df, "name_core")).alias("name_norm"),
        _col(df, "addr_norm").alias("addr_norm"),
        _col(df, "state_key").alias("state_key"),
        _col(df, "city_key").alias("city_key"),
        (pl.col("house_nums").cast(pl.List(pl.Utf8)) if "house_nums" in df.columns
         else pl.lit([], dtype=pl.List(pl.Utf8))).alias("house_nums"),
        (_col(df, "name_compact") if "name_compact" in df.columns else pl.lit(None, dtype=pl.Utf8)).alias("name_compact"),
    )
    return out.with_columns(
        pl.when(pl.col("name_compact").is_null() | (pl.col("name_compact") == ""))
          .then(pl.col("name_core").str.replace_all(" ", "", literal=True))
          .otherwise(pl.col("name_compact")).alias("name_compact"),
        pl.col("house_nums").fill_null([]),
    )


# =====================================================================================================
# exact-key passes
# =====================================================================================================
def _hash(e: pl.Expr) -> pl.Expr:
    return e.hash(seed=7)


def _keys_name(d: pl.DataFrame, idx: str) -> pl.DataFrame:
    """sorted unique core tokens (token-shuffle invariant)."""
    k = pl.col("name_core").str.split(" ").list.unique().list.sort().list.join(" ")
    return d.select(idx, k.alias("k")).filter(pl.col("k").str.len_chars() >= 2).select(idx, _hash(pl.col("k")).alias("key"))


def _keys_compact(d: pl.DataFrame, idx: str) -> pl.DataFrame:
    """glued core and glued full name (domain-form names 'unifiedpinnacleesports(llc)')."""
    a = d.select(idx, pl.col("name_compact").alias("k"))
    b = d.select(idx, pl.col("name_norm").str.replace_all(" ", "", literal=True).alias("k"))
    return (pl.concat([a, b]).filter(pl.col("k").str.len_chars() >= 5)
              .select(idx, _hash(pl.col("k")).alias("key")).unique())


def _keys_addr(d: pl.DataFrame, idx: str) -> pl.DataFrame:
    """sorted unique address tokens (component-order invariant); needs >= 3 tokens."""
    t = pl.col("addr_norm").str.split(" ").list.unique()
    return (d.select(idx, t.alias("t")).filter(pl.col("t").list.len() >= 3)
              .select(idx, _hash(pl.col("t").list.sort().list.join(" ")).alias("key")))


def street_words(min_len: int = 2) -> pl.Expr:
    """Street-name words of a normalized address: addr_norm tokens (>= min_len chars) without digits, without the
    city_key / state_key tokens and without STREET_STOP words, sorted, unique, ' '-joined ('' when none).
    Needs the columns addr_norm, city_key, state_key."""
    stop = sorted(STREET_STOP)
    w = pl.col("addr_norm").str.split(" ").list.eval(
        pl.element().filter((pl.element().str.len_chars() >= min_len) & ~pl.element().str.contains(r"\d")
                            & ~pl.element().is_in(stop)))
    return (w.list.set_difference(pl.col("city_key").str.split(" "))
             .list.set_difference(pl.col("state_key").str.split(" ")).list.sort().list.join(" "))


def _keys_akey(d: pl.DataFrame, idx: str, cfg: BlockingConfig) -> pl.DataFrame:
    """address key: base house number | street words | city_key, one key per distinct base number among the
    record's first akey_max_nums house numbers (house-number order of normalize_frame)."""
    base = pl.col("house_nums").list.eval(pl.element().str.extract(r"(\d+)", 1).str.strip_chars_start("0"))
    k = (d.select(idx, pl.col("city_key").alias("c"), street_words(cfg.street_min_len).alias("sw"), base.alias("n"))
          .filter((pl.col("c") != "") & (pl.col("sw") != ""))
          .with_columns(pl.col("n").list.drop_nulls()
                        .list.eval(pl.element().filter((pl.element().str.len_chars() >= 1)
                                                       & (pl.element().str.len_chars() <= cfg.hn_max_digits)))
                        .list.unique(maintain_order=True).list.head(cfg.akey_max_nums))
          .explode("n", empty_as_null=False).drop_nulls("n"))
    return k.select(idx, _hash(pl.col("n") + "|" + pl.col("sw") + "|" + pl.col("c")).alias("key")).unique()


def _keys_stname(d: pl.DataFrame, idx: str, cfg: BlockingConfig) -> pl.DataFrame:
    """street key without house number: street words | city_key | glued name, with two name variants (glued sorted
    unique name_core tokens = shuffle invariant; name_compact = handle / domain form '@nantessportive')."""
    srt = pl.col("name_core").str.split(" ").list.unique().list.sort().list.join("")
    k = (d.select(idx, pl.col("city_key").alias("c"), street_words(cfg.street_min_len).alias("sw"),
                  pl.concat_list(srt, pl.col("name_compact")).alias("nm"))
          .filter((pl.col("c") != "") & (pl.col("sw") != ""))
          .explode("nm", empty_as_null=False).filter(pl.col("nm").str.len_chars() >= 3))
    return k.select(idx, _hash(pl.col("sw") + "|" + pl.col("c") + "|" + pl.col("nm")).alias("key")).unique()


def _skeleton_map(tokens: pl.Series) -> pl.DataFrame:
    u = tokens.unique().drop_nulls()
    return pl.DataFrame({"t": u, "sk": [_N.skeleton_key(x) for x in u.to_list()]}, schema={"t": pl.Utf8, "sk": pl.Utf8})


def _keys_skel(d: pl.DataFrame, idx: str, skmap: pl.DataFrame) -> pl.DataFrame:
    """sorted unique consonant skeletons of the core tokens (transliteration / vowel-typo invariant)."""
    t = d.select(idx, pl.col("name_core").str.split(" ").alias("t")).explode("t", empty_as_null=False).filter(pl.col("t") != "")
    t = t.join(skmap, on="t", how="left").filter(pl.col("sk").str.len_chars() >= 1)
    k = t.group_by(idx).agg(pl.col("sk").unique().sort().str.join(" ").alias("k"))
    return k.filter(pl.col("k").str.len_chars() >= 3).select(idx, _hash(pl.col("k")).alias("key"))


def _tokens_long(d: pl.DataFrame, idx: str, min_len: int) -> pl.DataFrame:
    t = d.select(idx, pl.col("name_core").str.split(" ").list.unique().alias("t")).explode("t", empty_as_null=False)
    return t.filter(pl.col("t").str.len_chars() >= min_len).select(idx, pl.col("t"))


def _addr_nums_words(d: pl.DataFrame, idx: str, cfg: BlockingConfig):
    nums = d.select(idx, pl.concat_list(
        pl.col("addr_norm").str.extract_all(r"\d+"),
        pl.col("house_nums").list.eval(pl.element().str.extract(r"(\d+)", 1))).alias("n")).explode("n", empty_as_null=False)
    nums = (nums.drop_nulls("n").with_columns(pl.col("n").str.strip_chars_start("0"))
                .filter((pl.col("n").str.len_chars() >= 1) & (pl.col("n").str.len_chars() <= cfg.hn_max_digits))
                .unique([idx, "n"]))
    words = (d.select(idx, pl.col("addr_norm").str.split(" ").alias("w")).explode("w", empty_as_null=False)
              .filter(pl.col("w").str.contains(r"^[a-z]{3,}$")).unique([idx, "w"]))
    return nums, words


def _rarest(long: pl.DataFrame, idx: str, col: str, dfs: pl.DataFrame, n: int) -> pl.DataFrame:
    """keep, per record, the n values with the smallest pool document frequency (values absent from pool dropped)."""
    return (long.join(dfs, on=col, how="inner").sort([idx, "df", col])
                .group_by(idx, maintain_order=True).head(n).drop("df"))


def _doc_freq(long: pl.DataFrame, col: str) -> pl.DataFrame:
    return long.group_by(col).agg(pl.len().cast(pl.Int32).alias("df"))


def _key_join(kq: pl.DataFrame, kp: pl.DataFrame, bq: pl.DataFrame, bp: pl.DataFrame, cap: int, block_cap: int) -> pl.DataFrame:
    """(qi,key) x (pi,key) -> (qi,pi). Keys carried by <= cap pool rows join country-wide; keys with more pool rows
    join only inside the same geo block when <= block_cap there (generic names like 'primary care group')."""
    kp = kp.unique()
    df = kp.group_by("key").agg(pl.len().alias("df"))
    small = df.filter(pl.col("df") <= cap).select("key")
    out = [kq.join(small, on="key", how="semi").join(kp.join(small, on="key", how="semi"), on="key").select("qi", "pi")]
    if block_cap > 0:
        big = df.filter(pl.col("df") > cap).select("key")
        kq2 = kq.join(big, on="key", how="semi").join(bq, on="qi")
        kp2 = kp.join(big, on="key", how="semi").join(bp, on="pi").filter(pl.col("blk") != "")
        ok = kp2.group_by(["key", "blk"]).agg(pl.len().alias("df")).filter(pl.col("df") <= block_cap).select("key", "blk")
        out.append(kq2.join(ok, on=["key", "blk"], how="semi").join(kp2, on=["key", "blk"]).select("qi", "pi"))
    return pl.concat(out).unique()


def _run_key_passes(q: pl.DataFrame, p: pl.DataFrame, cfg: BlockingConfig, stats: dict) -> List[pl.DataFrame]:
    bq = q.select("qi", pl.col(cfg.block_col).alias("blk")) if cfg.block_col in q.columns else q.select("qi", pl.lit("").alias("blk"))
    bp = p.select("pi", pl.col(cfg.block_col).alias("blk")) if cfg.block_col in p.columns else p.select("pi", pl.lit("").alias("blk"))
    out = []
    skmap = None
    for kp_name in cfg.key_passes:
        t0 = time.time()
        if kp_name == "name":
            pairs = _key_join(_keys_name(q, "qi"), _keys_name(p, "pi"), bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "compact":
            pairs = _key_join(_keys_compact(q, "qi"), _keys_compact(p, "pi"), bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "addr":
            pairs = _key_join(_keys_addr(q, "qi"), _keys_addr(p, "pi"), bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "skel":
            if skmap is None:
                skmap = _skeleton_map(pl.concat([q["name_core"], p["name_core"]]).str.split(" ").explode(empty_as_null=False))
            pairs = _key_join(_keys_skel(q, "qi", skmap), _keys_skel(p, "pi", skmap), bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "tok":
            tq, tp = _tokens_long(q, "qi", cfg.tok_min_len), _tokens_long(p, "pi", cfg.tok_min_len)
            tq = _rarest(tq, "qi", "t", _doc_freq(tp, "t"), cfg.tok_per_record)
            pairs = _key_join(tq.select("qi", _hash(pl.col("t")).alias("key")), tp.select("pi", _hash(pl.col("t")).alias("key")),
                              bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "hn":
            nq, wq = _addr_nums_words(q, "qi", cfg)
            np_, wp = _addr_nums_words(p, "pi", cfg)
            ndf, wdf = _doc_freq(np_, "n"), _doc_freq(wp, "w")
            parts = []
            for (nn, ww, idx) in ((nq, wq, "qi"), (np_, wp, "pi")):
                nn = _rarest(nn, idx, "n", ndf, cfg.hn_nums)
                ww = _rarest(ww, idx, "w", wdf, cfg.hn_words)
                kk = nn.join(ww, on=idx).select(idx, _hash(pl.col("n") + "|" + pl.col("w")).alias("key"))
                parts.append(kk)
            pairs = _key_join(parts[0], parts[1], bq, bp, cfg.key_cap, cfg.key_block_cap)
        elif kp_name == "akey":
            pairs = _key_join(_keys_akey(q, "qi", cfg), _keys_akey(p, "pi", cfg), bq, bp, cfg.street_cap, cfg.street_block_cap)
        elif kp_name == "stname":
            pairs = _key_join(_keys_stname(q, "qi", cfg), _keys_stname(p, "pi", cfg), bq, bp, cfg.street_cap,
                              cfg.street_block_cap)
        else:
            raise ValueError(f"unknown key pass {kp_name}")
        pairs = pairs.with_columns(pl.lit(f"key_{kp_name}").alias("pass"))
        stats.setdefault("passes", {})[f"key_{kp_name}"] = {"seconds": round(time.time() - t0, 2), "pairs": pairs.height}
        _log(cfg, f"key pass {kp_name}: {pairs.height:,} pairs {time.time() - t0:.1f}s")
        out.append(pairs)
        gc.collect()
    return out


# =====================================================================================================
# TF-IDF passes (HashingVectorizer + own IDF + sparse_dot_topn), inside geo blocks, both directions
# =====================================================================================================
_HV = {}


def _hv(cfg: BlockingConfig):
    key = (cfg.ngram, cfg.n_features)
    if key not in _HV:
        from sklearn.feature_extraction.text import HashingVectorizer
        _HV[key] = HashingVectorizer(analyzer="char_wb", ngram_range=(cfg.ngram, cfg.ngram), n_features=cfg.n_features,
                                     alternate_sign=False, norm=None, lowercase=False, dtype=np.float32)
    return _HV[key]


def _counts(texts: List[str], cfg) -> sp.csr_matrix:
    X = _hv(cfg).transform(texts).tocsr()
    X.sum_duplicates()
    return X


def _tfidf(Xc: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    X = Xc.copy()
    X.data = np.log1p(X.data).astype(np.float32)                 # sublinear tf
    X.data *= idf[X.indices]
    sq = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()).astype(np.float32)
    sq[sq == 0] = 1.0
    X = sp.diags(1.0 / sq).dot(X).tocsr().astype(np.float32)   # L2 rows
    return X


def _prune(X: sp.csr_matrix, frequent: np.ndarray) -> sp.csr_matrix:
    """zero the columns of very frequent n-grams in the QUERY matrix only: scores become partial cosines over
    discriminative n-grams and sparse_dot_topn gets ~10x faster (measured)."""
    Y = X.copy()
    Y.data[frequent[Y.indices]] = 0.0
    Y.eliminate_zeros()
    return Y


def _topn(A: sp.csr_matrix, BT: sp.csr_matrix, k: int, cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """row-wise top-k of A @ BT -> (row, col, score, rank) arrays."""
    from sparse_dot_topn import sp_matmul_topn
    if A.shape[0] == 0 or BT.shape[1] == 0 or k <= 0:
        e = np.zeros(0, np.int32)
        return e, e, np.zeros(0, np.float32), e.astype(np.int16)
    R = sp_matmul_topn(A, BT, top_n=min(k, BT.shape[1]), threshold=cfg.min_score, sort=True, n_threads=cfg.n_threads)
    R = R.tocsr()
    cnt = np.diff(R.indptr)
    rows = np.repeat(np.arange(R.shape[0], dtype=np.int32), cnt)
    rank = (np.arange(R.nnz) - np.repeat(R.indptr[:-1], cnt) + 1).astype(np.int16)
    return rows, R.indices.astype(np.int32), R.data.astype(np.float32), rank


def _field_texts(d: pl.DataFrame, fld: str, skmap: Optional[pl.DataFrame]) -> List[str]:
    if fld == "name":
        return d["name_core"].to_list()
    if fld == "addr":
        return d["addr_norm"].to_list()
    if fld == "skel":
        t = d.select("row", pl.col("name_core").str.split(" ").alias("t")).explode("t", empty_as_null=False).join(skmap, on="t", how="left")
        s = t.group_by("row").agg(pl.col("sk").drop_nulls().str.join(" ").alias("s"))
        return d.select("row").join(s, on="row", how="left").sort("row")["s"].fill_null("").to_list()   # 1 text per row
    raise ValueError(fld)


def _block_matrices(qb: pl.DataFrame, pb: pl.DataFrame, fld: str, cfg, skmap):
    """TF-IDF matrices of one block for one field ('both' = hstack of sqrt(.5)-scaled name and addr matrices,
    so its cosine = mean of the name and address cosines)."""
    if fld == "both":
        Qn, Pn, fn = _block_matrices(qb, pb, "name", cfg, skmap)
        Qa, Pa, fa = _block_matrices(qb, pb, "addr", cfg, skmap)
        w = np.float32(math.sqrt(0.5))
        return (sp.hstack([Qn * w, Qa * w], format="csr"), sp.hstack([Pn * w, Pa * w], format="csr"),
                np.concatenate([fn, fa]))
    Cq = _counts(_field_texts(qb, fld, skmap), cfg)
    Cp = _counts(_field_texts(pb, fld, skmap), cfg)
    n = Cq.shape[0] + Cp.shape[0]
    df = np.bincount(Cq.indices, minlength=cfg.n_features) + np.bincount(Cp.indices, minlength=cfg.n_features)
    idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)
    frequent = df > max(cfg.max_df * n, cfg.max_df_min_docs)
    return _tfidf(Cq, idf), _tfidf(Cp, idf), frequent


def _run_block(qb: pl.DataFrame, pb: pl.DataFrame, cfg, skmap, k_fwd: Dict[str, int], k_rev: Dict[str, int],
               tag: str, stats: dict) -> List[pl.DataFrame]:
    out = []
    qb = qb.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("row"))
    pb = pb.with_columns(pl.int_range(0, pl.len(), dtype=pl.Int32).alias("row"))
    qi = qb["qi"].to_numpy()
    pi = pb["pi"].to_numpy()
    for fld in cfg.tfidf_passes:
        t0 = time.time()
        Q, P, frequent = _block_matrices(qb, pb, fld, cfg, skmap)
        for direction, k in (("fwd", k_fwd.get(fld, 0)), ("rev", k_rev.get(fld, 0))):
            if k <= 0:
                continue
            if direction == "fwd":
                r, c, s, rk = _topn(_prune(Q, frequent), P.T.tocsr(), k, cfg)
                a, b = qi[r], pi[c]
            else:
                r, c, s, rk = _topn(_prune(P, frequent), Q.T.tocsr(), k, cfg)
                a, b = qi[c], pi[r]
            out.append(pl.DataFrame({"qi": a, "pi": b, "score": s, "rank": rk},
                                    schema={"qi": pl.Int32, "pi": pl.Int32, "score": pl.Float32, "rank": pl.Int16})
                       .with_columns(pl.lit(f"{fld}_{direction}").alias("pass")))
        st = stats.setdefault("passes", {}).setdefault(f"tfidf_{fld}", {"seconds": 0.0, "pairs": 0})
        st["seconds"] = round(st["seconds"] + time.time() - t0, 2)
        st["pairs"] += sum(x.height for x in out[-2:])
        del Q, P
        gc.collect()
    _log(cfg, f"  block {tag}: S1={qb.height:,} pool={pb.height:,}")
    return out


def city_alt_blocks(q: pl.DataFrame, p: pl.DataFrame, cfg: BlockingConfig) -> pl.Series:
    """Per pool row: the alternate geo block implied by its city ("" = none). City -> dominant block is voted by
    the S1 rows (same split, no labels): >= city_alt_min_n S1 with that city_key, >= city_alt_min_share of them in
    one block. Only rows whose own block differs (incl. unknown) get an alternate."""
    bc = cfg.block_col
    if not cfg.city_alt_blocks or "city_key" not in q.columns or "city_key" not in p.columns or bc not in q.columns:
        return pl.Series("_alt", [""] * p.height, dtype=pl.Utf8)
    votes = (q.filter((pl.col("city_key") != "") & (pl.col(bc) != "")).group_by("city_key", bc).agg(pl.len().alias("n"))
              .with_columns(pl.col("n").sum().over("city_key").alias("tot"))
              .filter((pl.col("tot") >= cfg.city_alt_min_n) & (pl.col("n") >= cfg.city_alt_min_share * pl.col("tot")))
              .select("city_key", pl.col(bc).alias("_alt")))
    pb = p[bc] if bc in p.columns else pl.Series([""] * p.height)
    alt = (p.select(pl.col("city_key").cast(pl.Utf8).fill_null(""), pb.alias("_own"))
            .join(votes, on="city_key", how="left", maintain_order="left")
            .select(pl.when(pl.col("_alt").is_not_null() & (pl.col("_alt") != pl.col("_own")))
                      .then(pl.col("_alt")).otherwise(pl.lit("")).alias("_alt")))
    return alt["_alt"]


def _run_tfidf_passes(q: pl.DataFrame, p: pl.DataFrame, cfg: BlockingConfig, stats: dict) -> List[pl.DataFrame]:
    if not cfg.tfidf_passes:
        return []
    skmap = None
    if "skel" in cfg.tfidf_passes:
        skmap = _skeleton_map(pl.concat([q["name_core"], p["name_core"]]).str.split(" ").explode(empty_as_null=False))
    bc = cfg.block_col
    qblk = q[bc] if bc in q.columns else pl.Series([""] * q.height)
    pblk = p[bc] if bc in p.columns else pl.Series([""] * p.height)
    q = q.with_columns(qblk.alias("_blk"))
    p = p.with_columns(pblk.alias("_blk"), city_alt_blocks(q, p, cfg).alias("_alt"))
    n_alt = int((p["_alt"] != "").sum())
    stats["pool_city_alt_rows"] = n_alt
    if n_alt:
        _log(cfg, f"city-alternate blocks: {n_alt:,} pool rows also join the block their city votes for")
    out = []
    t0 = time.time()
    blocks = sorted(set(q["_blk"].unique().to_list()) - {""})
    for b in blocks:
        qb = q.filter(pl.col("_blk") == b)
        pb = p.filter((pl.col("_blk") == b) | (pl.col("_alt") == b))
        if pb.height:
            out += _run_block(qb, pb, cfg, skmap, cfg.k_fwd, cfg.k_rev, b, stats)
    # S1 whose geo key is unknown: query the whole country pool (rare: S1 addresses always end with the state)
    q_unk = q.filter(pl.col("_blk") == "")
    if q_unk.height:
        out += _run_block(q_unk, p, cfg, skmap, cfg.k_fwd, cfg.k_rev, "<S1 unknown geo>", stats)
    # pool whose geo key is unknown (empty / unparseable address): fallback block queried by every S1
    p_unk = p.filter(pl.col("_blk") == "")
    if cfg.fallback_block and p_unk.height and q.height - q_unk.height > 0:
        kf = {f: min(cfg.fallback_k_fwd, cfg.k_fwd.get(f, 0)) for f in cfg.tfidf_passes}
        out += _run_block(q.filter(pl.col("_blk") != ""), p_unk, cfg, skmap, kf, cfg.k_rev, "<pool unknown geo>", stats)
    stats["tfidf_seconds"] = round(time.time() - t0, 2)
    stats["n_blocks"] = len(blocks)
    stats["pool_unknown_geo"] = p_unk.height
    stats["s1_unknown_geo"] = q_unk.height
    return out


# =====================================================================================================
# union + priority + cap
# =====================================================================================================
def _union(key_pairs: List[pl.DataFrame], tf_pairs: List[pl.DataFrame], cfg: BlockingConfig) -> pl.DataFrame:
    aggs = []
    if key_pairs:
        kp = pl.concat(key_pairs).with_columns(pl.lit(None, dtype=pl.Float32).alias("score"), pl.lit(None, dtype=pl.Int16).alias("rank"))
    else:
        kp = pl.DataFrame(schema={"qi": pl.Int32, "pi": pl.Int32, "pass": pl.Utf8, "score": pl.Float32, "rank": pl.Int16})
    allp = pl.concat([kp.select("qi", "pi", "score", "rank", "pass")] + [t.select("qi", "pi", "score", "rank", "pass") for t in tf_pairs])
    for k in cfg.key_passes:
        aggs.append((pl.col("pass") == f"key_{k}").any().alias(f"p_key_{k}"))
    for f in cfg.tfidf_passes:
        is_f = pl.col("pass").is_in([f"{f}_fwd", f"{f}_rev"])
        aggs.append(pl.col("score").filter(is_f).max().alias(f"p_{f}_score"))
        aggs.append(pl.col("rank").filter(pl.col("pass") == f"{f}_fwd").min().alias(f"p_{f}_rank"))
        aggs.append(pl.col("rank").filter(pl.col("pass") == f"{f}_rev").min().alias(f"p_{f}_rrank"))
    aggs.append(pl.col("pass").n_unique().cast(pl.Int8).alias("n_passes"))
    # group_by output order is not deterministic (multithreaded hash tables): sort to a canonical order so the
    # learned priority, the cap's tie-breaks and everything downstream are identical run to run
    u = allp.group_by(["qi", "pi"]).agg(aggs).sort(["qi", "pi"])
    u = u.with_columns(_key_hit_expr(cfg.key_passes).alias("p_key_hit"))
    return u.with_columns(_hand_prio_expr(pl.col("p_key_hit"), pl.col("n_passes"), cfg).alias("prio"))


def _key_hit_expr(passes) -> pl.Expr:
    """Int8 number of the given key passes that hit."""
    if not passes:
        return pl.lit(0, dtype=pl.Int8)
    return pl.sum_horizontal([pl.col(f"p_key_{k}").fill_null(False).cast(pl.Int8) for k in passes]).cast(pl.Int8)


def _hand_prio_expr(key_hit: pl.Expr, n_passes: pl.Expr, cfg: BlockingConfig) -> pl.Expr:
    """hand-made priority: key agreement first, then best cosine, bonus when the pair is mutual/top-1 somewhere."""
    best = pl.max_horizontal([pl.col(f"p_{f}_score").fill_null(0.0) for f in cfg.tfidf_passes]) if cfg.tfidf_passes else pl.lit(0.0)
    top1 = pl.any_horizontal([(pl.col(f"p_{f}_rrank") == 1) | (pl.col(f"p_{f}_rank") == 1) for f in cfg.tfidf_passes]) \
        if cfg.tfidf_passes else pl.lit(False)
    return (0.35 * key_hit.cast(pl.Float32) + best + 0.2 * top1.fill_null(False).cast(pl.Float32)
            + 0.02 * n_passes.cast(pl.Float32)).cast(pl.Float32)


def reserve_expr(cfg: BlockingConfig, columns=None) -> pl.Expr:
    """Int8 number of cfg.reserve_passes (restricted to cfg.key_passes and, if given, to existing `columns`) that
    hit: the `reserve` argument of cap_candidates."""
    ps = [k for k in cfg.reserve_passes if k in cfg.key_passes and (columns is None or f"p_key_{k}" in columns)]
    return _key_hit_expr(ps)


# =====================================================================================================
# learned candidate priority
# =====================================================================================================
_PRIO_BOOSTERS: Dict[str, object] = {}


def prio_feature_names(cfg: BlockingConfig) -> List[str]:
    names = [f"p_key_{k}" for k in cfg.key_passes]
    for f in cfg.tfidf_passes:
        names += [f"p_{f}_score", f"p_{f}_rank", f"p_{f}_rrank"]
    return names + ["n_passes", "p_key_hit", "prio_hand", "best_rel", "prio_rank", "n_cand"]


def prio_unknown_passes(model_features: List[str], cfg: BlockingConfig) -> Optional[Tuple[str, ...]]:
    """Key passes of cfg that a priority model does not know: () when its feature list equals
    prio_feature_names(cfg); the missing passes when it equals the list of cfg without them (a model trained before
    those passes were added, e.g. v2/v3 prio_model.txt without akey/stname); None when incompatible."""
    model_features = list(model_features)
    if model_features == prio_feature_names(cfg):
        return ()
    missing = tuple(k for k in cfg.key_passes if f"p_key_{k}" not in model_features)
    if missing:
        import dataclasses
        sub = dataclasses.replace(cfg, key_passes=tuple(k for k in cfg.key_passes if k not in missing))
        if model_features == prio_feature_names(sub):
            return missing
    return None


def prio_features(u: pl.DataFrame, cfg: BlockingConfig, unknown_passes: Tuple[str, ...] = ()) -> pl.DataFrame:
    """Float32 features of union rows for the priority model: pass flags / scores / ranks (missing rank = 99),
    hand-made prio, and within-S1 context (best cosine relative to the S1's best, hand-prio rank, #candidates).
    unknown_passes (see prio_unknown_passes): key passes the model predates; they are left out of every feature
    (flags, p_key_hit, n_passes, prio_hand, prio_rank, n_cand), so rows another pass also found get exactly the
    features they had before those passes existed (rows found ONLY by them sort last and are not counted)."""
    grp = "qi" if "qi" in u.columns else "s1"
    known = [k for k in cfg.key_passes if k not in unknown_passes]
    unk = [k for k in cfg.key_passes if k in unknown_passes]
    cols = [pl.col(f"p_key_{k}").cast(pl.Float32).fill_null(0.0).alias(f"p_key_{k}") for k in known]
    for f in cfg.tfidf_passes:
        cols += [pl.col(f"p_{f}_score").cast(pl.Float32).fill_null(0.0).alias(f"p_{f}_score"),
                 pl.col(f"p_{f}_rank").cast(pl.Float32).fill_null(99.0).alias(f"p_{f}_rank"),
                 pl.col(f"p_{f}_rrank").cast(pl.Float32).fill_null(99.0).alias(f"p_{f}_rrank")]
    best = pl.max_horizontal([pl.col(f"p_{f}_score").fill_null(0.0) for f in cfg.tfidf_passes]) \
        if cfg.tfidf_passes else pl.lit(0.0)
    if unk:
        n_passes = (pl.col("n_passes").cast(pl.Int8) - _key_hit_expr(unk)).cast(pl.Int8)
        key_hit = _key_hit_expr(known)
        hand = _hand_prio_expr(key_hit, n_passes, cfg)
        n_cand = (n_passes > 0).sum().over(grp)
    else:
        n_passes, key_hit = pl.col("n_passes"), pl.col("p_key_hit")
        hand = pl.col("prio_hand" if "prio_hand" in u.columns else "prio")
        n_cand = pl.len().over(grp)
    cols += [n_passes.cast(pl.Float32).alias("n_passes"), key_hit.cast(pl.Float32).alias("p_key_hit"),
             hand.cast(pl.Float32).alias("prio_hand"),
             (best / best.max().over(grp).clip(1e-6)).cast(pl.Float32).alias("best_rel"),
             hand.rank("ordinal", descending=True).over(grp).cast(pl.Float32).alias("prio_rank"),
             n_cand.cast(pl.Float32).alias("n_cand")]
    return u.select(cols)


def _prio_booster(path: str):
    if path not in _PRIO_BOOSTERS:
        import lightgbm as lgb
        _PRIO_BOOSTERS[path] = lgb.Booster(model_file=path)
    return _PRIO_BOOSTERS[path]


def apply_prio_model(u: pl.DataFrame, cfg: BlockingConfig, chunk: int = 5_000_000) -> pl.DataFrame:
    """prio := learned P(true link) from the blocking columns (prio_hand keeps the hand-made value).
    Backward compatible: a model that predates some key passes (prio_unknown_passes) gets the features it was
    trained on, and rows that only those passes found get prio 0 (they fill free slots / the cap reserve)."""
    if not cfg.prio_model:
        return u
    b = _prio_booster(cfg.prio_model)
    names = list(b.feature_name())
    unknown = prio_unknown_passes(names, cfg)
    if unknown is None:
        raise ValueError(f"prio model features {names} != {prio_feature_names(cfg)} (blocking config changed?)")
    if unknown:
        _log(cfg, f"prio model predates key passes {list(unknown)}: legacy features; rows found only by them get prio 0")
    X = prio_features(u, cfg, unknown).select(names)
    pred = np.empty(u.height, dtype=np.float32)
    for i in range(0, u.height, chunk):
        pred[i:i + chunk] = b.predict(X.slice(i, chunk).to_numpy(), num_threads=cfg.n_threads)
    del X
    out = u.rename({"prio": "prio_hand"}).with_columns(pl.Series("prio", pred))
    if unknown:
        legacy_hits = pl.col("n_passes").cast(pl.Int8) - _key_hit_expr(unknown)
        out = out.with_columns(pl.when(legacy_hits > 0).then(pl.col("prio")).otherwise(pl.lit(0.0))
                               .cast(pl.Float32).alias("prio"))
    return out


def train_prio_model(u: pl.DataFrame, cfg: BlockingConfig, out_path: str, rounds: int = 300) -> dict:
    """Fit the priority model on UNCAPPED union rows with `label` (TRAIN S1 only) and save it (LightGBM text)."""
    import lightgbm as lgb
    t0 = time.time()
    names = prio_feature_names(cfg)
    if "s1" in u.columns and "cand" in u.columns:
        u = u.sort(["s1", "cand"])                     # canonical row order -> reproducible fit
    X = prio_features(u, cfg).select(names).to_numpy()
    y = u["label"].cast(pl.Int8).to_numpy()
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "verbose": -1, "num_threads": cfg.n_threads, "seed": 0, "deterministic": True}
    b = lgb.train(params, lgb.Dataset(X, y, feature_name=names, free_raw_data=True), num_boost_round=rounds)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    b.save_model(out_path)
    _PRIO_BOOSTERS.pop(out_path, None)
    gain = sorted(zip(names, b.feature_importance("gain").tolist()), key=lambda kv: -kv[1])
    rep = {"rows": int(len(y)), "positives": int(y.sum()), "rounds": rounds, "seconds": round(time.time() - t0, 1),
           "top_gain": gain[:10]}
    _log(cfg, f"prio model: {rep['rows']:,} rows ({rep['positives']:,} pos) in {rep['seconds']}s -> {out_path}")
    return rep


def cap_candidates(cands: pl.DataFrame, cap: Optional[int], reserve=None, n_reserve: int = 0) -> pl.DataFrame:
    """keep the `cap` highest-`prio` candidates per S1 (deterministic tie-break on cand id).
    reserve (column name or expression, integer / boolean: e.g. reserve_expr(cfg)) + n_reserve > 0: additionally
    keep, per S1, up to n_reserve of the rows the cap cut whose reserve value is > 0, ordered by reserve value,
    prio, prio_hand (when present) descending, then cand id. Output has at most cap + n_reserve rows per S1 and the
    input columns only."""
    if cap is None:
        return cands
    s1c = "s1" if "s1" in cands.columns else "qi"
    cc = "cand" if "cand" in cands.columns else "pi"
    order = cands.sort([s1c, "prio", cc], descending=[False, True, False])
    if reserve is None or n_reserve <= 0:
        return order.group_by(s1c, maintain_order=True).head(cap)
    rsv = pl.col(reserve) if isinstance(reserve, str) else reserve
    order = order.with_columns(rsv.cast(pl.Int16).alias("_rsv"), pl.int_range(pl.len(), dtype=pl.Int32).over(s1c).alias("_pos"))
    rest = order.filter((pl.col("_pos") >= cap) & (pl.col("_rsv") > 0))
    if rest.height == 0:
        return order.filter(pl.col("_pos") < cap).drop("_rsv", "_pos")
    by = [s1c, "_rsv", "prio"] + (["prio_hand"] if "prio_hand" in rest.columns else []) + [cc]
    desc = [False, True, True] + ([True] if "prio_hand" in rest.columns else []) + [False]
    extra = rest.sort(by, descending=desc).group_by(s1c, maintain_order=True).head(n_reserve)
    return (pl.concat([order.filter(pl.col("_pos") < cap), extra])
              .sort([s1c, "prio", cc], descending=[False, True, False]).drop("_rsv", "_pos"))


# =====================================================================================================
# public API
# =====================================================================================================
def _generate_country(s1: pl.DataFrame, pool: pl.DataFrame, cfg: BlockingConfig, stats: dict) -> pl.DataFrame:
    t0 = time.time()
    q = _prep(s1, "qi")
    p = _prep(pool, "pi")
    _log(cfg, f"country block: S1={q.height:,} pool={p.height:,}")
    key_pairs = _run_key_passes(q, p, cfg, stats) if cfg.key_passes else []
    tf_pairs = _run_tfidf_passes(q, p, cfg, stats)
    u = _union(key_pairs, tf_pairs, cfg)
    del key_pairs, tf_pairs
    gc.collect()
    n_before = u.height
    u = apply_prio_model(u, cfg)
    cap = cfg.max_cands_per_s1
    if cap is not None:
        n_plain = int(u.group_by("qi").len().select(pl.col("len").clip(upper_bound=cap).sum()).item() or 0)
        u = cap_candidates(u, cap, reserve_expr(cfg, u.columns), cfg.reserve_slots)
        stats["reserve_added"] = stats.get("reserve_added", 0) + u.height - n_plain
    for k in cfg.key_passes:
        if k in STREET_PASSES:
            stats.setdefault("street_pass_kept", {})[k] = stats.get("street_pass_kept", {}).get(k, 0) + \
                int(u[f"p_key_{k}"].sum())
    u = (u.join(q.select("qi", pl.col("id").alias("s1")), on="qi", how="left", maintain_order="left")
          .join(p.select("pi", pl.col("id").alias("cand")), on="pi", how="left", maintain_order="left")
          .drop("qi", "pi").sort(["s1", "cand"]))                              # canonical, run-to-run identical
    lead = ["s1", "cand"]
    u = u.select(lead + [c for c in u.columns if c not in lead])
    stats["pairs_before_cap"] = stats.get("pairs_before_cap", 0) + n_before
    stats["pairs"] = stats.get("pairs", 0) + u.height
    stats["seconds"] = round(stats.get("seconds", 0.0) + time.time() - t0, 2)
    stats["peak_rss_gb"] = max(stats.get("peak_rss_gb", 0.0), _rss_gb())
    _log(cfg, f"union: {n_before:,} pairs -> cap {cfg.max_cands_per_s1}: {u.height:,} "
              f"(+{stats.get('reserve_added', 0):,} reserve) ({time.time() - t0:.1f}s)")
    return u


def generate_candidates(s1: pl.DataFrame, pool: pl.DataFrame, cfg: Optional[BlockingConfig] = None,
                        stats: Optional[dict] = None) -> pl.DataFrame:
    """CONTRACT. s1, pool = normalized frames (pool = S2 u S3). Runs every country present in s1 separately
    (country is an open set: France or any unseen label works the same way). Returns s1, cand + pass columns."""
    cfg = cfg or BlockingConfig()
    stats = {} if stats is None else stats
    has_c = "country" in s1.columns and "country" in pool.columns
    countries = sorted(s1["country"].fill_null("").unique().to_list()) if has_c else [None]
    parts = []
    for c in countries:
        s1c = s1.filter(pl.col("country").fill_null("") == c) if c is not None else s1
        pc = pool.filter(pl.col("country").fill_null("") == c) if c is not None else pool
        if s1c.height == 0 or pc.height == 0:
            continue
        st = stats.setdefault("by_country", {}).setdefault(str(c), {})
        parts.append(_generate_country(s1c, pc, cfg, st))
    stats["pairs"] = sum(v.get("pairs", 0) for v in stats.get("by_country", {}).values())
    if not parts:
        return _union([], [], cfg).with_columns(pl.lit(None, dtype=pl.Utf8).alias("s1"), pl.lit(None, dtype=pl.Utf8).alias("cand")) \
            .drop("qi", "pi").select(["s1", "cand"] + [c for c in _union([], [], cfg).columns if c not in ("qi", "pi")])
    return pl.concat(parts, how="vertical_relaxed")


def pass_hit_expr(name: str) -> pl.Expr:
    """Boolean expression 'this candidate was produced by pass <name>' ('key_tok', 'name_fwd', 'addr_rev', ...)."""
    if name.startswith("key_"):
        return pl.col(f"p_{name}").fill_null(False)
    fld, direction = name.rsplit("_", 1)
    return pl.col(f"p_{fld}_{'rank' if direction == 'fwd' else 'rrank'}").is_not_null()


def candidate_recall(cands: pl.DataFrame, gt: pl.DataFrame, s1_ids=None) -> dict:
    """CONTRACT. cands: (s1, cand, ...); gt: long (s1, mid); s1_ids: evaluated S1 (Series / list / frame with
    entity_id|s1 [+ country]); default = all S1 in gt or cands. Returns pair_recall, s1_full_recall (share of
    S1 with >=1 link whose links are ALL candidates), cands/S1 stats over every evaluated S1 (0 for S1 without
    candidates), oracle_f05 (macro F0.5 if the matcher were perfect on these candidates) and by_country."""
    meta = None
    if s1_ids is None:
        ev = pl.concat([gt.select(pl.col("s1").cast(pl.Utf8)), cands.select(pl.col("s1").cast(pl.Utf8))]).unique()
    elif isinstance(s1_ids, pl.DataFrame):
        idc = "s1" if "s1" in s1_ids.columns else "entity_id"
        ev = s1_ids.select(pl.col(idc).cast(pl.Utf8).alias("s1")).unique()
        if "country" in s1_ids.columns:
            meta = s1_ids.select(pl.col(idc).cast(pl.Utf8).alias("s1"), pl.col("country").cast(pl.Utf8)).unique("s1")
    else:
        ev = pl.DataFrame({"s1": pl.Series(list(s1_ids), dtype=pl.Utf8)}).unique()
    g = gt.select(pl.col("s1").cast(pl.Utf8), pl.col("mid").cast(pl.Utf8)).unique().join(ev, on="s1", how="semi")
    c = cands.select(pl.col("s1").cast(pl.Utf8), pl.col("cand").cast(pl.Utf8).alias("mid")).unique().join(ev, on="s1", how="semi")
    hit = g.join(c, on=["s1", "mid"], how="semi")
    res = _recall_numbers(ev, g, c, hit)
    if meta is not None:
        byc = {}
        for ctry in sorted(meta["country"].fill_null("?").unique().to_list()):
            evc = meta.filter(pl.col("country").fill_null("?") == ctry).select("s1")
            byc[ctry] = _recall_numbers(evc, g.join(evc, on="s1", how="semi"), c.join(evc, on="s1", how="semi"),
                                        hit.join(evc, on="s1", how="semi"))
        res["by_country"] = byc
    return res


def _oracle_f05(ev: pl.DataFrame, g: pl.DataFrame, hit: pl.DataFrame) -> float:
    try:
        try:
            from . import metric as _M          # type: ignore
        except ImportError:
            try:
                import metric as _M             # type: ignore
            except ImportError:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "er-f05-decisions"))
                import metric as _M             # type: ignore
        return _M.macro_f05(hit, g, ev)
    except ImportError:                          # same formula as metric.py (P = 1 on the oracle)
        k = g.group_by("s1").len("k")
        tp = hit.group_by("s1").len("tp")
        r = ev.join(k, on="s1", how="left").join(tp, on="s1", how="left").fill_null(0)
        f = pl.when(pl.col("k") == 0).then(1.0).when(pl.col("tp") == 0).then(0.0) \
              .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("k") + pl.col("tp")))
        return float(r.select(f.mean()).item())


def _recall_numbers(ev: pl.DataFrame, g: pl.DataFrame, c: pl.DataFrame, hit: pl.DataFrame) -> dict:
    n_true = g.height
    per = ev.join(c.group_by("s1").len("n"), on="s1", how="left").fill_null(0)["n"]
    kk = g.group_by("s1").len("k").join(hit.group_by("s1").len("tp"), on="s1", how="left").fill_null(0)
    return {
        "n_s1": ev.height,
        "n_true_pairs": n_true,
        "n_cand_pairs": c.height,
        "pair_recall": (hit.height / n_true) if n_true else float("nan"),
        "s1_full_recall": float((kk["tp"] == kk["k"]).mean()) if kk.height else float("nan"),
        "s1_zero_recall": float((kk["tp"] == 0).mean()) if kk.height else float("nan"),
        "cands_per_s1_mean": float(per.mean()) if per.len() else 0.0,
        "cands_per_s1_p50": float(per.quantile(0.5)) if per.len() else 0.0,
        "cands_per_s1_p95": float(per.quantile(0.95)) if per.len() else 0.0,
        "cands_per_s1_max": int(per.max()) if per.len() else 0,
        "oracle_f05": _oracle_f05(ev, g, hit),
    }


def geo_shards(s1: pl.DataFrame, pool: pl.DataFrame, cfg: Optional[BlockingConfig] = None,
               max_pool_rows: int = 1_500_000) -> Iterator[Tuple[str, str, pl.DataFrame, pl.DataFrame]]:
    """Laptop-scale sharding for full train/test: yields (country, shard_name, s1_part, pool_part) where the parts are
    consecutive geo blocks packed until the pool part reaches `max_pool_rows`; every shard also receives the
    country's unknown-geo pool rows so the fallback block is preserved. Key passes then run per shard (cross-shard
    key hits are lost; measure that loss with candidate_recall on train before relying on it)."""
    cfg = cfg or BlockingConfig()
    bc = cfg.block_col
    for c in sorted(s1["country"].fill_null("").unique().to_list()):
        s1c = s1.filter(pl.col("country").fill_null("") == c)
        pc = pool.filter(pl.col("country").fill_null("") == c)
        unk = pc.filter(pl.col(bc).fill_null("") == "")
        sizes = pc.group_by(pl.col(bc).fill_null("")).len().filter(pl.col(bc) != "")
        s1_blocks = s1c.group_by(pl.col(bc).fill_null("")).len().rename({"len": "n_s1"})
        order = s1_blocks.join(sizes, on=bc, how="left").fill_null(0).sort(bc)
        cur, cur_rows, k = [], 0, 0
        rows = order.iter_rows(named=True)
        for r in list(rows) + [None]:
            if r is not None and (cur_rows + r["len"] <= max_pool_rows or not cur):
                cur.append(r[bc]); cur_rows += r["len"]
                continue
            if cur:
                s_part = s1c.filter(pl.col(bc).fill_null("").is_in(cur))
                p_part = pl.concat([pc.filter(pl.col(bc).fill_null("").is_in([x for x in cur if x != ""])), unk])
                if "" in cur:                       # S1 with unknown geo need the whole country pool
                    p_part = pc
                yield c, f"{c}#{k}", s_part, p_part.unique("entity_id", maintain_order=True)
                k += 1
            cur, cur_rows = ([r[bc]], r["len"]) if r is not None else ([], 0)
