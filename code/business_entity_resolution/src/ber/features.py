# -*- coding: utf-8 -*-
"""features.py -- pairwise, context, competition and group features for (S1, candidate) pairs
(Amazon ML Challenge 2026, Business Entity Resolution, team Master Bolt).
Copy into code/business_entity_resolution/src/ber/features.py.

Contract
--------
FEATURE_COLUMNS : list[str]                       fixed order used by the model (float32)
compute_features(cands, s1, pool, cfg=None) -> pl.DataFrame
    cands : s1, cand [+ label] [+ blocking columns p_key_*, p_<t>_score/_rank/_rrank, n_passes, p_key_hit, prio]
    s1, pool : NORMALIZED frames (entity_id, name_norm, name_core, name_compact, addr_norm, house_nums,
               city_key, state_key, script, name_is_domain) -- see er-text-normalization
    returns s1, cand [, label] + FEATURE_COLUMNS, one row per input pair, same order as `cands`.

Feature families (why they exist -> the noise operator they survive)
    name   : ratio / token_set / token_sort / partial / Jaro-Winkler on name_norm and name_core, normalized
             Levenshtein, sorted-token key equality, glued-name (domain form) ratio + containment, consonant
             skeleton ratio (transliteration vowel loss), token Jaccard, IDF-weighted overlap and containment
             (generic words like "group", "private" weigh little), legal-form agreement / conflict, lengths
    address: ratio / token_set / partial on addr_norm, house-number exact / base-digit / numeric distance
             (1804 vs 1804A, 3412 vs 3411), street-word Jaccard, city / state agreement with unknown flags
    record : candidate from S3, candidate originally non-Latin script, domain-form name, empty address
    block  : every blocking pass flag / score / rank (missing -> neutral default)
    context: rank of the pair inside its S1 by a base score, gap to the S1's best, number of candidates
    compet.: the candidate's best base score over ALL S1 in the frame, is-argmax, gap to the best other S1,
             number of S1 claiming it (exclusivity: each S2/S3 record belongs to at most one S1)
    group  : similarity of the candidate to the S1's other top candidates (S2/S3 hold duplicates of the
             same entity, so a true candidate usually resembles the other true candidates)
The country value is NEVER a feature (France is unseen in training).

Scale: rapidfuzz process.cpdist(workers=-1) does 0.7-3.6M pairs/s per scorer on the laptop. Pairs are processed
in chunks of whole S1 groups (cfg.chunk_pairs) so context/group features see every candidate of an S1; the
competition features are computed once over the whole frame, so pass a complete country (or geo shard + its
fallback pool) at a time. Memory ~ (len(FEATURE_COLUMNS) x 4 B + ~120 B) per pair.

Usage
-----
    from ber.features import FEATURE_COLUMNS, FeatureConfig, compute_features
    feats = compute_features(cands, s1_norm, pool_norm, FeatureConfig(chunk_pairs=2_000_000))
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

try:                                  # inside the ber package
    from . import normalize as _N     # type: ignore
except ImportError:                   # reference layout: sibling skill folder or same folder
    try:
        import normalize as _N        # type: ignore
    except ImportError:
        import importlib.util as _ilu
        import os as _os
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "er-text-normalization", "normalize.py")
        import sys as _sys
        _spec = _ilu.spec_from_file_location("normalize", _p)
        _N = _ilu.module_from_spec(_spec)
        _sys.modules["normalize"] = _N
        _spec.loader.exec_module(_N)  # type: ignore

# --------------------------------------------------------------------------------------------------------------
# feature list (fixed order)
# --------------------------------------------------------------------------------------------------------------
NAME_FEATS = ["n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw",
              "c_ratio", "c_tset", "c_partial", "c_jw", "c_lev",
              "k_eq", "g_ratio", "g_contain", "sk_ratio",
              "tok_jacc", "tok_inter", "idf_overlap", "idf_contain", "idf_s1", "idf_cand",
              "len_s1", "len_cand", "len_ratio", "legal_agree", "legal_conflict"]
ADDR_FEATS = ["a_ratio", "a_tset", "a_partial", "a_empty_any",
              "hn_both", "hn_exact", "hn_base", "hn_diff", "st_jacc", "st_inter", "city_eq", "state_eq"]
REC_FEATS = ["cand_is_s3", "cand_nonlatin", "domain_any"]
KEY_PASSES = ["name", "tok", "hn", "skel", "compact", "addr"]
TFIDF_PASSES = ["name", "addr", "both"]
BLOCK_FEATS = ([f"p_key_{k}" for k in KEY_PASSES]
               + [f"p_{t}_{s}" for t in TFIDF_PASSES for s in ("score", "rank", "rrank")]
               + ["n_passes", "p_key_hit", "prio"])
CTX_FEATS = ["base", "ctx_rank", "ctx_gap", "ctx_ncand"]
COMP_FEATS = ["comp_best", "comp_is_best", "comp_gap", "comp_nclaim"]
GRP_FEATS = ["grp_max_sim", "grp_n_sim"]
# name ambiguity (label-free counts inside the same split/country frame): how many S1 / pool records share the
# normalized name key of the candidate and of the S1. An address-less candidate whose name is unique among S1 is
# almost surely that S1's duplicate; a generic name ("Department of Veterans Affairs") is not. Measured on the v1
# full OOF: empty_address = 57% of model misses, 55% of blocking misses.
AMB_FEATS = ["amb_s1_c", "amb_s1_q", "amb_pool_c", "amb_pool_q"]
# the v3 model's 72 columns (order and definitions frozen: saved v3 models reference them by name)
FEATURE_COLUMNS_V3: List[str] = (NAME_FEATS + ADDR_FEATS + REC_FEATS + BLOCK_FEATS + CTX_FEATS + COMP_FEATS + GRP_FEATS
                                 + AMB_FEATS)

# ---- v4 groups (label-free; all computed by pair_features_v4 / competition_v4, see their docstrings) -------------
# a. house-number relation of the BEST-matching number pair (order-robust) + multiset relation of all address numbers.
#    Train US same-name+city pairs, P(true) by relation (work/reports/diag_results.md): copy edits exact / letter change /
#    digit added / dropped / big one-digit substitution 0.96-0.9999; distractor shifts (diff 3-100, last digit +-3..9)
#    0.02-0.13; +-1/2 0.26. hn_diff (first numbers, absolute) cannot express 495 vs 95.
HNREL_FEATS = ["hn_rel", "hn_r_copy", "hn_r_indel", "hn_r_letter", "hn_r_small", "hn_r_hard", "hn_rdiff", "hn_ldiff",
               "hn_mset"]
# b. street-only overlap: address words minus city / state / region words (both sides' keys), street types, unit,
#    designator, floor, stop and NULL words (France S1 carry the region name in every address -> st_inter, a_tset shift)
STREET_FEATS = ["st2_inter", "st2_jacc", "st2_tset"]
# c. legal-form-aware name signals: sorted name_norm tokens equal (keeps SARL / fils / freres / cie that name_core
#    strips), #S1 sharing that key, and a second competition score on name_norm instead of name_core
NAMEN_FEATS = ["norm_eq", "amb_s1n_c", "amb_s1n_q", "base_n", "comp_n_best", "comp_n_gap", "comp_n_is_best"]
# d. acronym: one side's compact name equals the initials of the other side's name_norm / name_core tokens
ACR_FEATS = ["acr_match"]
# e. co-located S1: #other S1 at the S1's address key, #other S1 at the candidate's address key
COLOC_FEATS = ["n_s1_akey_q", "n_s1_akey_c"]
# f. within-frame percentile ranks (scale-free across countries / pool sizes) of the record IDF mass and core length
PCT_FEATS = ["pct_idf_s1", "pct_idf_cand", "pct_len_s1", "pct_len_cand"]
V4_FEATS: List[str] = HNREL_FEATS + STREET_FEATS + NAMEN_FEATS + ACR_FEATS + COLOC_FEATS + PCT_FEATS
# g. OPTIONAL candidate-record noise flags (FeatureConfig.record_flags): the generator applies these noise operators
#    to copies only (train P(linked | flag): DBA 0.9998, lower-case 0.86-0.89, junk lead 0.83, accent 0.82-0.84 vs
#    base 0.73-0.75) -> a synthetic-data artifact; measured separately.
RECORD_FLAG_FEATS = ["cand_dba", "cand_lower", "cand_upper", "cand_accent", "cand_junk", "cand_bracket", "cand_addr_null"]
FEATURE_COLUMNS: List[str] = FEATURE_COLUMNS_V3 + V4_FEATS

RANK_MISSING = 99.0         # rank of a pass that did not retrieve the pair


@dataclass
class FeatureConfig:
    chunk_pairs: int = 2_000_000   # pairs per chunk (whole S1 groups)
    workers: int = -1              # rapidfuzz threads
    group_top: int = 5             # S1's top-N other candidates used by the group features
    group_sim_thr: float = 0.8
    verbose: bool = True
    record_flags: bool = False     # also emit RECORD_FLAG_FEATS (needs raw business_name / business_address columns)
    hn_max_nums: int = 4           # house numbers per side searched for the best-matching pair (<= 16 number pairs)


def feature_columns(cfg: Optional[FeatureConfig] = None) -> List[str]:
    """Model columns produced by compute_features under cfg (FEATURE_COLUMNS [+ RECORD_FLAG_FEATS])."""
    return FEATURE_COLUMNS + (RECORD_FLAG_FEATS if cfg is not None and cfg.record_flags else [])


def _log(cfg, *a):
    if cfg.verbose:
        print(time.strftime("%H:%M:%S"), "[features]", *a, flush=True)


# --------------------------------------------------------------------------------------------------------------
# side preparation (once per frame)
# --------------------------------------------------------------------------------------------------------------
def _s(df: pl.DataFrame, c: str) -> pl.Expr:
    return pl.col(c).cast(pl.Utf8).fill_null("") if c in df.columns else pl.lit("")


def _skeleton_lut(tokens: pl.Series) -> pl.DataFrame:
    u = tokens.drop_nulls().unique()
    return pl.DataFrame({"t": u, "sk": [_N.skeleton_key(x) for x in u.to_list()]},
                        schema={"t": pl.Utf8, "sk": pl.Utf8})


def _addr_stop_words() -> frozenset:
    """Address words that are not street names: every street-type / directional / unit / designator / floor word of the
    hand-written normalize maps (all countries: the set must be country-agnostic), French articles, NULL tokens and
    city-affix noise. 'main' / 'cross' stay (US 'Main St'; Indian '5th Main' carries them on both sides)."""
    s = set()
    for d in _N.STREET_TYPES.values():
        for k, vs in d.items():
            for x in [k] + list(vs):
                s.update(x.split())
    s |= set(_N.UNIT_WORDS) | set(_N.DESIGNATORS) | set(_N._FLOOR_WORDS) | set(_N.FR_ADDR_STOP)
    for x in _N.NULL_TOKENS:
        s.update(t for t in x.replace("/", " ").replace("<", " ").replace(">", " ").replace(".", " ").split())
    s |= {"and", "of", "the", "city", "town", "village", "county", "township", "twp", "cdp", "box", "cedex", "region",
          "urban", "district"}
    s -= {"main", "cross"}
    return frozenset(t for t in s if t.isalpha())


ADDR_STOP = _addr_stop_words()
NAME_STOP_INITIALS = frozenset(set(_N.FR_ARTICLES) | set(_N.NAME_FILLERS))


def _tokens_expr(col: str) -> pl.Expr:
    return pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))


def _initials_expr(col: str, stop: Optional[frozenset] = None) -> pl.Expr:
    t = _tokens_expr(col)
    if stop:
        t = t.list.eval(pl.element().filter(~pl.element().is_in(list(stop))))
    return t.list.eval(pl.element().str.slice(0, 1)).list.join("")


def _prep_side_v4(d: pl.DataFrame, df: pl.DataFrame, record_flags: bool) -> pl.DataFrame:
    """Record-level columns of the v4 groups (appended to _prep_side's output; v3 columns untouched).
    keyn   sorted unique name_norm tokens            ini_n / ini_c / ini_s  initials of name_norm / name_core /
    stw    street words (see ADDR_STOP) minus the record's own city / state tokens     core without articles
    cst    city + state tokens (removed from the OTHER side's street words per pair)
    hns    sorted unique base house numbers           anum  sorted digit groups of the whole address (multiset)
    mkey / anu  anum joined (multiset equality) / its unique values
    akey   address key hns | stw | city ('' when neither numbers nor street words)
    rf_*   raw-string noise flags of the record (record_flags=True)"""
    stop = list(ADDR_STOP)
    d = d.with_columns(
        pl.col("tokn").list.sort().list.join(" ").alias("keyn"),
        _initials_expr("nn").alias("ini_n"), _initials_expr("nc").alias("ini_c"),
        _initials_expr("nc", NAME_STOP_INITIALS).alias("ini_s"),
        pl.concat_list(_tokens_expr("city"), _tokens_expr("state")).list.unique().alias("cst"),
        pl.col("hn").list.eval(pl.element().str.extract(r"(\d+)", 1).str.strip_chars_start("0"))
          .list.drop_nulls().list.unique().list.sort().alias("hns"),
        pl.col("ad").str.extract_all(r"\d+").list.sort().alias("anum"),     # addr_norm numbers are canonical already
    )
    d = d.with_columns(
        _tokens_expr("ad").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$")
                                                         & ~pl.element().is_in(stop)))
          .list.set_difference(pl.col("cst")).list.unique().list.sort().alias("stw"),
    )
    d = d.with_columns(
        pl.when((pl.col("hns").list.len() > 0) | (pl.col("stw").list.len() > 0))
          .then(pl.concat_str([pl.col("hns").list.join(","), pl.col("stw").list.join(" "), pl.col("city")], separator="|"))
          .otherwise(pl.lit("")).alias("akey"),
        pl.col("anum").list.join(" ").alias("mkey"),
        pl.col("anum").list.unique().alias("anu"),
    )
    if record_flags:
        n = (pl.col("business_name").cast(pl.Utf8).fill_null("") if "business_name" in df.columns else pl.lit(""))
        a = (pl.col("business_address").cast(pl.Utf8).fill_null("") if "business_address" in df.columns else pl.lit(""))
        raw = df.select(n.alias("_n"), a.alias("_a"))
        n, a = pl.col("_n"), pl.col("_a")
        has_l = n.str.contains(r"[A-Za-z]")
        flags = raw.select(
            n.str.contains(r"(?i)\bd\.?b\.?a\b").alias("rf_dba"),
            ((n == n.str.to_lowercase()) & has_l).alias("rf_lower"),
            ((n == n.str.to_uppercase()) & has_l).alias("rf_upper"),
            n.str.contains(r"[À-ɏ]").alias("rf_accent"),
            n.str.contains(r"^\s*[^\w\s]").alias("rf_junk"),
            n.str.contains(r"[\[\]]").alias("rf_bracket"),
            a.str.contains(r"(?i)\b(?:null|none)\b").alias("rf_addr_null"),
        )
        d = d.hstack([s.cast(pl.Float32) for s in flags.get_columns()])
    return d


def _prep_side(df: pl.DataFrame, record_flags: bool = False) -> pl.DataFrame:
    """Columns the scorers need, derived once per record (not per pair)."""
    d = df.select(
        pl.col("entity_id").cast(pl.Utf8).alias("id"),
        _s(df, "name_norm").alias("nn"), _s(df, "name_core").alias("nc"),
        _s(df, "name_compact").alias("ng"), _s(df, "addr_norm").alias("ad"),
        _s(df, "city_key").alias("city"), _s(df, "state_key").alias("state"),
        _s(df, "script").alias("script"),
        (pl.col("name_is_domain").fill_null(False) if "name_is_domain" in df.columns else pl.lit(False)).alias("dom"),
        (pl.col("house_nums").cast(pl.List(pl.Utf8)).fill_null([]) if "house_nums" in df.columns
         else pl.lit([], dtype=pl.List(pl.Utf8))).alias("hn"),
    )
    d = d.with_columns(
        pl.when(pl.col("ng") == "").then(pl.col("nc").str.replace_all(" ", "", literal=True)).otherwise(pl.col("ng")).alias("ng"),
        pl.col("nc").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("tok"),
        pl.col("nn").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("tokn"),
        pl.col("ad").str.split(" ").list.eval(pl.element().filter(pl.element().str.contains(r"^[a-z]{3,}$"))).list.unique().alias("aw"),
        pl.col("hn").list.eval(pl.element().str.extract(r"(\d+)", 1).str.strip_chars_start("0")).list.drop_nulls().list.unique().alias("hnb"),
    )
    d = d.with_columns(
        pl.col("tok").list.sort().list.join(" ").alias("key"),
        pl.col("tokn").list.set_difference(pl.col("tok")).alias("legal"),
    )
    return _prep_side_v4(d, df, record_flags)


def _add_skel_and_idf(side: pl.DataFrame, lut: pl.DataFrame, idf: pl.DataFrame, max_idf: float) -> pl.DataFrame:
    idx = side.with_row_index("_r")
    ex = idx.select("_r", pl.col("tok").alias("t")).explode("t").drop_nulls("t")
    sk = (ex.join(lut, on="t", how="left").group_by("_r", maintain_order=True)
            .agg(pl.col("sk").drop_nulls().str.join(" ").alias("skel")))
    w = (ex.join(idf, on="t", how="left").with_columns(pl.col("idf").fill_null(max_idf))
           .group_by("_r").agg(pl.col("idf").sum().alias("w")))
    out = idx.join(sk, on="_r", how="left").join(w, on="_r", how="left")
    return out.with_columns(pl.col("skel").fill_null(""), pl.col("w").fill_null(0.0).cast(pl.Float32)).sort("_r").drop("_r")


# --------------------------------------------------------------------------------------------------------------
# per-chunk pairwise scores
# --------------------------------------------------------------------------------------------------------------
def _cp(a: List[str], b: List[str], scorer, workers: int) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _pair_scores(P: pl.DataFrame, cfg: FeatureConfig, idf: pl.DataFrame, max_idf: float) -> pl.DataFrame:
    """P has the joined side columns with suffixes _q (S1) and _c (candidate)."""
    w = cfg.workers
    L = lambda c: P[c].to_list()
    nn_q, nn_c, nc_q, nc_c = L("nn_q"), L("nn_c"), L("nc_q"), L("nc_c")
    ad_q, ad_c = L("ad_q"), L("ad_c")
    out = {
        "n_ratio": _cp(nn_q, nn_c, fuzz.ratio, w) / 100, "n_tset": _cp(nn_q, nn_c, fuzz.token_set_ratio, w) / 100,
        "n_tsort": _cp(nn_q, nn_c, fuzz.token_sort_ratio, w) / 100, "n_partial": _cp(nn_q, nn_c, fuzz.partial_ratio, w) / 100,
        "n_jw": _cp(nn_q, nn_c, JaroWinkler.normalized_similarity, w),
        "c_ratio": _cp(nc_q, nc_c, fuzz.ratio, w) / 100, "c_tset": _cp(nc_q, nc_c, fuzz.token_set_ratio, w) / 100,
        "c_partial": _cp(nc_q, nc_c, fuzz.partial_ratio, w) / 100, "c_jw": _cp(nc_q, nc_c, JaroWinkler.normalized_similarity, w),
        "c_lev": _cp(nc_q, nc_c, Levenshtein.normalized_similarity, w),
        "g_ratio": _cp(L("ng_q"), L("ng_c"), fuzz.ratio, w) / 100,
        "sk_ratio": _cp(L("skel_q"), L("skel_c"), fuzz.ratio, w) / 100,
        "a_ratio": _cp(ad_q, ad_c, fuzz.ratio, w) / 100, "a_tset": _cp(ad_q, ad_c, fuzz.token_set_ratio, w) / 100,
        "a_partial": _cp(ad_q, ad_c, fuzz.partial_ratio, w) / 100,
    }
    F = pl.DataFrame(out)

    # IDF-weighted overlap of core tokens: sum idf(common) / sum idf(union)
    inter = P.select(pl.int_range(0, pl.len(), dtype=pl.Int64).alias("_i"),
                     pl.col("tok_q").list.set_intersection(pl.col("tok_c")).alias("t"))
    wi = (inter.explode("t").drop_nulls("t").join(idf, on="t", how="left")
               .with_columns(pl.col("idf").fill_null(max_idf)).group_by("_i").agg(pl.col("idf").sum().alias("wi")))
    wi = inter.select("_i").join(wi, on="_i", how="left").sort("_i")["wi"].fill_null(0.0).to_numpy()
    wq, wc = P["w_q"].to_numpy(), P["w_c"].to_numpy()
    wu = np.maximum(wq + wc - wi, 1e-6)

    ex = P.select(
        (pl.col("key_q") == pl.col("key_c")).cast(pl.Float32).alias("k_eq"),
        ((pl.col("ng_q").str.len_chars() >= 5) & (pl.col("ng_c").str.len_chars() >= 5)
         & (pl.col("ng_q").str.contains(pl.col("ng_c"), literal=True) | pl.col("ng_c").str.contains(pl.col("ng_q"), literal=True))
         ).cast(pl.Float32).alias("g_contain"),
        pl.col("tok_q").list.set_intersection(pl.col("tok_c")).list.len().cast(pl.Float32).alias("tok_inter"),
        pl.col("tok_q").list.set_union(pl.col("tok_c")).list.len().cast(pl.Float32).alias("_tu"),
        pl.col("tok_q").list.len().cast(pl.Float32).alias("len_s1"),
        pl.col("tok_c").list.len().cast(pl.Float32).alias("len_cand"),
        (pl.min_horizontal(pl.col("nc_q").str.len_chars(), pl.col("nc_c").str.len_chars()).cast(pl.Float32)
         / pl.max_horizontal(pl.col("nc_q").str.len_chars(), pl.col("nc_c").str.len_chars(), pl.lit(1)).cast(pl.Float32)).alias("len_ratio"),
        ((pl.col("legal_q").list.len() > 0) & (pl.col("legal_c").list.len() > 0)
         & (pl.col("legal_q").list.set_intersection(pl.col("legal_c")).list.len() > 0)).cast(pl.Float32).alias("legal_agree"),
        ((pl.col("legal_q").list.len() > 0) & (pl.col("legal_c").list.len() > 0)
         & (pl.col("legal_q").list.set_intersection(pl.col("legal_c")).list.len() == 0)).cast(pl.Float32).alias("legal_conflict"),
        ((pl.col("ad_q") == "") | (pl.col("ad_c") == "")).cast(pl.Float32).alias("a_empty_any"),
        ((pl.col("hn_q").list.len() > 0) & (pl.col("hn_c").list.len() > 0)).cast(pl.Float32).alias("hn_both"),
        (pl.col("hn_q").list.set_intersection(pl.col("hn_c")).list.len() > 0).cast(pl.Float32).alias("hn_exact"),
        (pl.col("hnb_q").list.set_intersection(pl.col("hnb_c")).list.len() > 0).cast(pl.Float32).alias("hn_base"),
        pl.col("hnb_q").list.first().cast(pl.Float64, strict=False).alias("_h1"),
        pl.col("hnb_c").list.first().cast(pl.Float64, strict=False).alias("_h2"),
        pl.col("aw_q").list.set_intersection(pl.col("aw_c")).list.len().cast(pl.Float32).alias("st_inter"),
        pl.col("aw_q").list.set_union(pl.col("aw_c")).list.len().cast(pl.Float32).alias("_su"),
        pl.when((pl.col("city_q") == "") | (pl.col("city_c") == "")).then(-1.0)
          .otherwise((pl.col("city_q") == pl.col("city_c")).cast(pl.Float32)).cast(pl.Float32).alias("city_eq"),
        pl.when((pl.col("state_q") == "") | (pl.col("state_c") == "")).then(-1.0)
          .otherwise((pl.col("state_q") == pl.col("state_c")).cast(pl.Float32)).cast(pl.Float32).alias("state_eq"),
        pl.col("cand").str.starts_with("S3-").cast(pl.Float32).alias("cand_is_s3"),
        ((pl.col("script_c") != "") & (pl.col("script_c") != "latin")).cast(pl.Float32).alias("cand_nonlatin"),
        (pl.col("dom_q") | pl.col("dom_c")).cast(pl.Float32).alias("domain_any"),
    )
    ex = ex.with_columns(
        (pl.col("tok_inter") / pl.max_horizontal(pl.col("_tu"), pl.lit(1.0))).alias("tok_jacc"),
        (pl.col("st_inter") / pl.max_horizontal(pl.col("_su"), pl.lit(1.0))).alias("st_jacc"),
        pl.when(pl.col("_h1").is_null() | pl.col("_h2").is_null()).then(-1.0)
          .otherwise(((pl.col("_h1") - pl.col("_h2")).abs() + 1).log()).cast(pl.Float32).alias("hn_diff"),
    ).drop("_tu", "_su", "_h1", "_h2")
    F = F.hstack(ex.get_columns()).with_columns(
        pl.Series("idf_overlap", (wi / wu).astype(np.float32)),
        pl.Series("idf_contain", (wi / np.maximum(np.minimum(wq, wc), 1e-6)).astype(np.float32)),
        pl.Series("idf_s1", wq.astype(np.float32)), pl.Series("idf_cand", wc.astype(np.float32)),
    )
    return F


def _block_feats(P: pl.DataFrame) -> pl.DataFrame:
    cols = []
    for k in KEY_PASSES:
        c = f"p_key_{k}"
        cols.append((pl.col(c).fill_null(False).cast(pl.Float32) if c in P.columns else pl.lit(0.0, pl.Float32)).alias(c))
    for t in TFIDF_PASSES:
        for s, dflt in (("score", 0.0), ("rank", RANK_MISSING), ("rrank", RANK_MISSING)):
            c = f"p_{t}_{s}"
            cols.append((pl.col(c).cast(pl.Float32).fill_null(dflt) if c in P.columns else pl.lit(dflt, pl.Float32)).alias(c))
    for c in ("n_passes", "p_key_hit", "prio"):
        cols.append((pl.col(c).cast(pl.Float32).fill_null(0.0) if c in P.columns else pl.lit(0.0, pl.Float32)).alias(c))
    return P.select(cols)


def _group_feats(P: pl.DataFrame, F: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    """Similarity of each candidate to the S1's top-N OTHER candidates (by base score)."""
    g = pl.DataFrame({"_i": np.arange(P.height, dtype=np.int64), "s1": P["s1"], "nc": P["nc_c"], "ad": P["ad_c"],
                      "base": F["base"]})
    top = (g.sort(["s1", "base", "_i"], descending=[False, True, False])
             .group_by("s1", maintain_order=True).head(cfg.group_top))          # ties broken by row -> deterministic
    pairs = g.select("_i", "s1", "nc", "ad").join(top.select("s1", pl.col("_i").alias("_o"), pl.col("nc").alias("nc_o"),
                                                           pl.col("ad").alias("ad_o")), on="s1").filter(pl.col("_i") != pl.col("_o"))
    if pairs.height == 0:
        return pl.DataFrame({"grp_max_sim": np.zeros(P.height, np.float32), "grp_n_sim": np.zeros(P.height, np.float32)})
    sim = (0.5 * _cp(pairs["nc"].to_list(), pairs["nc_o"].to_list(), fuzz.token_set_ratio, cfg.workers)
           + 0.5 * _cp(pairs["ad"].to_list(), pairs["ad_o"].to_list(), fuzz.ratio, cfg.workers)) / 100
    agg = (pairs.select("_i").with_columns(pl.Series("sim", sim))
                .group_by("_i").agg(pl.col("sim").max().alias("grp_max_sim"),
                                    (pl.col("sim") >= cfg.group_sim_thr).sum().cast(pl.Float32).alias("grp_n_sim")))
    out = g.select("_i").join(agg, on="_i", how="left").sort("_i")
    return out.select(pl.col("grp_max_sim").fill_null(0.0).cast(pl.Float32), pl.col("grp_n_sim").fill_null(0.0).cast(pl.Float32))


# --------------------------------------------------------------------------------------------------------------
# v4 groups: house-number relation, street-only overlap, legal-form-aware names, acronyms, co-location, percentiles
# --------------------------------------------------------------------------------------------------------------
# ordinal code of the house-number relation (roughly ordered by train P(true): copy edits < distractor shifts)
HN_REL_CODES = {"none": 0, "one_missing": 1, "other": 2, "exact": 3, "letter_change": 4, "digit_added": 5,
                "digit_dropped": 6, "subst_big": 7, "diff_gt100": 8, "subst_1_2": 9, "diff_1_2": 10, "diff_11_100": 11,
                "diff_3_10": 12, "subst_3_9": 13}
_HN_COPY = ("exact", "letter_change", "digit_added", "digit_dropped", "subst_big")
_HN_SMALL = ("subst_1_2", "diff_1_2")
_HN_HARD = ("subst_3_9", "diff_3_10", "diff_11_100")
# multiset relation of ALL address numbers (candidate vs S1); train US same-name pairs: same 0.99985, one_changed 0.456
HN_MSET_CODES = {"cand_no_numbers": 0, "same": 1, "subset": 2, "superset": 3, "one_changed": 4, "other": 5}


def hn_relation(hn_q: pl.Series, hn_c: pl.Series, max_nums: int = 4, workers: int = -1) -> pl.DataFrame:
    """Relation of two canonical house-number lists (normalize house_nums, e.g. ['1804a'], ['31-7-15/a', '12']).
    Over the first `max_nums` numbers of each side, every number pair is classified on its digit strings
    (d = digits, leading zeros stripped):
        exact (strings equal) / letter_change (digits equal: 12 vs 12b) / digit_added, digit_dropped (candidate has one
        digit more / less, any position; or a longer prefix/suffix run: 1804 vs 184, 95 vs 495) / one-digit substitution
        split by |a-b|: subst_1_2, subst_3_9, subst_big (>= 10) / otherwise by |a-b|: diff_1_2, diff_3_10, diff_11_100,
        diff_gt100 / other (a side without digits)
    and the BEST pair is kept (priority exact < letter < indel < substitution < numeric diff, then smallest |a-b|), so
    component order ('Unit 10, 47 Bacon St') does not matter. Rows where both / one list is empty -> none / one_missing.
    Returns a frame aligned with the inputs: hn_rel (HN_REL_CODES), hn_rdiff = |a-b| / max(a, b, 1) and
    hn_ldiff = log1p |a-b| of the best pair (-1 when no numeric pair), all Float32."""
    X = pl.DataFrame({"a": hn_q, "b": hn_c}).with_row_index("_i")
    X = X.with_columns(pl.col("a").fill_null([]).list.head(max_nums), pl.col("b").fill_null([]).list.head(max_nums))
    X = X.with_columns(pl.col("a").list.len().alias("_na"), pl.col("b").list.len().alias("_nb"))
    E = X.filter((pl.col("_na") > 0) & (pl.col("_nb") > 0)).select("_i", "a", "b").explode("a").explode("b")
    E = E.with_columns(pl.col("a").fill_null(""), pl.col("b").fill_null(""))
    E = E.with_columns(pl.col("a").str.replace_all(r"\D", "").str.strip_chars_start("0").alias("da"),
                       pl.col("b").str.replace_all(r"\D", "").str.strip_chars_start("0").alias("db"))
    lev = (process.cpdist(E["da"].to_list(), E["db"].to_list(), scorer=Levenshtein.distance, workers=workers,
                          dtype=np.int32) if E.height else np.zeros(0, np.int32))
    la, lb = pl.col("da").str.len_chars().cast(pl.Int32), pl.col("db").str.len_chars().cast(pl.Int32)
    E = E.with_columns(pl.Series("_lev", lev), la.alias("_la"), lb.alias("_lb"),
                       pl.col("da").str.slice(0, 15).cast(pl.Float64, strict=False).alias("_ia"),
                       pl.col("db").str.slice(0, 15).cast(pl.Float64, strict=False).alias("_ib"))
    valid = (pl.col("_la") > 0) & (pl.col("_lb") > 0)
    dl = (pl.col("_la") - pl.col("_lb")).abs()
    run = (pl.when(pl.col("_la") > pl.col("_lb"))
             .then(pl.col("da").str.starts_with(pl.col("db")) | pl.col("da").str.ends_with(pl.col("db")))
             .otherwise(pl.col("db").str.starts_with(pl.col("da")) | pl.col("db").str.ends_with(pl.col("da"))))
    exact = pl.col("a") == pl.col("b")
    letter = valid & (pl.col("da") == pl.col("db"))
    indel = valid & (dl > 0) & (((dl == 1) & (pl.col("_lev") == 1)) | ((dl >= 2) & run))
    subst = valid & (dl == 0) & (pl.col("_lev") == 1)
    diff = (pl.col("_ia") - pl.col("_ib")).abs()
    E = E.with_columns(diff.alias("_d"), (diff / pl.max_horizontal(pl.col("_ia"), pl.col("_ib"), pl.lit(1.0))).alias("_r"))
    d = pl.col("_d")
    C = HN_REL_CODES
    E = E.with_columns(
        pl.when(exact).then(0).when(letter).then(1).when(indel).then(2).when(subst).then(3).when(valid).then(4)
          .otherwise(5).alias("_prio"),
        pl.when(exact).then(C["exact"]).when(letter).then(C["letter_change"])
          .when(indel & (pl.col("_lb") > pl.col("_la"))).then(C["digit_added"]).when(indel).then(C["digit_dropped"])
          .when(subst & (d >= 10)).then(C["subst_big"]).when(subst & (d <= 2)).then(C["subst_1_2"])
          .when(subst).then(C["subst_3_9"])
          .when(valid & (d <= 2)).then(C["diff_1_2"]).when(valid & (d <= 10)).then(C["diff_3_10"])
          .when(valid & (d <= 100)).then(C["diff_11_100"]).when(valid).then(C["diff_gt100"])
          .otherwise(C["other"]).alias("_code"),
        pl.when(valid).then(d).otherwise(None).alias("_d"),
        pl.when(valid).then(pl.col("_r")).otherwise(None).alias("_r"),
    )
    best = (E.select("_i", "_prio", "_d", "_code", "_r")
              .sort(["_i", "_prio", "_d", "_code"], nulls_last=True)
              .unique("_i", keep="first", maintain_order=True))
    out = X.select("_i", "_na", "_nb").join(best, on="_i", how="left", maintain_order="left")
    out = out.select(
        pl.when((pl.col("_na") == 0) & (pl.col("_nb") == 0)).then(C["none"])
          .when((pl.col("_na") == 0) | (pl.col("_nb") == 0)).then(C["one_missing"])
          .otherwise(pl.col("_code")).cast(pl.Float32).alias("hn_rel"),
        pl.col("_r").fill_null(-1.0).cast(pl.Float32).alias("hn_rdiff"),
        pl.when(pl.col("_d").is_null()).then(-1.0).otherwise(pl.col("_d").log1p()).cast(pl.Float32).alias("hn_ldiff"),
    )
    return out


def _mset_expr(sfx_q: str = "_q", sfx_c: str = "_c") -> pl.Expr:
    """Multiset relation of all address numbers -> HN_MSET_CODES, from the side columns mkey (sorted numbers joined)
    and anu (unique numbers)."""
    uq, uc = pl.col(f"anu{sfx_q}"), pl.col(f"anu{sfx_c}")
    inter = uq.list.set_intersection(uc).list.len()
    nq, nc = uq.list.len(), uc.list.len()
    M = HN_MSET_CODES
    return (pl.when(nc == 0).then(M["cand_no_numbers"])
              .when(pl.col(f"mkey{sfx_q}") == pl.col(f"mkey{sfx_c}")).then(M["same"])
              .when((inter == nc) & (nc < nq)).then(M["subset"])
              .when((inter == nq) & (nq < nc)).then(M["superset"])
              .when((nq == nc) & (inter == nq - 1)).then(M["one_changed"])
              .otherwise(M["other"]).cast(pl.Float32))


def base_n_expr() -> pl.Expr:
    """Second competition score on name_norm (legal forms kept): 0.5 n_tset + 0.3 a_tset + 0.2 hn_exact."""
    return (0.5 * pl.col("n_tset").cast(pl.Float32) + 0.3 * pl.col("a_tset").cast(pl.Float32)
            + 0.2 * pl.col("hn_exact").cast(pl.Float32)).cast(pl.Float32)


def side_stats_v4(S: pl.DataFrame, Q: pl.DataFrame):
    """Frame-level, label-free counts and ranks (S = S1 side, Q = pool side, both from _prep_side + _add_skel_and_idf):
    n_s1n = #S1 with the record's sorted name_norm key, n_s1a = #S1 with its address key (0 for empty keys),
    pct_w / pct_len = percentile rank of the IDF mass / #core tokens among the records of the SAME side of the frame."""
    kn = S.filter(pl.col("keyn") != "").group_by("keyn").agg(pl.len().cast(pl.Float32).alias("n_s1n"))
    ka = S.filter(pl.col("akey") != "").group_by("akey").agg(pl.len().cast(pl.Float32).alias("n_s1a"))

    def add(D):
        n = max(D.height, 1)
        return (D.join(kn, on="keyn", how="left", maintain_order="left").join(ka, on="akey", how="left", maintain_order="left")
                 .with_columns(pl.col("n_s1n").fill_null(0.0), pl.col("n_s1a").fill_null(0.0),
                               (pl.col("w").rank("average") / n).cast(pl.Float32).alias("pct_w"),
                               (pl.col("tok").list.len().rank("average") / n).cast(pl.Float32).alias("pct_len")))
    return add(S), add(Q)


def pair_features_v4(P: pl.DataFrame, base: pl.DataFrame, cfg: Optional[FeatureConfig] = None) -> pl.DataFrame:
    """All v4 PAIR features (groups a-f, + g when cfg.record_flags) for a joined pair frame.
    P    : one row per pair with the side columns of _prep_side/side_stats_v4 suffixed _q (S1) / _c (candidate):
           hn, mkey, anu, stw, cst, keyn, ng, ini_n, ini_c, ini_s, akey, n_s1n, n_s1a, pct_w, pct_len [, rf_*]
    base : same rows, v3 columns n_tset, a_tset, hn_exact (for base_n)
    Returns the columns HNREL_FEATS + STREET_FEATS + norm_eq, amb_s1n_c, amb_s1n_q, base_n + ACR_FEATS + COLOC_FEATS +
    PCT_FEATS [+ RECORD_FLAG_FEATS], Float32, no NaN / null, row-aligned with P. comp_n_* need the whole frame ->
    competition_v4 on the concatenated output."""
    cfg = cfg or FeatureConfig()
    w = cfg.workers
    hr = hn_relation(P["hn_q"], P["hn_c"], cfg.hn_max_nums, w)
    code = pl.col("hn_rel")
    C = HN_REL_CODES
    hr = hr.with_columns(
        code.is_in([C[k] for k in _HN_COPY]).cast(pl.Float32).alias("hn_r_copy"),
        code.is_in([C["digit_added"], C["digit_dropped"]]).cast(pl.Float32).alias("hn_r_indel"),
        (code == C["letter_change"]).cast(pl.Float32).alias("hn_r_letter"),
        code.is_in([C[k] for k in _HN_SMALL]).cast(pl.Float32).alias("hn_r_small"),
        code.is_in([C[k] for k in _HN_HARD]).cast(pl.Float32).alias("hn_r_hard"),
    )
    # street-only token sets: each side minus the OTHER side's city / state tokens too
    st = P.select(pl.col("stw_q").list.set_difference(pl.col("cst_c")).alias("a"),
                  pl.col("stw_c").list.set_difference(pl.col("cst_q")).alias("b"))
    st = st.with_columns(pl.col("a").list.set_intersection(pl.col("b")).list.len().cast(pl.Float32).alias("_in"),
                         pl.col("a").list.set_union(pl.col("b")).list.len().cast(pl.Float32).alias("_un"),
                         ((pl.col("a").list.len() > 0) & (pl.col("b").list.len() > 0)).alias("_both"))
    ts = _cp(st["a"].list.join(" ").to_list(), st["b"].list.join(" ").to_list(), fuzz.token_set_ratio, w) / 100
    both = st["_both"].to_numpy()
    stf = st.select(pl.col("_in").alias("st2_inter"),
                    pl.when(pl.col("_both")).then(pl.col("_in") / pl.max_horizontal(pl.col("_un"), pl.lit(1.0)))
                      .otherwise(-1.0).cast(pl.Float32).alias("st2_jacc"))
    stf = stf.with_columns(pl.Series("st2_tset", np.where(both, ts, -1.0).astype(np.float32)))
    ln = lambda c: pl.col(c).str.len_chars()   # noqa: E731
    ex = P.select(
        ((pl.col("keyn_q") == pl.col("keyn_c")) & (pl.col("keyn_q") != "")).cast(pl.Float32).alias("norm_eq"),
        pl.col("n_s1n_c").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_s1n_c"),
        pl.col("n_s1n_q").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_s1n_q"),
        (((ln("ng_c") >= 2) & ((pl.col("ng_c") == pl.col("ini_n_q")) | (pl.col("ng_c") == pl.col("ini_c_q"))
                                | (pl.col("ng_c") == pl.col("ini_s_q"))))
         | ((ln("ng_q") >= 2) & ((pl.col("ng_q") == pl.col("ini_n_c")) | (pl.col("ng_q") == pl.col("ini_c_c"))
                                  | (pl.col("ng_q") == pl.col("ini_s_c"))))).cast(pl.Float32).alias("acr_match"),
        (pl.col("n_s1a_q").fill_null(0.0) - 1).clip(0.0, None).log1p().cast(pl.Float32).alias("n_s1_akey_q"),
        (pl.col("n_s1a_c").fill_null(0.0)
         - ((pl.col("akey_c") == pl.col("akey_q")) & (pl.col("akey_q") != "")).cast(pl.Float32))
          .clip(0.0, None).log1p().cast(pl.Float32).alias("n_s1_akey_c"),
        pl.col("pct_w_q").fill_null(0.0).cast(pl.Float32).alias("pct_idf_s1"),
        pl.col("pct_w_c").fill_null(0.0).cast(pl.Float32).alias("pct_idf_cand"),
        pl.col("pct_len_q").fill_null(0.0).cast(pl.Float32).alias("pct_len_s1"),
        pl.col("pct_len_c").fill_null(0.0).cast(pl.Float32).alias("pct_len_cand"),
        _mset_expr().alias("hn_mset"),
    )
    bn = base.select(base_n_expr().alias("base_n"))
    cols = hr.get_columns() + ex.get_columns() + stf.get_columns() + bn.get_columns()
    if cfg.record_flags:
        cols += P.select([pl.col(f"rf_{c[len('cand_'):]}_c").fill_null(0.0).cast(pl.Float32).alias(c)
                          if f"rf_{c[len('cand_'):]}_c" in P.columns else pl.lit(0.0, pl.Float32).alias(c)
                          for c in RECORD_FLAG_FEATS]).get_columns()
    out = pl.DataFrame(cols)
    order = [c for c in V4_FEATS if c in out.columns] + (RECORD_FLAG_FEATS if cfg.record_flags else [])
    return out.select(order)


def competition_v4(df: pl.DataFrame) -> pl.DataFrame:
    """comp_n_best / comp_n_gap / comp_n_is_best from base_n over ALL S1 of the frame (df: cand, base_n, one row per
    pair). Tie-symmetric (no row-order dependence): is_best = base_n equals the candidate's max; gap = base_n - max of
    the OTHER claimants (0 if none)."""
    g = df.group_by("cand").agg(
        pl.col("base_n").max().alias("_t1"),
        (pl.col("base_n") == pl.col("base_n").max()).sum().alias("_n1"),
        pl.col("base_n").filter(pl.col("base_n") < pl.col("base_n").max()).max().alias("_t2"))
    x = df.select("cand", "base_n").join(g, on="cand", how="left", maintain_order="left")
    other = (pl.when(pl.col("base_n") < pl.col("_t1")).then(pl.col("_t1"))
               .when(pl.col("_n1") >= 2).then(pl.col("_t1"))
               .otherwise(pl.col("_t2").fill_null(0.0)))
    return x.select(pl.col("_t1").cast(pl.Float32).alias("comp_n_best"),
                    (pl.col("base_n") - other).cast(pl.Float32).alias("comp_n_gap"),
                    (pl.col("base_n") == pl.col("_t1")).cast(pl.Float32).alias("comp_n_is_best"))


def prepare_sides(s1: pl.DataFrame, pool: pl.DataFrame, cfg: Optional[FeatureConfig] = None):
    """Side tables exactly as compute_features builds them -> (S, Q, idf, max_idf). Reusable by add-on scripts that
    compute only new columns for an existing feature file (same record-level values as a full recompute)."""
    cfg = cfg or FeatureConfig()
    S = _prep_side(s1, cfg.record_flags)
    Q = _prep_side(pool, cfg.record_flags)
    # IDF over the pool's core tokens (unlabelled data only -> no leakage)
    df_tok = Q.select(pl.col("tok").alias("t")).explode("t").drop_nulls("t").group_by("t").len()
    n_docs = max(Q.height, 1)
    idf = df_tok.select("t", (((n_docs + 1) / (pl.col("len") + 1)).log() + 1).cast(pl.Float64).alias("idf"))
    max_idf = float(math.log(n_docs + 1) + 1)
    lut = _skeleton_lut(pl.concat([S.select(pl.col("tok").alias("t")).explode("t")["t"],
                                   Q.select(pl.col("tok").alias("t")).explode("t")["t"]]))
    S = _add_skel_and_idf(S, lut, idf, max_idf)
    Q = _add_skel_and_idf(Q, lut, idf, max_idf)
    # name-ambiguity counts per normalized name key (empty key -> 0)
    ks = S.filter(pl.col("key") != "").group_by("key").agg(pl.len().cast(pl.Float32).alias("n_s1k"))
    kp = Q.filter(pl.col("key") != "").group_by("key").agg(pl.len().cast(pl.Float32).alias("n_poolk"))
    S = S.join(ks, on="key", how="left").join(kp, on="key", how="left").with_columns(
        pl.col("n_s1k").fill_null(0.0), pl.col("n_poolk").fill_null(0.0))
    Q = Q.join(ks, on="key", how="left").join(kp, on="key", how="left").with_columns(
        pl.col("n_s1k").fill_null(0.0), pl.col("n_poolk").fill_null(0.0))
    S, Q = side_stats_v4(S, Q)
    return S, Q, idf, max_idf


def join_pairs(C: pl.DataFrame, Sq: pl.DataFrame, Qc: pl.DataFrame) -> pl.DataFrame:
    """Pair frame: C (s1, cand, ...) + suffixed side columns, row order = C order; missing strings -> ''."""
    P = (C.join(Sq, on="s1", how="left", maintain_order="left")
          .join(Qc, on="cand", how="left", maintain_order="left"))
    for c in P.columns:
        if P[c].dtype == pl.Utf8:
            P = P.with_columns(pl.col(c).fill_null(""))
    return P


def suffix_sides(S: pl.DataFrame, Q: pl.DataFrame, cols: Optional[List[str]] = None):
    """(Sq, Qc): side tables renamed for join_pairs (id -> s1 / cand, other columns -> <col>_q / <col>_c)."""
    keep = lambda D: D if cols is None else D.select(["id"] + [c for c in cols if c in D.columns])   # noqa: E731
    S, Q = keep(S), keep(Q)
    Sq = S.rename({c: f"{c}_q" for c in S.columns if c != "id"}).rename({"id": "s1"})
    Qc = Q.rename({c: f"{c}_c" for c in Q.columns if c != "id"}).rename({"id": "cand"})
    return Sq, Qc


# columns of the side tables pair_features_v4 reads (an add-on joins only these)
V4_SIDE_COLS = ["hn", "mkey", "anu", "stw", "cst", "keyn", "ng", "ini_n", "ini_c", "ini_s", "akey", "n_s1n", "n_s1a",
                "pct_w", "pct_len", "rf_dba", "rf_lower", "rf_upper", "rf_accent", "rf_junk", "rf_bracket", "rf_addr_null"]


# --------------------------------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------------------------------
def compute_features(cands: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame,
                     cfg: Optional[FeatureConfig] = None) -> pl.DataFrame:
    cfg = cfg or FeatureConfig()
    t0 = time.time()
    keep = ["s1", "cand"] + (["label"] if "label" in cands.columns else [])
    blk = [c for c in cands.columns if c.startswith("p_") or c in ("n_passes", "prio")]
    C = cands.select(keep + blk).with_row_index("_row")

    S, Q, idf, max_idf = prepare_sides(s1, pool, cfg)
    _log(cfg, f"sides ready: s1 {S.height:,} pool {Q.height:,} ({time.time() - t0:.1f}s)")
    Sq, Qc = suffix_sides(S, Q)

    # chunk by whole S1 groups
    C = C.sort("s1", maintain_order=True)
    sizes = C.group_by("s1", maintain_order=True).len()
    bounds, acc, start = [], 0, 0
    for n in sizes["len"].to_list():
        acc += n
        if acc - start >= cfg.chunk_pairs:
            bounds.append((start, acc)); start = acc
    if start < acc:
        bounds.append((start, acc))

    parts = []
    for i, (lo, hi) in enumerate(bounds):
        P = join_pairs(C.slice(lo, hi - lo), Sq, Qc)       # row order = cands order (ties)
        F = _pair_scores(P, cfg, idf, max_idf)
        F = F.hstack(_block_feats(P).get_columns())
        F = F.with_columns((0.5 * pl.col("c_tset") + 0.3 * pl.col("a_tset") + 0.2 * pl.col("hn_exact")).alias("base"))
        s1col = P["s1"]
        ctx = pl.DataFrame({"s1": s1col, "base": F["base"]}).with_columns(
            pl.col("base").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("ctx_rank"),
            (pl.col("base").max().over("s1") - pl.col("base")).cast(pl.Float32).alias("ctx_gap"),
            pl.len().over("s1").cast(pl.Float32).alias("ctx_ncand"))
        F = F.hstack(ctx.select("ctx_rank", "ctx_gap", "ctx_ncand").get_columns() + _group_feats(P, F, cfg).get_columns())
        F = F.hstack(P.select(pl.col("n_s1k_c").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_s1_c"),
                              pl.col("n_s1k_q").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_s1_q"),
                              pl.col("n_poolk_c").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_pool_c"),
                              pl.col("n_poolk_q").fill_null(0.0).log1p().cast(pl.Float32).alias("amb_pool_q")).get_columns())
        F = F.hstack(pair_features_v4(P, F, cfg).get_columns())
        parts.append(P.select(["_row"] + keep).hstack(F.get_columns()))
        del P
        _log(cfg, f"chunk {i + 1}/{len(bounds)}: {hi - lo:,} pairs ({time.time() - t0:.1f}s)")

    out = pl.concat(parts) if parts else pl.DataFrame()
    # competition features over the whole frame (exclusivity: a record belongs to at most one S1)
    out = out.with_columns(
        pl.col("base").max().over("cand").cast(pl.Float32).alias("comp_best"),
        pl.len().over("cand").cast(pl.Float32).alias("comp_nclaim"),
        pl.col("base").rank("ordinal", descending=True).over("cand").alias("_cr"),
    )
    second = (out.filter(pl.col("_cr") <= 2).group_by("cand")
                 .agg(pl.col("base").sort(descending=True).slice(1, 1).first().alias("_second")))
    out = out.join(second, on="cand", how="left").with_columns(
        (pl.col("_cr") == 1).cast(pl.Float32).alias("comp_is_best"),
        pl.when(pl.col("_cr") == 1).then(pl.col("base") - pl.col("_second").fill_null(0.0))
          .otherwise(pl.col("base") - pl.col("comp_best")).cast(pl.Float32).alias("comp_gap"),
    ).drop("_cr", "_second")
    out = out.hstack(competition_v4(out).get_columns())
    out = out.sort("_row").drop("_row")
    cols = feature_columns(cfg)
    out = out.select(keep + [pl.col(c).cast(pl.Float32).fill_nan(0.0).fill_null(0.0) for c in cols])
    _log(cfg, f"done: {out.height:,} pairs x {len(cols)} features in {time.time() - t0:.1f}s "
              f"({out.height / max(time.time() - t0, 1e-9):,.0f} pairs/s)")
    return out
