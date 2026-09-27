---
name: er-pair-features
description: Use when computing features for (S1, candidate) pairs in the Master Bolt entity-resolution pipeline — string similarity (rapidfuzz, Jaro-Winkler, token set, TF-IDF/IDF overlap), house-number and address features, blocking-pass features, per-S1 context, exclusivity competition or duplicate-group features — or when feature computation is slow, runs out of memory, produces NaN, or leaks labels or country.
---

# Pair features

## Overview
The matcher only sees what the features encode. Encode each **noise operator** as a feature family, add **context** (how this candidate ranks among the S1's candidates), **competition** (how strongly other S1 claim the same record — exclusivity), and **group** similarity (true candidates resemble each other because S2/S3 hold duplicates of one entity). Never encode the country value.

## When to use
- Building/changing `src/ber/features.py`, adding a feature, chasing a feature bug, sizing feature memory.
- Not for candidate generation (**See also:** er-blocking-candidates) or training (**See also:** er-matcher-training).

## Quick reference
| Item | Value |
|---|---|
| Code | `features.py` → `src/ber/features.py`: `FEATURE_COLUMNS` (68, fixed order), `FeatureConfig`, `compute_features(cands, s1, pool, cfg)` |
| Inputs | candidates from blocking (`s1, cand [,label]` + `p_*` pass columns) and NORMALIZED s1/pool frames (er-text-normalization columns) |
| Output | `s1, cand [,label]` + 68 float32 features, same row order as `cands`, no NaN/inf |
| Speed (measured) | Karnataka slice, 1.77M pairs × 68 features in **29.5 s (~60k pairs/s)**, laptop, `chunk_pairs=600_000` |
| Memory | ≈ 68 × 4 B + ~250 B strings per pair in a chunk → 2M-pair chunk ≈ 0.8 GB; output 1.77M pairs ≈ 0.5 GB |
| Tests | `python -m pytest -q -p no:cacheprovider test_features.py` |

## Feature families (→ noise operator)
| Family | Features | Why |
|---|---|---|
| name (norm/core) | `n_ratio n_tset n_tsort n_partial n_jw`, `c_ratio c_tset c_partial c_jw c_lev` | typos, shuffles (token_set/sort), truncation (partial), prefixes |
| name keys | `k_eq` (sorted tokens), `g_ratio g_contain` (glued), `sk_ratio` (consonant skeleton) | shuffles, domain-form names, transliteration vowel loss |
| token weights | `tok_jacc tok_inter idf_overlap idf_contain idf_s1 idf_cand` (IDF from the pool) | generic words ("group", "private", "club") must weigh little; `idf_s1` low = generic S1 name |
| name shape | `len_s1 len_cand len_ratio legal_agree legal_conflict` | legal-suffix noise vs genuinely different legal forms |
| address | `a_ratio a_tset a_partial a_empty_any`, `hn_both hn_exact hn_base hn_diff`, `st_jacc st_inter`, `city_eq state_eq` (−1 = unknown) | reordering, 1804→1804A / 3412→3411 noise, city typos, missing components |
| record | `cand_is_s3 cand_nonlatin domain_any` | source style differences, native-script names |
| blocking | `p_key_*` (6), `p_{name,addr,both}_{score,rank,rrank}`, `n_passes p_key_hit prio` (missing → 0 / rank 99) | how/where the pair was retrieved |
| context | `base` (=0.5·c_tset+0.3·a_tset+0.2·hn_exact), `ctx_rank ctx_gap ctx_ncand` | a candidate far below the S1's best is rarely true |
| competition | `comp_best comp_is_best comp_gap comp_nclaim` | exclusivity: a record claimed more strongly by another S1 is rarely this S1's |
| group | `grp_max_sim grp_n_sim` (vs the S1's top-5 other candidates) | S2/S3 duplicates of the same entity look alike |

## Procedure
1. Compute per country (or per geo shard **plus** its fallback pool): the competition features see every S1 in the frame, so never feed a random S1 sample — use complete blocks.
2. Train and test candidates go through the **same** `compute_features` and `FeatureConfig`.
3. Save to `work/feats/<split>_<country>.parquet`; check `out.select(FEATURE_COLUMNS).to_numpy()` is finite.
4. After training, read feature gain (er-matcher-training report) and drop families with ~0 gain only if OOF macro F0.5 does not drop.

## Common mistakes
| Mistake | Fix |
|---|---|
| Python loops / dict-of-strings over 40M pairs (hours, ~10 GB) | `rapidfuzz.process.cpdist(workers=-1)` on chunk lists; polars list ops |
| Chunking by rows, splitting an S1's candidates across chunks | chunks are whole S1 groups (context/group features need all candidates) |
| Competition features computed on a sampled or sharded frame without the competitors | whole country / whole geo block + fallback pool |
| NaN from empty strings, missing ranks | explicit neutral defaults (0, −1, rank 99) — `compute_features` fills them |
| A `country` one-hot or country-specific feature | forbidden: France is unseen; keep features language-agnostic |
| Features derived from labels (e.g. "is in GT", counts of true links) | only data available at test time; IDF from the unlabeled pool is fine |
| Different feature code/config for test | one module, one config, saved with the model |
