---
name: er-blocking-candidates
description: Use when generating candidate pairs (blocking) for the Master Bolt entity-resolution pipeline, when blocking recall or candidates-per-S1 must be measured, when true matches never reach the model, when top-K TF-IDF is too slow or runs out of memory on millions of S2/S3 records, when choosing the per-S1 candidate cap, or when writing candidate_pairs.tsv.
---

# Blocking / candidate generation

## Overview
Blocking sets the **recall ceiling**: a true S2/S3 record that is not a candidate can never be predicted. No single pass is enough. The measured answer is a **union of cheap exact-key passes plus TF-IDF char-3-gram top-K passes run inside geographic blocks in both directions**, then a per-S1 priority cap. Reverse direction (pool → S1) is strong because of exclusivity: each S2/S3 record's best S1 is usually its true one.

## When to use
- Building or changing `ber/blocking.py`, choosing K / caps, measuring recall, sharding for full scale, writing `candidate_pairs.tsv`.
- Not for scoring pairs (**See also:** er-pair-features) or choosing final matches (**See also:** er-f05-decisions).

## Quick reference
| Item | Value / command |
|---|---|
| Code | `blocking.py` (copy to `src/ber/blocking.py`): `BlockingConfig`, `generate_candidates(s1, pool, cfg, stats)`, `candidate_recall(cands, gt, s1_ids)`, `cap_candidates`, `geo_shards` |
| Measure on a slice | `python measure_blocking.py --slice IN_KA --bench-dir <slice dir> --out-dir <out> --cap 80 --caps 10,20,30,40,60,80,100,150` |
| Inputs | NORMALIZED frames (er-text-normalization columns); partition by `country` first — links never cross countries |
| Output | `s1, cand` + `p_key_<k>` flags, `p_<t>_score/_rank/_rrank`, `p_key_hit`, `n_passes`, `prio` |
| Default cap | `max_cands_per_s1=80` (98.5% pair recall on Karnataka); use `None` only to measure |
| Tests | `python -m pytest -q -p no:cacheprovider test_blocking.py` (12 pass) |

## Passes (defaults in `BlockingConfig`)
| Pass | What joins | Why (noise operator it survives) |
|---|---|---|
| `key_name` | sorted unique `name_core` tokens | token shuffles, legal-suffix moves |
| `key_tok` | each record's 3 rarest core tokens (len ≥ 3, df ≤ `key_cap`=30 country-wide, else ≤ 150 in-block) | truncation, extra words |
| `key_hn` | rare house number × rare address word | alias names ("Arcmira"), name typos |
| `key_skel` | consonant skeleton of core tokens | transliteration vowel loss (`devlprs` ↔ developers), typos |
| `key_compact` | glued name (spaces removed) | domain-form names `unifiedpinnacleesports.com` |
| `key_addr` | sorted unique address tokens | component reordering |
| `name_fwd/rev` | TF-IDF `name_core`, K=20 forward, 3 reverse | fuzzy names |
| `addr_fwd/rev` | TF-IDF `addr_norm`, 20 / 3 | same address, different name |
| `both_fwd/rev` | TF-IDF `name_core | addr_norm`, 20 / 3 | the strongest single pass |
TF-IDF = `HashingVectorizer(char_wb 3-grams, 2**20, alternate_sign=False)` + own IDF, `sparse_dot_topn` (Apache-2.0) with query-side df pruning (`max_df=0.01`, 9.5× faster), inside `state_key` blocks; S1 also query the unknown-state pool with K=5.

## Measured (train Karnataka slice: 58,993 S1 × 291k pool, 204,460 true links, 2026-09-25)
Union order and cumulative pair recall (no cap):
| after pass | alone recall | cum. recall | cum. cands/S1 |
|---|---|---|---|
| key_name | 0.728 | 0.728 | 7.9 |
| + key_tok | 0.475 | 0.843 | 26.5 |
| + key_hn | 0.768 | 0.965 | 154 |
| + key_skel / key_compact / key_addr | — | 0.976 | 156 |
| + name_fwd/rev | 0.544 / 0.522 | 0.982 | 179 |
| + addr_fwd/rev | 0.774 / 0.773 | 0.988 | 197 |
| + both_fwd/rev | 0.926 / 0.930 | **0.990** | 200 |
Union: 96.6% of S1 get all their links; oracle macro-F0.5 ceiling 0.997. Priority cap trade-off:
| cap | 10 | 20 | 30 | 40 | 60 | **80** | 100 | 150 |
|---|---|---|---|---|---|---|---|---|
| pair recall | 0.940 | 0.966 | 0.977 | 0.981 | 0.984 | **0.985** | 0.986 | 0.988 |
| oracle F0.5 | 0.981 | 0.990 | 0.993 | 0.994 | 0.995 | **0.996** | 0.996 | 0.996 |
Cost: 288 s blocking + 68 s normalization, peak RSS 1.63 GB (house-number pass spikes to 2.5 GB). Name-only TF-IDF is poor: top-20 forward = 0.54 recall. **Still to measure**: a US slice (e.g. Texas `US_TX`) and a France-like check on test (no labels: report cands/S1 only).

## Procedure
1. Normalize S1 and pool (**REQUIRED SUB-SKILL:** er-text-normalization); load GT long form (er-data-loading).
2. On a geographic dev slice, run `measure_blocking.py`; keep the per-pass table in `work/experiments.md`. Gate: pair recall ≥ 0.97 at the chosen cap.
3. Full scale: loop `for country in s1["country"].unique()` (never a hard-coded list) and inside it `geo_shards(...)` so one state block is in memory at a time; write each shard's candidates to `work/cands/<split>/<country>/<block>.parquet`, then concatenate. France (test only) has 3 regions and 15 cities, so region blocks are big: the key passes + `both_*` passes do most of the work there; check cands/S1 and runtime on France before the full run.
4. Train and test must use the **same** config (the model learns on the candidate distribution it will see).
5. `candidate_pairs.tsv` = exactly the pairs the model scores (after the cap), one row per test S1, empty list allowed (**See also:** er-submission-packaging).

## Scaling estimates
Whole-country brute force is impossible on the laptop (15k queries against half the 6.2M US pool took ~72 min; ≈38 h per field for all of train). State blocking cuts pair-work 13–29× (France only 3.9×); with df-pruning the full train+test blocking is ≈0.5–1.3 h on the laptop, 20–45 min on an r6a.4xlarge (**See also:** er-compute-and-aws). Karnataka needed 1.6 GB for 59k S1 — the biggest US states (TX/CA ~100k S1) need ~3–4 GB: run one block at a time and close other apps.

## Common mistakes
| Mistake | Fix |
|---|---|
| Name-only blocking (0.54–0.71 recall) | Union with key + address + combined passes |
| sklearn `TfidfVectorizer` on millions of strings (MemoryError at 350k) | HashingVectorizer + own IDF |
| Whole-country top-K | Geo blocks + unknown-state fallback block |
| Forgetting reverse direction | pool→S1 top-3 adds recall cheaply and encodes exclusivity |
| Different caps/config for train vs test | One `BlockingConfig`, saved with the model |
| Candidate file ≠ scored pairs (e.g. pre-cap list) | Write the post-cap pairs actually fed to the model |
| Hard-coding `["US","India"]` | Iterate over countries present; France must get candidates |
| Measuring recall on a random S1 sample against a random pool sample | Keep full pools per block; report the slice's own ceiling |
