---
name: er-text-normalization
description: Use when normalizing business names or addresses for the Master Bolt entity-resolution pipeline - Indic-script (Devanagari, Kannada, Tamil, Telugu...) S2/S3 names vs Latin S1 names, transliteration, injected accents, legal suffixes (LLC, Pvt Ltd, SARL), DBA / .com / @handle names, NULL tokens in addresses, state code vs name vs native script, France department vs region, unseen countries, the native-token dictionary, or slow / out-of-memory normalization.
---

# Text Normalization (names + addresses)

## Overview
Every noise operator of the data generator (native script, accents, suffix swaps, DBA aliases, domain forms, NULL tokens, state code/name/native, department/region) should become string equality after normalization. Core principle: **NFKC → transliterate (learned dictionary first, rules fallback) → fold accents → clean with country-keyed maps that fall back to generic rules**, applied to DISTINCT strings in a process pool. The native-token dictionary learned from TRAIN links is the largest single gain for India.

## When to use
- Playbook step 2 (before blocking / features), or when rebuilding the normalized parquet caches.
- India links with native-script names score low; a France (test-only) or unknown-country row behaves oddly; a new noise pattern shows up in error analysis.
- Normalization is slow or runs out of memory.

Not for: reading TSVs safely (**REQUIRED SUB-SKILL:** er-data-loading — literal "NULL"/"None" must stay strings), choosing blocking passes (**See also:** er-blocking-candidates), pairwise similarity features (**See also:** er-pair-features).

## Quick reference
| Task | Call (folder files are copied to `src/ber/`) |
|---|---|
| Normalize a frame (contract columns) | `add_normalized_columns(df, n_jobs=-1)` in `normalize_frame.py` (also reachable as `normalize.add_normalized_columns`) |
| Extra columns (city list, all numbers, street / other tokens) | `add_normalized_columns(df, extras=True)` |
| Same city_key on S1 and pool | `add_normalized_columns(s1, city_counts=city_counts(pool_n))` |
| Learn the native dictionary (TRAIN only) | `learn_dict_from_train(s1, pool, gt_long, "work/native_token_dict.tsv")` |
| Switch dictionary in this process | `use_native_dict(path)` (`None` = rules only) |
| Step-2 exit gate (held-out S1) | `python normalize_frame.py gate --s1 ... --pool ... --gt ...` |
| Dictionary coverage on unlabeled test | `python normalize_frame.py coverage --input test_source2.tsv --country India --native-dict ...` |
| Speed / distinct ratio of a file | `python normalize_frame.py bench --input <tsv/parquet> [--country India --limit 200000]` |
| Scalar, one pass | `normalize.name_variants(name, country)`, `normalize.address_all(addr, country)` |
| Pairwise glued / domain score | `normalize.glued_similarity(a, b, country)`, `normalize.split_concat(glued, other_tokens)` |
| Phonetic keys | `normalize.phonetic_key(tok)`, `normalize.skeleton_key(tok)` |

## Order of operations (never reorder)
1. `unicodedata.normalize("NFKC")` + quote/dash unification (fullwidth and math letters become ASCII).
2. `transliterate`: each Indic run (9 scripts) looked up in the learned dictionary per token, else rules (ISCII-parallel table, schwa deletion for northern scripts, ph→f). Latin accents are untouched here.
3. `fold`: NFKD, drop combining marks, lowercase (ASCII fast path).
4. Clean: DBA/aka/fka split (keep the part AFTER the marker; 1,007/1,007 measured), `[[junk]]`, `" | tail"`, phone numbers, domain → glued token, `@handle`, dotted abbreviations (`L.L.C.`→`llc`), `M/s`, `&`→`and`, possessive, leetspeak (`5ervices`), `lnc`→`inc`; tokens `[0-9a-z]+`.
5. Country-keyed maps (legal forms, street types, states), then the generic union.

Why this order: NFD+strip before transliteration deletes Indic vowel signs (`लिमिटेड` → `लमटड`); crude folding gives mean Jaro-Winkler 0.425 on native-script true links.

## Representations (columns of `add_normalized_columns`)
Measured 2026-09-25 on mini-train true links (42,094 links) vs random same-country S1 unless stated.

| Column | What it is | Use it for | Measured |
|---|---|---|---|
| `name_norm` | cleaned name, legal forms canonical (`pvt`, `ltd`, `llc`, `sarl`) but kept | legal-form agreement, TF-IDF text | exact equal: US 38.0%, India 47.9% |
| `name_core` | norm minus legal forms / honorifics / fillers / leading French articles; never empty | main name field (blocking, JW/TSR) | exact equal US 30.5%→61.5%, India 18.5%→71.2% (crude→core) |
| `name_key` | sorted unique core tokens | exact-key blocking (token shuffles) | exact US 64.7%, India 72.1%; key precision 97.1% / 95.7% |
| `name_compact` | core without spaces | `.com` / `@handle` / glued names | glued equal-or-substring US 75.9%, India 81.6% |
| `skel` | `skeleton_key` of each core token (vowel-less phonetic) | safety net for native words the dictionary lacks | rules-only native TSR 77.5 → 92.2 |
| `script`, `name_is_domain` | name script (latin/indic/other); domain or @handle form | routing flags / features (never the country) | |
| `addr_norm` | NULL tokens gone, street types / states / numbers canonical | TF-IDF, token Jaccard | Jaccard US 0.556→0.795, India 0.695→0.761 |
| `house_nums` | house numbers + `hno`/`dno`/`plot`/`no` numbers (no unit/floor) | number agreement (compare base digits) | India 54.0%→76.9% vs house-only; US 80.7%; random 1-3% |
| `state_key` | US 2-letter code, India full name, France region | hard-ish geo block | equal 95.1% (crude last component 37.6% US / 30.0% India) |
| `city_key` | city candidate named by most distinct addresses of the country | soft geo signal only | equal India 80.8% (last candidate 55.3%), US 80.1% |
| extras: `city_cands`, `all_nums`, `street_key`, `addr_other`, `name_distinct` | all city candidates; every number; US/France street words; India bag of words; core minus center/services/partners | India: any-city overlap 94.8%, `all_nums` base overlap 81.4%, `addr_other` Jaccard AUC 0.976 | |

`house_nums` and `city_key` numbers were measured by this skill's author (mini train, same protocol); the rest come from `reference-normalization-research.md`.

## Countries: keyed maps + generic fallback
`country_key(country)` → `us` | `india` | `france` | `generic`. Any other label, `None` or `""` gets `generic` = union of the three maps minus ambiguous short forms (`dr`, `ste`, `ch`, `mg`, ...); bare 2-letter state codes are mapped only inside a known country (IN = Indiana vs India, FL = Florida vs floor). France (15% of test S1, absent from train) has its own legal forms (SARL, SAS, SASU, EURL, SCI, EI, Cie, Ets, & Fils/Frères), street types (R./Rue, AV., BD, Imp., Allée), `N°`/`bis`/`ter`, department→region and city→department→region maps. To support a new country, add a key to the maps; never filter rows by country or one-hot it.

## The native-token dictionary (a learned artifact)
The generator spells each English token the same way per script, so 162,174 distinct (S1 name, native name) train pairs contain only 1,347 native tokens: a token dictionary inverts it almost exactly.

| Representation (3,852 held-out native-script true links, mini train) | TSR | JW |
|---|---|---|
| crude fold, no transliteration | 14.8 | 0.425 |
| anyascii (ISC) | 74.3 | 0.840 |
| rule transliteration (ours) | 77.5 | 0.865 |
| rules + skeleton key | 92.2 | 0.956 |
| **dictionary + rules fallback** | **99.18** | **0.9945** |

End to end, `name_core` exact equality on these links goes from 1.4% to 95.3%. Coverage on the unlabeled test India names (867,245 rows): **96.4% of token occurrences**, 88.7% of types. The 171 missing types are new shop words (stores, traders, jewellers, bakery, motors, ...), so keep the rules fallback and soft token matching in the matcher (unseen token vs the true S1: phonetic-key JW 0.864, vs a random S1 0.616). Hard snapping to a large vocabulary is wrong 58% of the time; `enable_snapping(counts, min_count=500, thr=0.7)` is 62.7% right / 3.1% wrong and stays off by default.

Regenerate it inside the pipeline (prepare stage) from train links; the committed `native_token_dict.example.tsv` is for tests only, and nobody hand-edits it:
```python
import polars as pl
from ber import normalize_frame as NF        # flat layout: import normalize_frame as NF

def normalize_split(tr, te, work, val_s1=None):
    # TRAIN links only. When scoring a validation fold, hold its S1 out of the dictionary too.
    NF.learn_dict_from_train(tr["s1"], pl.concat([tr["s2"], tr["s3"]]), tr["gt"],
                             f"{work}/native_token_dict.tsv", exclude_s1=val_s1)
    out = {}
    for name, split in (("train", tr), ("test", te)):
        pool = NF.add_normalized_columns(pl.concat([split["s2"], split["s3"]]), extras=True)
        s1 = NF.add_normalized_columns(split["s1"], extras=True, city_counts=NF.city_counts(pool))
        out[name] = (s1, pool)                # write each to work/*.parquet right away
    return out

if __name__ == "__main__":                    # spawn workers re-import __main__ on Windows
    ...
```
The workers receive the dictionary loaded in the parent (initializer), so load or learn it before calling `add_normalized_columns`. Full train: 53-64 s, under 1 GB (measured). Record in the Documentation that the dictionary is derived from the provided train links only.

## Running it at scale
- Distinct values, not rows: `unique()` per (string, country), map, join back (`maintain_order="left"`). On S2/S3 the saving is modest: distinct names/addresses are 94.3% / 95.8% of 200k real India S2 rows, 96.5% / 87.0% on mini train S2.
- Speed (i5-1235U laptop, CPU 77-91% busy with other agents, so treat as upper bounds): mini files 300-600 µs/row in-process; 200k real India S2 rows: 71 s with the pool (357 µs/row wall) (single-process wall time was not measured; the pool only pays off above ~20k distinct strings, see `_min_parallel`). Scalar costs: `name_variants` 50-80 µs, `address_all` 150-240 µs.
- Pool: `n_jobs=-1` → min(8, cpu−2) spawn workers, capped at free RAM / 150 MB; frames under 20k distinct values run in-process (spawn start-up costs more than it saves). One pool serves both passes; at most 2×workers chunks are in flight.
- Full data (≈24M rows) is about 2 CPU-hours: run per country partition, write parquet after each, or rent a box (**See also:** er-compute-and-aws).

## Verify
```
cd C:/Users/harsh/Downloads/AmazonMLChallenge/.claude/skills/er-text-normalization
set PYTHONDONTWRITEBYTECODE=1 & set PYTHONIOENCODING=utf-8
..\..\..\.venv\Scripts\python -m pytest -q -p no:cacheprovider test_normalize.py test_normalize_frame.py
```
Expected: 38 + 15 passed (the two mini-data tests skip when `ER_MINI_DIR` has no mini train). Exit gate for playbook step 2: `gate` on full train reports dictionary `name_core` TSR ≳ 99 on held-out native-script links (measured on 3,852 held-out mini native-script links: core TSR 99.7, JW 0.996, exact core match 95.3% with the dictionary vs TSR 63.5 without it).

## Common mistakes
| Mistake | Fix |
|---|---|
| Accent stripping (NFD + drop combining marks) before transliteration | Transliterate first, `fold` after; `fold` also keeps Indic marks as a guard |
| `unidecode` (GPL-2.0+), `text-unidecode`, `aksharamukha` (AGPL-3.0) in `src/` or requirements | stdlib rules in `normalize.py` (or `anyascii`, ISC). **See also:** er-compliance-licensing |
| Hand-rolled normalizer (baseline plan): kept DBA alias + name together, phone digits, `[[FFSAHIEGX]]`, `ms` from M/s, `shri`, `praivet`, emptied "Private Limited Company" | Use `normalize.py`; core falls back to norm, never empty |
| Snapping unknown native tokens to a big vocabulary (58% wrong: லைஃப்→"latif") | Dictionary + rules fallback + soft matching in features |
| Shipping or hand-editing `native_token_dict.example.tsv` | Regenerate with `learn_dict_from_train` in the pipeline |
| Learning the dictionary on test, or on the validation fold you score | TRAIN links only; `exclude_s1=val_s1` when measuring |
| Custom multiprocessing pool: children start with an EMPTY dictionary | Use `add_normalized_columns`, or replicate `_worker_init(_nlp_state())` |
| `MG` → `marg` (MG Road = Mahatma Gandhi Road) | `mg` is left alone |
| Taking the FIRST state-like component ("Washington, DC" → WA) | LAST one is the state; earlier ones become city candidates |
| Comparing France S1 region with S2/S3 department ("Hauts-de-France" vs "Nord") | `state_key` maps department and known city → region |
| Using `city_key` as a hard block | 80% agreement only; block on `state_key`, use `city_key` / `city_cands` as soft signals |
| `house_base`-style exact house number for India (54%) | `house_nums` (76.9%) or `all_nums` base overlap (81.4%) |
| Calling `normalize_name`, `name_core`, `name_key` separately on every row (732 µs/name) | `name_variants` / `address_all`, distinct values only |
| `"po box"` / `"c"` (Suite C) / `"dl"` as city (fixed in normalize.py 2026-09-25) | Keep the fixed `address_parts`; its regression tests are in `test_normalize_frame.py` |

**See also:** er-error-analysis (adding a new noise rule: add a failing test in `test_normalize.py` first), `reference-normalization-research.md` (full measurements, per-script numbers, all maps).
