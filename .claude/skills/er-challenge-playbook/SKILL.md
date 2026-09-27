---
name: er-challenge-playbook
description: Use when starting, planning, resuming, or deciding the next step on the Amazon ML Challenge 2026 business entity resolution project (team Master Bolt) — matching Source 2/Source 3 business records to deduplicated Source 1, macro F0.5 scoring, candidate_pairs.tsv / matching_results.tsv, France test-only country, or the final submission zip.
---

# ER Challenge Playbook (Master Bolt)

## Overview
Build a reproducible pipeline that, for every Source 1 (S1) business, outputs the Source 2/3 (S2/S3) records of the same real business. Scored by **per-S1 F0.5, macro-averaged over all test S1 (singletons included)**. Core principle: **blocking sets the recall ceiling, the exclusivity-aware decision sets precision** — and everything must run on a 16 GB Windows laptop (or a rented AWS box) with only MIT/Apache models and no external lookups.

This skill is the map. Load the step skill before doing the step.

## Hard rules (violating any = rejection or disqualification)
| Rule | Consequence |
|---|---|
| Only provided data. No external DB/API/geocoder/LLM API (Groq, OpenAI…), no downloaded gazetteers | Disqualification. **See er-compliance-licensing** |
| Final model MIT or Apache-2.0 and ≤ 8B params (LightGBM = MIT) | Disqualification |
| Every test S1 exactly once; ids only S2-/S3- that exist; no dup ids in a list; TAB-separated, exact headers | Submission rejected |
| `country` is an open set: never hard-code/filter/one-hot {US, India}; France (15% of test S1) is unseen in train | Collapse on France |
| `candidate_pairs.tsv` = exactly the pairs the model scored; matches ⊆ candidates | Audit flag |
| Zip must regenerate outputs from `code/business_entity_resolution/` alone | Unreproducible → rejected in review |
| Never modify `student_resource/` (data, validator, blank template are read-only inputs) | Destroys originals |

## Facts that drive every design decision (measured on full train)
- Sizes: train S1 2.21M / S2 5.03M / S3 5.29M; test S1 1.73M (India 810k, US 663k, **France 259k**) / S2 4.89M / S3 5.08M.
- Matches per S1: mean **3.46**, only **5.58% singletons** → predicting empty for a matched S1 costs a full 1.0; recall matters despite F0.5.
- **Exclusivity:** every S2/S3 id belongs to **at most one** S1 (0 violations in 7.64M links). S2 and S3 each hold several duplicate records per entity.
- ~25% of S2/S3 records are unlinked distractors. Country **never** crosses (safe hard partition).
- India: ~24% of S2 and ~13% of S3 names are in Indic scripts (9 scripts); S1 is always Latin. PIN codes are absent (0.3%); US ZIPs are in only ~11% of addresses.
- Test pools are ~23% denser per S1 than train (more distractors or more matches) → watch precision.
- Full details and noise operators: `reference-data-facts.md` (this folder).

## Pipeline architecture
```
TSV ──io──▶ parquet cache ──normalize──▶ normalized frames (+ native-token dict learned from TRAIN links)
   ──blocking (per country; multi-pass union: keys + TF-IDF top-K in both directions, inside geo blocks)──▶ candidates
   ──features (pairwise + context + competition + group)──▶ LightGBM (GroupKFold by S1) ──▶ calibrated p
   ──decide (exclusivity: each S2/S3 → ≤1 S1; per-S1 expected-F0.5 selection)──▶ links
   ──outputs──▶ output/candidate_pairs.tsv + output/matching_results.tsv ──check + official validator──▶ zip
```

## Project layout (create exactly this; see CONTRACT in `reference-contract.md`)
```
AmazonMLChallenge/
├── student_resource/                    # READ-ONLY organiser files (dataset/, utils/validate_submission.py, template)
├── code/business_entity_resolution/     # deliverable: src/run_pipeline.py, src/ber/*.py, README.md, requirements.txt
├── output/                              # matching_results.tsv, candidate_pairs.tsv
├── work/                                # caches, candidates, features, models, logs, experiments.md (never zipped)
├── Documentation_template.md            # filled copy (copied from student_resource first)
└── .venv/                               # Python 3.10 venv (exists)
```
Run: `cd code/business_entity_resolution && python src/run_pipeline.py --data-dir ../../student_resource/dataset --work-dir ../../work --out-dir ../../output --stage all`

Reference modules live in the skill folders: copy them into `src/ber/` and adapt (they follow the contract):
| module | skill folder |
|---|---|
| `io.py`, `make_dev_slice.py` | er-data-loading |
| `normalize.py`, `normalize_frame.py`, `native_token_dict.example.tsv` | er-text-normalization |
| `blocking.py`, `measure_blocking.py` | er-blocking-candidates |
| `features.py` | er-pair-features |
| `model.py` | er-matcher-training |
| `metric.py`, `decide.py` | er-f05-decisions |
| `outputs.py`, `check_submission.py`, `package_submission.py` | er-submission-packaging |
| `memory_guard.py` | er-compute-and-aws |
| `error_report.py` | er-error-analysis |
| `audit_compliance.py` (tool, not part of `ber`) | er-compliance-licensing |
| `run_pipeline.py` (→ `src/`), `config.py`, `smoke_test.py` | er-challenge-playbook (this folder) |

`python .claude/skills/er-challenge-playbook/smoke_test.py` assembles `io, normalize, normalize_frame, blocking, features, model, decide, metric, outputs, config` into a temp `ber` package, builds `work/dev/mini` from the real data if missing (3,000 S1 per country), runs `run_pipeline.py --stage all`, then `check_submission.py` + the official validator, and prints PASS/FAIL. Run it after any change to a reference module. Every folder with `test_*.py` also has unit tests: `python -m pytest -q -p no:cacheprovider` inside it.

## Roadmap — do these in order, each with an exit gate
| # | Step | Load skill | Exit gate (write result to `work/experiments.md`) |
|---|---|---|---|
| 0 | Environment + layout + compliance check | er-compliance-licensing, er-compute-and-aws | venv works, `smoke_test.py` passes, no GPL/API deps |
| 1 | Parquet cache, GT long form, dev slices, validation folds | er-data-loading | row counts match `wc -l`−1 for all 7 files |
| 2 | Normalization + native-token dict (TRAIN links only) | er-text-normalization | tests pass; India native-name core TSR ≳ 99 on held-out links |
| 3 | Blocking on a geo dev slice, then full train/test | er-blocking-candidates | pair recall ≥ 97% at ≤ ~40 cands/S1 on dev slice; report per pass |
| 4 | Features | er-pair-features | no NaN/inf; no country encoding; chunked, < RAM budget |
| 5 | Matcher with GroupKFold OOF + leave-one-country-out | er-matcher-training | OOF AUC, LOCO gap recorded |
| 6 | Decision rule tuned on OOF macro F0.5 | er-f05-decisions | OOF macro F0.5 per country + oracle ceiling |
| 7 | Error analysis loop (repeat 3–6) | er-error-analysis | each change = one log line with Δ F0.5 |
| 8 | Test inference, outputs, validators, first leaderboard upload | er-submission-packaging | official validator PASS; predicted links/S1 on test vs OOF compared |
| 9 | Package zip + README + requirements + Documentation | er-submission-packaging | clean-venv reproduction on mini data; zip listing checked |

Start small: every step first on a **geographic dev slice** (e.g. one US state + one Indian state), then full scale. Full scale on the laptop requires sharding by country × state; the comfortable option is an AWS r6a.4xlarge (128 GB, ~$0.57/h in Mumbai) — **see er-compute-and-aws**.

## Status when these skills were written (2026-09-25) — start here
- Done & verified: all reference modules + unit tests; `smoke_test.py` PASS (full pipeline on mini data in 223 s; both validators PASS; 0 exclusivity violations; France gets candidates + predictions).
- Measured on the real **Karnataka** train slice (58,993 S1): blocking pair recall 0.990 (0.977 at cap 30); OOF AUC 0.99984; **OOF macro F0.5 0.9772** (soft exclusivity + expected-F, empty_bias 2) vs oracle 0.9931.
- Not yet measured: a US slice (e.g. Texas), leave-one-country-out, France behaviour (mini test showed France with ~7× the links/S1 of US/India on random pools — check for false merges on generic French names), the full-scale run (use AWS r6a.4xlarge).
- Next actions: step 1 (parquet cache on the real data) → US geo slice through `measure_blocking.py` + `eval_slice.py` → full train/test run → first upload.

## Working rules for Claude Code on this project
- Log every experiment in `work/experiments.md`: date, change, OOF macro F0.5 (overall / US / India), blocking recall, runtime, notes. Never claim an improvement without the number.
- Checkpoint every stage to parquet in `work/` so a crash or session end loses one stage, not the run.
- Watch memory: `psutil.Process().memory_info().rss`; free RAM on this laptop is often 1–4 GB. Stream, chunk, and normalize **distinct** strings only.
- Tune only on train OOF. The public leaderboard is a small subset of test — use it as a sanity check, not an optimisation target.
- Before any upload: run `check_submission.py` + the official validator (**see er-submission-packaging**).

## Common mistakes (from a baseline agent's first plan — full catalog in `reference-baseline-failures.md`)
| Mistake | Fix |
|---|---|
| Writing code/outputs/zip inside `student_resource/`, filling the template in place | Separate project tree; copy the template first |
| `pip freeze > requirements.txt` (ships GPL `unidecode`, AGPL `aksharamukha`, `groq`) | Hand-write pins from `src/` imports only |
| Reading TSVs with pandas defaults (quotes, "NULL"/"None" → NaN) | `er-data-loading` safe reader |
| Accent-stripping before transliteration (destroys Indic vowel signs) | Transliterate first, fold after |
| Name-only TF-IDF blocking (measured pair recall only 0.71 at top-40 on a Karnataka shard) | Union of key + name + address passes, both directions |
| Whole-country brute-force top-K (≈38 h per field on laptop) | Block by state/city + df-pruning |
| Global threshold ignoring exclusivity; random pair-level CV split | Exclusivity + per-S1 expected-F0.5; GroupKFold by S1 with full pools |
| One-hot `country` feature | Country only as a partition key; features language-agnostic |
| Using Groq / hosted LLM "8B" model | Forbidden (external service; Llama license) — local MIT/Apache only |
| Validator "skip candidate" by omitting `--candidate` (it still loads `output/candidate_pairs.tsv`) | Pass `--candidate NONE` |
