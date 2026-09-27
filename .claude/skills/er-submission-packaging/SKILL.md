---
name: er-submission-packaging
description: Use when writing matching_results.tsv or candidate_pairs.tsv for the Master Bolt entity-resolution challenge, validating outputs before a leaderboard upload, the official validator fails or runs out of memory, building Master_Bolt_submission.zip, writing the code README or requirements.txt, or filling Documentation_template.md.
---

# Outputs, validation and the submission zip

## Overview
A correct model is worth nothing if the file is rejected. Write both TSVs with one writer (`outputs.py`), prove them with a **streaming checker plus the official validator**, then build the zip from an **allowlist** after gates that catch what gets a package rejected in review (GPL/API deps, stale outputs, unfilled template, caches).

## When to use
- Every time outputs are (re)generated, before every upload, and when assembling the final package.
- Not for choosing which ids to output (**See also:** er-f05-decisions).

## Quick reference
| Task | Command (absolute paths; run from project root) |
|---|---|
| Write both files | `from ber.outputs import read_s1_order, write_submission; write_submission(read_s1_order(test_s1_tsv), links, scored, out_dir)` |
| Streaming check (both files, subset rule, exclusivity) | `python .claude/skills/er-submission-packaging/check_submission.py --matching <abs>/output/matching_results.tsv --candidate <abs>/output/candidate_pairs.tsv --test-dir <abs>/student_resource/dataset/test --official` |
| Official validator, scored file only | `python student_resource/utils/validate_submission.py --matching <abs>/output/matching_results.tsv --candidate NONE --test-dir <abs>/student_resource/dataset/test --check-ids` |
| Zip dry run / build | `python .claude/skills/er-submission-packaging/package_submission.py --project-root <abs root> --dry-run`, then without `--dry-run` → `submission/Master_Bolt_submission.zip` |
| Templates | `README_template.md`, `requirements_template.txt`, `documentation_guide.md` (this folder) |
| Tests | `python -m pytest -q -p no:cacheprovider test_outputs.py` (65 pass; 23 matching + 10 candidate cases agree with the official validator) |

## Output spec (both files)
- Header exactly `source1_entity_id<TAB>matched_entity_ids` / `source1_entity_id<TAB>candidate_entity_ids`.
- One row per test S1, **in test_source1.tsv order**, every S1 exactly once (1,732,544 rows); empty list field when none.
- List: unique S2-/S3- ids that exist in test, sorted, joined by `,` with **no spaces**.
- UTF-8 **without BOM**, LF newlines, no quoting.
- `candidate_pairs.tsv` = exactly the pairs the model scored (post-cap); matches ⊆ candidates (`write_submission` raises otherwise).

## Validation procedure (before every upload)
1. `check_submission.py ... --official` — stdlib-only streaming re-implementation of every official rule for both files + strict gates the official validator lets through (CRLF, header bytes, blank lines, whitespace in ids) + subset rule (lockstep merge, O(1) memory) + exclusivity diagnostic (an id under two S1 = decision bug) + stats (ids/row histogram, empty rows, per country). Exit 0 PASS, 1 FAIL, 3 = disagrees with the official validator.
2. The official validator on the scored file with `--candidate NONE`. **Omitting `--candidate` does not skip it**: the validator defaults to `output/candidate_pairs.tsv` relative to the cwd and loads it. On the full test set the official candidate check holds tens of millions of ids in Python sets — too much for the laptop; use step 1 for the candidate file.
3. Sanity: predicted links per S1 and empty-rate on test vs OOF, per country (France included) — big gaps mean a bug or a shift (**See also:** er-error-analysis).
4. Upload `matching_results.tsv` only; record the score in `work/experiments.md`. The public board is a subset — never tune to it.

## Final package (Master_Bolt_submission.zip)
```
output/matching_results.tsv   output/candidate_pairs.tsv
code/business_entity_resolution/src/**/*.py   README.md   requirements.txt
Documentation_template.md            (filled copy, zip root)
```
`package_submission.py` gates (any error → no zip): G1 required files; G2 forbidden strings in code/requirements (groq, openai, anthropic, unidecode, aksharamukha, requests.get/post, urllib.request, http(s)://); G3 requirements pinned `name==version`, no GPL/AGPL/API clients, every third-party import pinned; G4 documentation differs from the blank template and has no placeholders; G5 outputs newer than all `src/*.py` (else stale); G6 outputs pass `check_submission.py`; G7 nothing written inside `student_resource/`. Then it re-opens the zip, CRC-checks it and prints the listing. Caches (.npy/.parquet/.pkl/.log/model binaries) are excluded unless `--extra ... --allow-binary` and documented.
- requirements.txt: start from `requirements_template.txt`; keep only what `src/` imports.
- README.md: start from `README_template.md`; fill runtimes, RAM, machine, seeds.
- Documentation: follow `documentation_guide.md`; license table from `audit_compliance.py --report` (**REQUIRED SUB-SKILL:** er-compliance-licensing).
- Reproduction check: fresh venv → `pip install -r requirements.txt` → `run_pipeline.py --stage all` on the mini data (`er-challenge-playbook/smoke_test.py`) must reproduce outputs that pass both validators.

## Common mistakes
| Mistake | Fix |
|---|---|
| UTF-8 BOM (PowerShell `Out-File`/`>`, pandas `utf-8-sig`) → official FAIL `'﻿source1_entity_id'` | `outputs.write_id_lists` (utf-8, `newline=""` semantics) |
| pandas `to_csv` on Windows writes CRLF by default | write LF explicitly |
| `", "` separator → `IDs without an S2-/S3- prefix` | `","` only |
| Rows not in test_source1 order / missing France rows | iterate `read_s1_order(test_source1.tsv)` |
| Writing outputs or the zip inside `student_resource/`; editing the template in place | project-root `output/`, `submission/`; copy the template |
| `pip freeze > requirements.txt` (ships GPL unidecode, AGPL aksharamukha, groq) | hand-written pins |
| Candidate file = pre-cap blocking output | the exact scored pairs |
| Zipping stale outputs after a code change | G5 gate; regenerate |
