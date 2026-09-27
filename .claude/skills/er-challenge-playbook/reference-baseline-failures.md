> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# R7 — Baseline failure catalog (critic review of the naive R6 plan)

Amazon ML Challenge 2026, Business Entity Resolution, team Master Bolt. Written 2026-09-25.
Reviewed artefact: `research/R6_baseline_plan.md` (598 lines, "first draft by an unguided coding agent").
Checked against: `EDA_FACTS.md` (measured on the full data), `student_resource/README.md`,
`student_resource/utils/validate_submission.py`, `student_resource/Documentation_template.md`, and measured outputs
of the parallel research runs (R1/R2/R5/R8 logs under `research/`).

STATUS: IN PROGRESS. The report is written incrementally; sections marked PENDING are not finished yet.

Everything tagged **[measured]** was run by me on this laptop for this report. The scripts and raw outputs are in
`research/r7/`. Items tagged **[cited]** quote a number that another research run measured (file named). Items tagged
**[reasoned]** follow from the rules or the code and were not run.

## Severity scale used for ranking
| level | meaning |
|---|---|
| **S0** | disqualification, rejected submission, or an unreproducible package |
| **S1** | large score loss: more than 0.01 macro F0.5 overall, or a collapse on France (15% of test S1) |
| **S2** | moderate loss (about 0.002-0.01), or a stage that cannot run on this laptop as specified (OOM, or days of compute) |
| **S3** | hygiene: small loss, time waste, fragility |

(sections follow)

## 1. What I verified, and how

All scripts are under `research/r7/`. Every python process was kept under about 1 GB RSS (DuckDB `memory_limit=900MB`, spill directory `r7/tmp`).

| script | what it checks | raw output |
|---|---|---|
| `plan_norm_check.py`, `plan_norm_check2.py` | runs the plan's `norm_name()` **verbatim** on EDA examples and on real S2/S3 rows; counts DBA / pipe / `aka` cases; times it | `plan_norm_check.out`, `plan_norm_check2.out`, `plan_norm_check.json` |
| `k1_check.py` | the plan's K1 key ("2 rarest core-name tokens, df over S1∪S2∪S3, both sides, cap 3000 targets/key") on the **full** train-US split (recall against GT) and the full test-France split (volume only) | `k1_train_US.out`, `k1_test_France.out` |
| `valtest/mk.py` + the official validator | BOM header, `", "` separator, CRLF, pandas' default line ending, and whether omitting `--candidate` really skips the candidate check | shown inline in §2 |
| `tsr_hash_check.out` | rapidfuzz `token_set_ratio` subset behaviour; polars `Expr.hash` stability note | `tsr_hash_check.out` |
| `val_mem.py` | peak RSS of the official validator and of the plan's dict-of-lists writer on a synthetic candidate file, extrapolated to 1.73 M S1 | `val_mem.out` (PENDING) |
| `hn_check.py` | how often the S1 house number reappears verbatim in the true target's address (the plan's K4 key needs an exact match) | `hn_check.out` (PENDING) |

## 2. The catalog, ranked (S0 first, then by expected score impact)

Each entry gives: **(a) Plan did** (verbatim quote from R6), **(b) Why it is wrong here** (the fact, with source), and
**(c) Correct practice**. The `skills:` line says which SKILL.md files should carry it in "Common mistakes".

---

### F01 [S0] It writes code, outputs, zip and the filled doc inside `student_resource/`, and edits the template in place
skills: `packaging-submission`, `project-conventions`
- **(a) Plan did:**
  - It sets `root = pathlib.Path(r"C:\Users\harsh\Downloads\AmazonMLChallenge\student_resource")`.
  - It then calls `z.write(root / f, f)` for `"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md"`,
    with `code = root / "code" / "business_entity_resolution"`, and writes the zip to `root / "Master_Bolt_submission.zip"`.
  - It says: "Fill in `Documentation_template.md` from the logged EDA ...".
  - Relative paths such as `"dataset/train/train_ground_truth.tsv"` and `write_id_lists("output/candidate_pairs.tsv", ...)` only resolve when the
    working directory is `student_resource/`.
- **(b) Why wrong:**
  - This project's hard rule is *do not modify anything under `student_resource/`*. That folder holds the official data, the validator and the
    blank template. The plan puts `code/`, `output/`, the zip and a modified `Documentation_template.md` there.
  - Filling the template in place destroys the blank original.
  - The zip loop copies whatever is in the folder at zip time, so a stale output ships too.
- **(c) Correct practice:**
  - Keep a separate project tree:
    - `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution\{src,README.md,requirements.txt}`
    - `...\AmazonMLChallenge\output\`
  - Build the package in a staging folder (e.g. `...\AmazonMLChallenge\submission\`) from an **explicit allowlist**.
  - Treat `student_resource/dataset` and `student_resource/utils` as **read-only inputs**, passed as CLI flags (`--data-dir`, `--out-dir`).
  - Copy the template to staging before filling it.
  - After zipping, list the archive and assert both its structure and its size:
    ```python
    import zipfile
    for i in zipfile.ZipFile("Master_Bolt_submission.zip").infolist():
        print(i.filename, i.file_size)
    ```
    The expected contents are:
    - `output/matching_results.tsv`
    - `output/candidate_pairs.tsv`
    - `code/business_entity_resolution/src/...`
    - `code/business_entity_resolution/README.md`
    - `code/business_entity_resolution/requirements.txt`
    - the filled `Documentation_template.md` at the root.
  - The plan's exclusion rule (`not p.suffix == ".parquet"`) is a denylist, so `.npy`, `.duckdb`, `.pkl` and `.log` caches in the code folder would
    be zipped. The R8 blocking experiment alone left a **4.0 GB** vector cache (`research/R8_blocking_experiment.md` §1).
    Use an allowlist instead: `*.py`, `README.md`, `requirements.txt`, and small model files that are needed and documented.

### F02 [S0] `pip freeze > requirements.txt` from the shared venv pins GPL/AGPL packages and 77 unrelated pins
skills: `packaging-submission`, `environment-setup`
- **(a) Plan did:** `pip freeze > code\business_entity_resolution\requirements.txt   # pin exactly what ran`. It also lists `groq 0.25` among the
  installed tools.
- **(b) Why wrong:**
  - **[measured]** `.venv\Scripts\python -m pip freeze` returns **77 lines** today, including:
    - **`aksharamukha==2.3` (GNU AGPL-3.0)**
    - **`Unidecode==1.4.0` (GPL-2.0+)**

    Both were installed by the transliteration benchmarks (R2). `duckdb`, `indic_transliteration`, `sparse-dot-topn` and
    `polars-runtime-32` appear too, whether or not the pipeline imports them.
  - The README says the zip is used "to reproduce your results, audit your blocking, and check the fair-play and model-license rules".
    A requirements file that pins AGPL/GPL packages or an LLM API client invites exactly the audit questions the team wants to avoid.
  - The plan chose anyascii *because* Unidecode is GPL, yet its freeze would still ship Unidecode.
  - A freeze of a shared, mutating venv (other agents install into it concurrently) is not a faithful record of what ran.
- **(c) Correct practice:**
  - Write `requirements.txt` by hand from the imports in `src/` only, with the exact versions installed in the venv, e.g.
    ```
    polars==1.44.2
    pyarrow==25.0.1
    numpy==2.2.6
    rapidfuzz==3.14.5
    anyascii==0.3.3
    lightgbm==4.7.0
    scikit-learn==1.7.2
    ```
    Add `duckdb==1.5.5` or `sparse-dot-topn==1.2.0` only if `src/` imports them.
  - Verify from a clean venv:
    `python -m venv C:\tmp\cleanvenv; C:\tmp\cleanvenv\Scripts\python -m pip install -r requirements.txt`.
    Then run the whole pipeline on `scratchpad/mini/` from it.
  - Add two gates. Both must return nothing:
    - `grep -iE "unidecode|aksharamukha|groq|openai|anthropic|text-unidecode" requirements.txt`
    - `grep -rniE "groq|openai|anthropic|requests\.(get|post)|urllib\.request|http[s]?://" src/`
  - Core-stack licenses, from `research/pypi_licenses.txt` and `research/github_licenses.txt`:
    - LightGBM, polars, rapidfuzz, duckdb and jellyfish: MIT
    - anyascii: ISC
    - pyarrow: Apache-2.0
    - numpy: BSD
    - scikit-learn: BSD-3-Clause

### F03 [S0 if used] The phase-3 model suggestions include a non-Apache model and misdescribe the encoder
skills: `models-and-licenses`
- **(a) Plan did:**
  - "A **local** Apache-2.0 model (e.g. a Qwen2.5 / Qwen3 1.5–7B instruct) or a small Apache-2.0 cross-encoder
    (`paraphrase-multilingual-MiniLM-L12-v2`, which handles Indic scripts natively)".
  - "The '8B model' on Groq is almost certainly `llama-3.1-8b-instant` ... Confirm the exact model ID with the user".
- **(b) Why wrong:**
  - **Qwen2.5-3B-Instruct is not Apache-2.0.** Its HF license tag is `other` / `qwen-research` (`research/hf_license_audit.json`). The 0.5B, 1.5B
    and 7B siblings are Apache-2.0, but "Qwen2.5 1.5–7B" includes the 3B. Rule 5: "Final model should be a MIT/Apache 2.0 License model and up to
    8 Billion parameters."
  - **Size edge cases.** Anything named "8B" needs its exact parameter count checked:
    - `Qwen/Qwen3-8B` has **8,190,735,360** parameters (same audit), over 8.0 B on a strict reading.
    - Llama-3.1-8B is 8.03 B *and* uses the Llama Community License.
  - **The encoder is misdescribed.**
    - `paraphrase-multilingual-MiniLM-L12-v2` is a **bi-encoder**, not a cross-encoder.
    - Its card lists only **gu, hi, mr, ur** among Indic languages. Tamil, Telugu, Kannada, Bengali and Malayalam all occur in S2/S3
      (`EDA_FACTS.md` scripts table).
    - `intfloat/multilingual-e5-small` (MIT, 117.7 M) lists 14 Indic languages.
  - The Groq question needs no user confirmation:
    - No Groq-hosted model is both at most 8 B and MIT/Apache.
    - `llama-3.1-8b-instant` left the free and Developer tiers on 2026-08-16 (`research/R3_models_licenses_groq.md`).
    - Above all, sending records to a hosted API to decide matches is "using external ... APIs, or services to ... resolve entities"
      (README Fair Play).
- **(c) Correct practice:**
  - The final model is the team's own LightGBM booster (MIT library).
  - For any neural add-on, check the license tag **and** the exact parameter count via the HF API before downloading. Record model id, sha,
    license and params in the Documentation.
  - Allowed examples:
    - `intfloat/multilingual-e5-small` (MIT, 117.7 M)
    - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 117.7 M)
    - Qwen2.5-0.5B/1.5B/7B-Instruct (Apache-2.0)
  - Measured CPU throughput on this laptop (`research/bench_log.txt`, `research/bench_model2vec_results.txt`):

    | model | throughput | time for 10 M |
    |---|---|---|
    | multilingual MiniLM-L12 | ~105 names/s | ~26 h |
    | cross-encoders | 14-16 pairs/s | 175-205 h |
    | model2vec `potion-base-8M` (MIT) | 2,786 texts/s | ~1 h |

    At full scale on this CPU, only static embeddings are feasible.
  - Never put `groq` or any other API client into the shipped environment.

### F04 [S0 gate] The validation step cannot run on this laptop as written, and its "skip" command does not skip
skills: `output-and-validation`
- **(a) Plan did:**
  ```
  python utils\validate_submission.py --matching output\matching_results.tsv --candidate output\candidate_pairs.tsv --test-dir dataset\test
  python utils\validate_submission.py --matching output\matching_results.tsv --test-dir dataset\test --check-ids
  ```
  with the note "The second run skips `--candidate` so that `--check-ids` fits in 16 GB RAM".
- **(b) Why wrong:**
  - **[measured]** The validator does `candidate_path = args.candidate or "output/candidate_pairs.tsv"`. Omitting `--candidate` therefore **still
    loads `output/candidate_pairs.tsv`** whenever that path exists relative to the working directory.
  - I planted a broken `output/candidate_pairs.tsv` and ran the plan's second command. The validator printed
    `candidate_pairs.tsv: repeated ID inside a candidate_entity_ids list` and `required S1 entity(ies) missing`, then **FAIL**. Nothing was skipped.
  - At the plan's K = 20-30, the candidate file holds 35-52 M IDs, which the validator keeps in a dict of Python sets. The measured memory
    (F16) is far above the 1-4 GB free. Both commands risk OOM, and a team under deadline pressure then uploads an unvalidated file.
  - Both commands assume the working directory is `student_resource\`, so they require outputs under `student_resource\output\` (F01).
- **(c) Correct practice:**
  - Run the validator from the project root with **absolute paths**.
  - To really skip the candidate cross-check, pass a path that does not exist: `--candidate NONE`. Verified: it prints
    "NONE not found — skipping candidate_pairs.tsv checks" and then PASS.
  - Recommended runs:
    1. The scored file: `--matching <abs>\matching_results.tsv --candidate NONE --test-dir <abs>\student_resource\dataset\test --check-ids`.
    2. The candidate file and the subset rule: either the official validator with `--candidate`, run when enough RAM is free (F16), or a
       **streaming re-implementation** of the same rules for the candidate file. One pass, per-row checks, with sets held only for S1 IDs, and the
       subset check done by sorted merge on (s1, id).
  - Format facts **[measured]** against the official validator:
    - A UTF-8 **BOM fails**: `unexpected header ['\ufeffsource1_entity_id', 'matched_entity_ids']`. Windows PowerShell 5.1 `Out-File`/`>`
      and pandas `encoding="utf-8-sig"` both write a BOM.
    - `", "` as the list separator **fails**: `contains IDs without an S2-/S3- prefix:  S3-20`.
    - CRLF line endings **pass**, because Python text mode translates `\r\n`.
    - pandas 2.3.3 `to_csv(path, sep="\t")` on Windows **writes CRLF by default** (measured).
  - How the scorer treats CRLF is unknown, so write LF explicitly with any of:
    - `open(p, "w", encoding="utf-8", newline="")`
    - `df.to_csv(p, sep="\t", index=False, lineterminator="\n", encoding="utf-8")`
    - polars `write_csv(p, separator="\t", quote_style="never")`
  - The plan's own writer is correct and should be kept: `encoding="utf-8", newline=""`, `",".join`, a literal header, and iteration over
    `raw_ids(test_source1)`.

