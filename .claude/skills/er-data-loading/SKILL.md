---
name: er-data-loading
description: Use when reading the challenge TSV files (train/test source1-3, train_ground_truth) for the Master Bolt entity-resolution project, building the parquet cache, exploding matched_entity_ids to long form, creating validation folds, building geographic or random dev slices, or when rows go missing, fields merge, "NULL"/"None" turn into NaN, quotes break parsing, or loading runs out of memory.
---

# Loading the ER data safely

## Overview
Every later number is wrong if one field is corrupted on read. Read with **TAB separator, quoting disabled, every column as string, no NA parsing**, cache once to parquet, and work lazily from the cache. Validation is always **by S1 entity with full pools**.

## When to use
- First contact with the data, any new loader, fold creation, dev slices, "row count doesn't match", memory blow-ups on load.
- Not for normalizing text (**See also:** er-text-normalization).

## Quick reference
| Task | Command / call (code in this folder; copy `io.py` to `src/ber/io.py`) |
|---|---|
| Check files | `python io.py verify --data-dir student_resource/dataset --head 100000` |
| Build cache | `python io.py cache --data-dir student_resource/dataset --work-dir work` → `work/cache/*.parquet` (streaming, skips if fresh) |
| Folds | `python io.py folds --work-dir work --n-folds 5 --seed 0` → `work/cache/train_folds.parquet` (hash-based, same as `metric.fold_of`) |
| Lazy frames | `from ber import io as bio; lz = bio.scan_split("work", "train")` → keys `s1, s2, s3, gt` |
| Eager (small data only) | `bio.load_split(data_dir, split)`, `bio.read_source(path)`, `bio.read_ground_truth(path)` (long `s1, mid`) |
| Output order | `bio.s1_ids_in_file_order("student_resource/dataset/test/test_source1.tsv")` |
| Geo dev slice | `python make_dev_slice.py geo --work-dir work --split train --country India --state Karnataka --out-dir work/dev/IN_KA` |
| Pick a state | `python make_dev_slice.py states --work-dir work --split train --country US` |
| Random slice (full pool) | `python make_dev_slice.py random --work-dir work --split train --countries India --n-s1 5000 --out-dir work/dev/IN_rand5k` |
Never `import io` inside the package (stdlib clash): use `from ber import io as bio`.

## Format facts (measured on the first 100k lines of all 7 files)
- TAB, header, UTF-8, LF, no BOM; exactly 3 tabs per source line, 1 per GT line.
- Literal `NULL`, `None`, `NA` occur as **data** (address noise tokens, even a whole business name "NA") → keep them as strings; empty field → `""`, never null/NaN.
- The files were written with CSV QUOTE_MINIMAL: a rare field containing `"` is wrapped in quotes with inner quotes doubled (1–6 per 100k rows, mostly France/junk prefixes). Parse with quoting **off** (so rows == lines − 1 always), then decode only well-formed quoted tokens (`unquote=True`, the default in `read_source`/`scan_source`).
- Apostrophes (`'`, `’`) appear in ~2% of fields: plain data.
- IDs: `S1-/S2-/S3-` + random integer of variable length (not sortable by number, not zero-padded). No ID overlap between train and test.

## Sizes and memory
| file | rows | TSV | parquet (zstd) |
|---|---|---|---|
| train S1 / S2 / S3 | 2.21M / 5.03M / 5.29M | 210 / 489 / 504 MB | 81 / 197 / 203 MB |
| test S1 / S2 / S3 | 1.73M / 4.89M / 5.08M | 175 / 509 / 506 MB | 65 / 198 / 200 MB |
| train GT | 2.21M rows → 7.64M links | 127 MB | 58 MB |
A 5M-row source frame in polars is ~0.6–0.9 GB; normalized with extra columns ~2–2.5 GB. On the laptop (1–4 GB free) filter by country (and state block) inside the lazy scan and `collect()` only that part.

## Validation protocol
- Split **S1 ids** into folds (`make_folds`); each fold's S1 are matched against the **full** S2/S3 pool of their country. Never restrict the pool to GT-linked records (hides distractors, inflates precision).
- Links never cross countries, so work per country; iterate over `s1["country"].unique()` — France exists only in test.
- The GT has one row per train S1; empty `matched_entity_ids` = singleton (5.58%). Keep singletons in every evaluation.

## Dev slices
- **Geo slice** (default for iteration): all train S1 of one state + every same-country S2/S3 whose parsed state is that state or unknown. Realistic density and runtime; reports `recall_ceiling` (targets whose address parses to another state — also the cost of hard state sharding). Karnataka: 59k S1 × 291k pool, ceiling 1.0 on its links; 96% of links share the block, 3.9% sit in the unknown-state pool.
- **Random slice**: N S1 per country with the full country pool — honest but expensive (millions of pool rows).
- Slices are also written in the organiser TSV layout, so `run_pipeline.py --data-dir <slice>` runs unchanged.

## Verification checklist (write results to work/experiments.md)
- [ ] rows == `count_data_lines` for every file; ids unique; prefixes match file; no empty names.
- [ ] country values seen per file (train: US, India; test: + France).
- [ ] GT: every S1 has one row; every linked id exists in S2/S3; no id linked to two S1 (exclusivity holds).

## Common mistakes
| Mistake | Fix |
|---|---|
| `pd.read_csv(path)` without `sep="\t"` → one column | `read_source` (polars, TAB) |
| pandas default NA parsing turns "NULL"/"None"/"NA" into NaN | all-string, `null_values=None` / `keep_default_na=False, na_filter=False` |
| Default CSV quoting swallows tabs/rows at a stray `"` | quoting disabled + selective unquote |
| Loading all 12.5M rows eagerly on the laptop | `scan_split` + filter by country/block, then collect |
| Sampling the pool for "speed" in validation | full pools per block, sample S1 only |
| Hard-coding `["US","India"]` | iterate over the countries present |
| `import io` in `src/ber/` | `from ber import io as bio` |
