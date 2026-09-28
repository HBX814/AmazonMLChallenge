# Business Entity Resolution — Team Master_Bolt (Amazon ML Challenge 2026)

This folder is a self-contained, runnable copy of our pipeline. From the organiser data it regenerates both submission
files:

- `output/matching_results.tsv` — the final matches (identical to the file of our best leaderboard submission, 0.983714)
- `output/candidate_pairs.tsv` — the candidate set of the last blocking step (5.02 candidates per Source-1 entity)

No external data, databases, geocoders, APIs, API keys or hosted models / LLM services are used. All learned models are
trained here from the provided training data: LightGBM gradient-boosted trees (MIT) and two fine-tuned open-weights
transformer cross-encoders (MIT / Apache-2.0, largest 567.8M parameters, far below the 8B cap; see
`models_manifest.json`). The only network access is the one-time download of the two pretrained backbones from the
Hugging Face Hub at pinned revisions (`intfloat/multilingual-e5-small`, `BAAI/bge-reranker-v2-m3`).

Pipeline at a glance:

```
TSV data -> prepare (normalize, transliterate) -> prio (learned candidate priority)
         -> block (exact keys + TF-IDF top-K both directions, 60 candidates per S1)
         -> features (98 pair features) -> train (stage-1 LightGBM, calibrated OOF)
         -> crossenc (fine-tune 2 cross-encoders on TRAIN pairs; stage-2 LightGBM; adaptation priors)
         -> predict (stage 1 -> filter p >= 0.001 = final candidate set -> cross-encoders + stage 2
                     -> shift adaptation -> per-S1 expected-F0.5 list selection -> targeted rule)
         -> write (matching_results.tsv, candidate_pairs.tsv)
```

---

## 1. Requirements

| resource | needed for the full data | notes |
|---|---|---|
| OS / Python | Linux x86_64 (tested) or Windows, **Python 3.10** | |
| CPU / RAM | 16-32 cores, **64 GB RAM minimum, 128 GB recommended** | peak 20-40 GB per stage shard; stage-1 training ~28 GB |
| Disk | **~50 GB free** for `<WORK_DIR>` | cache 1 GB, normalized frames 2 GB, candidates 4 GB, features 22 GB, cross-encoder data + models ~5 GB |
| GPU | **one CUDA GPU (>= 16 GB) strongly recommended for the `crossenc` stage** | the large cross-encoder takes ~4 h to train + ~1 h to score on 2 x T4; all other stages are CPU-only |
| Network | only for the first `crossenc` run (backbone download, ~2.5 GB) | afterwards everything runs offline |


## 2. Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
# GPU machine: first install the torch 2.5.1 wheel built for your CUDA version (PyTorch "previous versions" page),
# then the rest; CPU-only machines can install everything directly:
pip install -r requirements.txt
```

`requirements.txt` pins every dependency (polars, pyarrow, numpy, scipy, scikit-learn, rapidfuzz, lightgbm,
sparse-dot-topn, psutil, torch, transformers, tokenizers, safetensors). Optional environment variables:

| variable | purpose |
|---|---|
| `HF_HOME=<dir>` | where the pretrained backbones are cached (downloaded once) |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | after the first download: run provably offline |
| `CE_THREADS=<n>` | CPU threads for the cross-encoders when no GPU is present |
| `OMP_NUM_THREADS=<n>`, `POLARS_MAX_THREADS=<n>` | cap threads on shared machines |

## 3. Data layout expected

The unchanged organiser `dataset/` folder:

```
<DATA_DIR>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA_DIR>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

All TSVs are read as tab-separated with quoting and NA parsing disabled (`src/ber/io.py`); row counts are verified
against the raw files. `country` is treated as an open set of labels (France exists only in test and goes through the
same code path).

## 4. Reproduce end-to-end (one command)

```bash
cd code/business_entity_resolution
python src/run_pipeline.py --data-dir <DATA_DIR> --work-dir <WORK_DIR> --out-dir <OUT_DIR> \
       --config configs/submission_v5.json --stage all --n-jobs 14
```

`configs/submission_v5.json` is the exact configuration of the submitted outputs. The run takes about 10 hours on the
machine of Section 1 (about 7 h of it is the cross-encoder stage on a GPU). Every stage writes parquet under
`<WORK_DIR>`, and a stage whose outputs already exist is skipped, so an interrupted run resumes where it stopped
(re-run a single stage with `--stage <name> --force`). Progress, timings and memory are logged to the console and to
`<WORK_DIR>/logs/pipeline.log`.

Command-line options of `src/run_pipeline.py`:

| option | meaning |
|---|---|
| `--data-dir` | organiser dataset folder (with `train/` and `test/`) |
| `--work-dir` | intermediate files (parquet caches, candidates, features, models, predictions, logs) |
| `--out-dir` | where `matching_results.tsv` and `candidate_pairs.tsv` are written |
| `--config` | JSON overrides of `src/ber/config.py` (`configs/submission_v5.json` for the submission) |
| `--stage` | `prepare`, `prio`, `block`, `features`, `train`, `crossenc`, `predict`, `write` or `all` |
| `--n-jobs` | worker processes / threads for the CPU stages |
| `--force` | recompute the selected stage even if its outputs exist |
| `--splits train,test` / `--countries c1,c2` | run `prepare` / `block` / `features` / `predict` for a shard only |
| `--max-s1 N` | development only: keep N Source-1 entities per country (pools stay full); not used for the submission |

## 5. Step by step (recommended for the full data)

Run the stages in this order from `code/business_entity_resolution/`; `ARGS` stands for
`--data-dir <DATA_DIR> --work-dir <WORK_DIR> --out-dir <OUT_DIR> --config configs/submission_v5.json --n-jobs 14`.
Times were measured on 16-32-core cloud CPU containers.

| # | command | what it does | writes | time / peak RAM |
|---|---|---|---|---|
| 1 | `python src/run_pipeline.py ARGS --stage prepare` | TSV -> parquet cache (row counts verified); native-script -> Latin token dictionary learned from TRAIN links only; normalized names / addresses / house numbers / city / state per split and country | `cache/`, `native_token_dict.tsv`, `norm/` | 10 min / 11 GB |
| 2 | `... --stage prio` | candidate-priority LightGBM on blocking signals only, fitted on TRAIN labels of 20% of each train country's S1 (whole state blocks, excluding the models' training S1) | `model/prio_model.txt` | 9 min |
| 3 | `... --stage block` | per split and country: exact-key passes, address-key and street-name passes (5 reserved slots), TF-IDF character-3-gram top-K on name / address / name+address in both directions inside state blocks, city-voted alternate blocks, empty-address fallback block; learned priority keeps 60 (+5) candidates per S1; train candidates get their label | `cands/` | 3-14 min per shard / 23-26 GB |
| 4 | `... --stage features` | 98 language-agnostic pair features (name, address, house-number relation classes, blocking signals, per-S1 context, competition over all S1, group similarity, name ambiguity) | `feats/` | 2-12 min per shard |
| 5 | `... --stage train` | stage-1 LightGBM on a 15% hash sample of train S1 (331,205 S1, 19.6M pairs), 5-fold GroupKFold by S1, isotonic calibration, decision parameters chosen on out-of-fold macro F0.5 | `model/` | 22 min (32 cores) / 28 GB |
| 6 | `... --stage crossenc` (**GPU**) | stage-1 p of every train pair; for each cross-encoder: fine-tune on the pairs with stage-1 p in [0.01, 0.99] of its own train S1 hash buckets (never the 15% sample), score the same band of the OOF sample and of the test set; then the stage-2 LightGBM and the adaptation source priors | `crossenc/`, `model/stage2/`, `model/adapt_source.json` | ce2 on 2 x T4 GPUs: 4.1 h training + 1.1 h scoring; ce on a 32-core CPU: 75 min training + ~1.5 h scoring (much faster on a GPU); CPU part (stage-1 p of all train pairs, stage 2, priors) 12 min / 19 GB |
| 7 | `... --stage predict` | per test country: stage 1 on all ~59 candidates -> **filter: keep pairs with stage-1 p >= 0.001 = final candidate set** -> cross-encoder features + stage-2 re-scoring of exactly these pairs -> label-free density adaptation of the house-number-relation classes -> soft at-most-one-owner exclusivity + per-S1 expected-F0.5 list selection -> targeted rule | `pred/` | 16 min / 10 GB |
| 8 | `... --stage write` | both TSVs in `test_source1` order (every S1 exactly once, empty lists allowed, ids ordered by probability) | `<OUT_DIR>/` | 1 min |

Parallel machines: steps 1, 3, 4 and 7 accept `--splits` / `--countries`, so the five shards (train/India, train/US,
test/France, test/India, test/US) can run on separate machines that share `<WORK_DIR>`
(e.g. `--stage features --splits train --countries US`).

Running the GPU stage on a separate machine: run steps 1-5 on the CPU machine, copy (or mount) `<WORK_DIR>` on a GPU
machine, run step 6 there, copy `<WORK_DIR>/crossenc/` and `<WORK_DIR>/model/` back, then run steps 7-8. Without a GPU,
step 6 runs on CPU (bf16 where the CPU supports AVX512-BF16/AMX, else fp32) but the large cross-encoder then takes days.

## 6. Configuration (`configs/submission_v5.json`)

| key | value | meaning |
|---|---|---|
| `train_s1_frac` | 0.15 | hash sample of train S1 used for stage-1 / stage-2 training and OOF decision tuning |
| `learned_prio`, `prio_block_frac`, `prio_exclude_frac` | true, 0.2, 0.3 | learned candidate priority fitted on 20% of the S1 (whole blocks), excluding the 30% hash sample |
| `blocking.max_cands_per_s1`, `blocking.city_alt_blocks` | 60, true | per-S1 cap after the learned priority; city-voted alternate state blocks |
| `model.params` | LightGBM, 127 leaves, lr 0.1, deterministic | stage-1 matcher |
| `stage2` | enabled, guard G1 | collective re-scoring of pairs with stage-1 p >= 0.001; house-number-mismatch pairs may only go down |
| `adapt` | `density_hs`, rule true | label-free adaptation of the house-number-relation classes to the test density + targeted rule |
| `crossenc.band` | [0.01, 0.99] | stage-1 p band scored by the cross-encoders |
| `crossenc.models` | `ce`: multilingual-e5-small (first 6 layers), train buckets 300-449, 1.5 epochs; `ce2`: bge-reranker-v2-m3, train buckets 300-999, 1 epoch, frozen word embeddings, 25% synthetic shifted-house-number negatives | backbones pinned by `revision` |
| `candidate_prune_p1` | 0.001 | last blocking step: the stage-1 matcher keeps pairs with p >= 0.001 (= the stage-2 floor); `candidate_pairs.tsv` = these pairs |

All other settings are the defaults in `src/ber/config.py` (every field is documented there).

## 7. Expected outputs and validation

| file | rows | content | sha256 of the submitted file |
|---|---|---|---|
| `matching_results.tsv` | 1,732,544 S1 + header | 5,813,210 links (France 875,090 · India 2,706,468 · US 2,231,652) | `6448e77df7e240bdf3da43cf955c780adada2d564712def12c96cfc8a5feb40e` |
| `candidate_pairs.tsv` | 1,732,544 S1 + header | 8,689,809 candidate pairs = 5.02 per S1 (France 5.80 · India 4.75 · US 5.03); blocking alone gave 102,281,392 (~59 per S1) | `db613c573514781950ab5d5fdbce2e537de15ac5b6486dade7efa4b17fd469e5` |

Every matched id is a candidate of the same S1. Validate with the organiser validator (from the `student_resource/`
folder) and our streaming checker:

```bash
python utils/validate_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate <OUT_DIR>/candidate_pairs.tsv --test-dir dataset/test
python tools/check_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate <OUT_DIR>/candidate_pairs.tsv --test-dir <DATA_DIR>/test --check-ids
```

Expected quality (macro F0.5, singletons included): out-of-fold on 331,205 train S1 **0.98870**; honest holdout of
330,649 train S1 that no model ever saw **0.98864**; candidate-set ceiling (perfect decisions on the candidates)
0.99536; public leaderboard **0.983714**.

## 8. How the submitted outputs were produced

All CPU stages (1-5, the CPU part of 6, 7 and 8) were run with this code and `configs/submission_v5.json` on cloud CPU
containers (16-32 cores, 64-128 GB RAM). The two cross-encoders were fine-tuned with the recipe implemented in
`src/ber/crossenc.py` (same data selection, hyperparameters and synthetic negatives): `ce` on a 32-core CPU container,
`ce2` on 2 x T4 GPUs; their scores were placed in `<WORK_DIR>/crossenc/` exactly as the `crossenc` stage writes them,
and steps 6 (stage 2 + adaptation priors), 7 and 8 were then run by `run_pipeline.py`. Re-running steps 7-8 on that
work directory reproduces both submitted files byte for byte (sha256 above).

## 9. Determinism

Seeds are fixed in `src/ber/config.py` / `configs/submission_v5.json`; samples and folds are a version-stable hash of
the S1 id (`metric.fold_of`); LightGBM runs with `deterministic=true` on canonically sorted rows; joins keep row order.
The CPU stages are therefore bit-identical across re-runs. Fine-tuning the cross-encoders on GPUs is not
bit-deterministic (cuDNN kernels, mixed precision), so a full re-run from scratch reproduces the recipe and the quality (in our runs the
validation macro F0.5 of repeated cross-encoder variants agreed within ~0.0001) but not the exact same links.

## 10. Troubleshooting

| symptom | fix |
|---|---|
| out of memory in `block` / `features` / `predict` | run one shard at a time with `--splits` / `--countries`; lower `--n-jobs`; `features.chunk_pairs` in the config controls the batch size |
| out of GPU memory in `crossenc` | lower the model's `bs` in `crossenc.models` (e.g. 32) |
| `crossenc` cannot download the backbones | set `HF_HOME` to a folder that already holds the two pinned revisions and `HF_HUB_OFFLINE=1` |
| a stage does nothing | its outputs already exist; add `--force` |
| validator complains about ids | check that `--data-dir` points at the unchanged organiser files |

## 11. Code map

- `src/run_pipeline.py` — entry point, the eight stages
- `src/ber/io.py` — safe TSV reading (tab, no quoting, no NA parsing), parquet cache with row-count checks
- `src/ber/normalize.py`, `src/ber/normalize_frame.py` — transliteration, name / address normalization, native-token dictionary
- `src/ber/blocking.py` — candidate generation, learned candidate priority
- `src/ber/features.py` — the 98 pair features
- `src/ber/model.py` — LightGBM training with GroupKFold by S1, isotonic calibration
- `src/ber/crossenc.py` — cross-encoder training recipe, synthetic hard negatives, scoring, stage-2 features
- `src/ber/stage2.py` — collective stage-2 model and guard
- `src/ber/adapt.py` — label-free shift adaptation, targeted rule
- `src/ber/decide.py` — soft exclusivity and per-S1 expected-F0.5 list selection
- `src/ber/metric.py` — exact macro F0.5 and error breakdowns
- `src/ber/outputs.py` — submission writers; `src/ber/config.py` — all settings; `src/ber/memory_guard.py` — RAM guard
- `configs/submission_v5.json` — submitted configuration; `models_manifest.json` — models, licenses, exact parameter
  counts, pinned revisions; `tools/check_submission.py` — streaming output validator
