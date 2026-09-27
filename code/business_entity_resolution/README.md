# Business Entity Resolution — Team Master Bolt (Amazon ML Challenge 2026)

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the organiser data using only this
folder. No network access, no external data, no API keys, no hosted models. The only learned models are LightGBM
gradient-boosted trees (MIT license), trained here from the provided training data.

## 1. Environment (Python 3.10, Linux x86_64 or Windows)
```bash
python -m venv .venv
# Linux: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data layout expected
```
<DATA_DIR>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA_DIR>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
(the organiser `dataset/` folder, unchanged)

## 3. Run end-to-end
```bash
cd code/business_entity_resolution
python src/run_pipeline.py --data-dir <DATA_DIR> --work-dir <WORK_DIR> --out-dir <OUT_DIR> \
       --config configs/submission_v3.json --stage all --n-jobs 14
```
`configs/submission_v3.json` is the configuration that produced the submitted outputs (shipped in the zip):
60 candidates per S1, learned candidate priority (`learned_prio`, fitted on 20% of train S1 by whole state blocks,
excluding the 30% hash sample `prio_exclude_frac`), city-voted alternate blocks, matcher trained on a 15% hash sample
of train S1 (`train_s1_frac`) with LightGBM learning rate 0.1. `models_manifest.json` lists both LightGBM models with
their licenses and exact parameter counts (checked by the compliance audit).
Stages run in order and are resumable: a stage whose outputs exist is skipped (re-run one with `--stage <name> --force`).
`--splits train|test` and `--countries <c1,c2>` restrict `prepare/block/features/predict` to a shard, so independent
shards can run on separate machines against a shared `<WORK_DIR>`.

| stage | does | writes |
|---|---|---|
| prepare | TSV -> parquet cache (row counts verified), native-script token dictionary learned from TRAIN links, normalized names/addresses per split and country | `cache/`, `native_token_dict.tsv`, `norm/` |
| prio | fits the candidate-priority model (LightGBM on blocking signals) on TRAIN labels of 20% of each train country's S1 (whole geo blocks), excluding the S1 used by the matcher | `model/prio_model.txt` |
| block | candidate generation per split/country: exact-key passes + TF-IDF char-3gram top-K in both directions inside state blocks (+ city-voted alternate blocks, empty-address fallback block), learned priority, 60 candidates per S1 | `cands/` |
| features | 72 pairwise / context / competition / group / name-ambiguity features | `feats/` |
| train | LightGBM, 5-fold GroupKFold by S1 on a 15% hash sample of train S1, isotonic calibration, decision parameters chosen on OOF macro F0.5 | `model/` |
| predict | scores test candidates, soft exclusivity + per-S1 expected-F0.5 list selection (each S2/S3 id -> at most one S1) | `pred/` |
| write | both output TSVs in `test_source1` order | `<OUT_DIR>/` |

Measured (Modal CPU containers, 16-32 cores; 5 split/country shards of prepare/block/features in parallel):
prepare 10 min · prio 9 min · block 3-14 min per shard · features 2-12 min per shard · train 27 min (32 cores,
~23 GB peak) · predict 15 min · write 1 min. Peak memory per shard 20-40 GB, so a 64 GB+ machine is needed at full scale.
Out-of-fold macro F0.5 of this configuration on train: 0.98152 (US 0.98364, India 0.97836).

## 4. Determinism
All seeds are fixed in `src/ber/config.py` / `configs/*.json` (`seed` 0; LightGBM `deterministic=true`). Folds and
samples are a version-stable hash of the S1 id (`metric.fold_of`). Re-running on the same machine type gives the same
outputs (bit-identical re-runs were not verified across different CPU types).

## 5. Validate
```bash
python student_resource/utils/validate_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate NONE --test-dir <DATA_DIR>/test --check-ids
python tools/check_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate <OUT_DIR>/candidate_pairs.tsv --test-dir <DATA_DIR>/test --check-ids
```

## 6. Code map
`src/run_pipeline.py` (entry point, stages) · `src/ber/io.py` (safe TSV reading, parquet cache) · `normalize.py`,
`normalize_frame.py` (transliteration, name/address normalization, native-token dictionary) · `blocking.py`
(candidate generation, learned priority) · `features.py` · `model.py` (LightGBM + calibration) · `decide.py`
(exclusivity + expected-F0.5 selection) · `metric.py` (exact macro F0.5) · `outputs.py` · `config.py` ·
`memory_guard.py` · `configs/` (run configurations) · `tools/check_submission.py` (streaming output validator).
