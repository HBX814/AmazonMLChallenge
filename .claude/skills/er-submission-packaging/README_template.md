# Business Entity Resolution — Team Master Bolt (Amazon ML Challenge 2026)

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the organiser data using only this folder.
No network access, no external data, no API keys. Final model: LightGBM (MIT). <!-- list any other model + license -->

## 1. Environment (Python 3.10, Windows or Linux x86_64)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Data layout expected
```
<DATA_DIR>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA_DIR>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
(the organiser `dataset/` folder unchanged)

## 3. Run end-to-end
```bash
python src/run_pipeline.py --data-dir <DATA_DIR> --work-dir <WORK_DIR> --out-dir <OUT_DIR> --stage all --n-jobs 8
```
Stages (each resumable; re-run one with `--stage <name> --force`):
| stage | does | writes |
|---|---|---|
| prepare | TSV → parquet cache, normalization, native-token dictionary learned from TRAIN links | `<WORK_DIR>/cache`, `<WORK_DIR>/norm` |
| block | candidate generation per country (train and test) | `<WORK_DIR>/cands` |
| features | pairwise/context/competition features | `<WORK_DIR>/feats` |
| train | LightGBM, 5-fold GroupKFold by S1, calibration, decision parameters chosen on OOF macro F0.5 | `<WORK_DIR>/model` |
| predict | scores test candidates, exclusivity + per-S1 selection | `<WORK_DIR>/pred` |
| write | both output TSVs | `<OUT_DIR>/` |
<!-- fill in measured runtimes / peak RAM per stage and the machine used, e.g. "r6a.4xlarge, 128 GB: 2 h 10 min total" -->

## 4. Determinism
Seeds fixed in `src/ber/config.py` (<!-- values -->). Folds are a hash of the S1 id. Re-running gives identical outputs <!-- verified: yes/no -->.

## 5. Validate
```bash
python <path>/student_resource/utils/validate_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate NONE --test-dir <DATA_DIR>/test
```

## 6. Code map
`src/run_pipeline.py` (entry point) · `src/ber/io.py` · `normalize.py` · `normalize_frame.py` · `blocking.py` · `features.py` · `model.py` · `decide.py` · `metric.py` · `outputs.py` · `config.py`
