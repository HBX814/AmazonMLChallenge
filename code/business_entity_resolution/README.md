# Business Entity Resolution — Team Master_Bolt (Amazon ML Challenge 2026)

Regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the organiser data using only this
folder. No external data, no API keys, no hosted models / LLM services. Learned models: LightGBM gradient-boosted trees
(MIT) and fine-tuned open-weights transformer cross-encoders (MIT / Apache-2.0, all far below 8B parameters; see
`models_manifest.json`), all trained here from the provided training data. The only network access is the one-time
download of the open-weights backbones from the Hugging Face Hub (`intfloat/multilingual-e5-small`,
`BAAI/bge-reranker-v2-m3`); set `HF_HOME` to cache them.

## 1. Environment (Python 3.10, Linux x86_64 or Windows)
```bash
python -m venv .venv
# Linux: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install -r requirements.txt          # for a GPU install the torch wheel matching your CUDA first
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
       --config configs/submission_v5.json --stage all --n-jobs 14
```
`configs/submission_v5.json` is the configuration of the submitted outputs. Stages run in order and are resumable: a
stage whose outputs exist is skipped (re-run one with `--stage <name> --force`). `--splits train|test` and
`--countries <c1,c2>` restrict `prepare/block/features/predict` to a shard, so shards can run on separate machines
against a shared `<WORK_DIR>`.

| stage | does | writes |
|---|---|---|
| prepare | TSV -> parquet cache (row counts verified), native-script token dictionary learned from TRAIN links, normalized names/addresses per split and country | `cache/`, `native_token_dict.tsv`, `norm/` |
| prio | candidate-priority LightGBM on blocking signals, TRAIN labels of 20% of each train country's S1 (whole geo blocks), excluding the 30% hash sample | `model/prio_model.txt` |
| block | per split/country: exact-key passes (+ address-key and street-name passes with 5 reserved slots) + TF-IDF char-3gram top-K both directions inside state blocks, city-voted alternate blocks, empty-address fallback block, learned priority, 60 (+5) candidates per S1 | `cands/` |
| features | 98 pairwise / context / competition / group / name-ambiguity / house-number-relation features | `feats/` |
| train | stage-1 LightGBM, 5-fold GroupKFold by S1 on a 15% hash sample of train S1, isotonic calibration, decision parameters on OOF macro F0.5; stage 2 and the adaptation priors run after the cross-encoders | `model/` |
| crossenc | final-model stage-1 p of every train pair; per cross-encoder: fine-tune on the uncertain-band pairs (stage-1 p in [0.01, 0.99]) of its own train S1 hash buckets (disjoint from the 15% sample), score the band pairs of the OOF sample and of the test set; then the stage-2 collective LightGBM (stage-1 features + within-S1 context + stage-1 decision + competition over all S1 + sibling similarities + cross-encoder logits, guard G1) and the label-free shift-adaptation source priors | `crossenc/`, `model/stage2/`, `model/adapt_source.json` |
| predict | test: stage 1 -> **last blocking filter: keep pairs with stage-1 p >= `candidate_prune_p1` (0.001) = the final candidate set, 5.02 per S1** -> stage 2 (+ cross-encoder features) -> density adaptation of the house-number-relation classes -> soft exclusivity + per-S1 expected-F0.5 list selection -> targeted rule (drop a shifted-house-number link when the S1 also has an exact-number link, p < 0.99) | `pred/` |
| write | both output TSVs in `test_source1` order; `candidate_pairs.tsv` = exactly the filtered pairs the stage-2 model and the decision run on (8,689,809 test pairs; blocking alone gives ~59 per S1) | `<OUT_DIR>/` |

Hardware. CPU stages: measured on Modal CPU containers (16-32 cores, 64-128 GB RAM): prepare 10 min · prio 9 min ·
block 3-14 min per shard · features 2-12 min per shard · stage-1 train 22 min · stage 2 ~5 min · predict 8 min ·
write 1 min; peak memory per shard 20-40 GB. `crossenc`: the large cross-encoders were trained on Kaggle GPUs
(2 x T4, fp16; 4.1 h training + 1.1 h scoring); the small one (6-layer e5-small) on a 32-core CPU (fp32,
75 min training, ~20 min scoring on 5 containers). On a single modern GPU (A100/L4) the whole stage takes ~2-3 h.
Out-of-fold macro F0.5 on train (331,205 S1 incl. singletons): stage 1 0.98473 · + stage 2 0.98569 · + cross-encoders
0.98870; honest holdout (330,649 train S1 never used by any model): 0.98864. Public leaderboard: 0.983714.

## 4. Determinism
Seeds are fixed in `src/ber/config.py` / `configs/*.json`; folds and samples are a version-stable hash of the S1 id
(`metric.fold_of`); LightGBM is `deterministic=true` with canonical row order, so the CPU stages are bit-identical across
re-runs. Cross-encoder fine-tuning on GPUs is not bit-deterministic (cuDNN / mixed precision): a re-run reproduces
the recipe and the scores to within noise, not bit for bit.

## 5. Validate
```bash
python student_resource/utils/validate_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate NONE --test-dir <DATA_DIR>/test --check-ids
python tools/check_submission.py --matching <OUT_DIR>/matching_results.tsv --candidate <OUT_DIR>/candidate_pairs.tsv --test-dir <DATA_DIR>/test --check-ids
```

## 6. Code map
`src/run_pipeline.py` (entry point, stages) · `src/ber/io.py` (safe TSV reading, parquet cache) · `normalize.py`,
`normalize_frame.py` (transliteration, name/address normalization, native-token dictionary) · `blocking.py`
(candidate generation, learned priority) · `features.py` (98 features incl. house-number relation classes) ·
`model.py` (LightGBM + calibration) · `stage2.py` (collective stage-2 model, guard) · `crossenc.py` (transformer
cross-encoders: training recipe, synthetic hard negatives, scoring, stage-2 features) · `adapt.py` (label-free shift
adaptation, targeted rule) · `decide.py` (exclusivity + expected-F0.5 selection) · `metric.py` (exact macro F0.5) ·
`outputs.py` · `config.py` · `memory_guard.py` · `configs/` (run configurations) · `tools/check_submission.py`.
