---
name: er-matcher-training
description: Use when training or validating the pairwise match model for the Master Bolt entity-resolution pipeline — LightGBM on candidate-pair features, GroupKFold by Source-1 entity, out-of-fold (OOF) probabilities, calibration, negative subsampling, leave-one-country-out as a proxy for unseen France, saving/loading the model — or when validation looks too good, AUC is high but macro F0.5 is low, or an optional neural encoder / cross-encoder is being considered.
---

# Training the matcher

## Overview
The final model is a **LightGBM booster (MIT)** on the pair features. Train it with **folds grouped by S1** so no S1 is seen in both training and validation, collect **calibrated OOF probabilities** for every train pair, and hand them to the decision stage — the metric is decided per S1, not per pair.

## When to use
- Training, re-training, validating, or packaging the model; deciding on class weights, subsampling or a neural add-on.
- Not for the decision rule (**REQUIRED SUB-SKILL:** er-f05-decisions) or feature code (**See also:** er-pair-features).

## Quick reference
| Item | Value |
|---|---|
| Code | `model.py` → `src/ber/model.py`: `ModelConfig`, `train_matcher(feats, cfg) -> (bundle, oof)`, `predict_matcher(bundle, feats)`, `save_bundle/load_bundle`, `loco_eval({country: feats})` |
| Folds | `metric.fold_of(s1, n_folds=5, seed=0)` (same hash as `io.make_folds`) |
| Params (default) | binary, lr 0.05, 127 leaves, min_data_in_leaf 100, feature/bagging fraction 0.8, λ2 1, deterministic, early stopping 100 on an inner 10% S1 split, final model on all folds with 1.1 × mean best_iter |
| Calibration | isotonic on OOF raw scores (monotone), applied to test |
| Output | `oof`: `s1, cand, label, p_raw, p`; bundle dir: `model_final_0.txt` + `bundle.json` (features, params, calibration knots, report incl. top gain) |
| Tests | `python -m pytest -q -p no:cacheprovider test_model.py` (3 pass) |

## Measured (train Karnataka slice, 58,993 S1, cap 30 → 1.77M pairs, 2026-09-25)
Blocking recall at cap 30 = 0.977 (λ_missing = 0.079 per S1). Features 29.5 s; 5 folds + final model 620 s on the laptop (best_iter 489–609, final 611 rounds).
| Decision on OOF p (all 58,993 S1 in the denominator) | macro F0.5 |
|---|---|
| threshold 0.3 / 0.5 / 0.7 | 0.9712 / 0.9757 / 0.9767 |
| expected-F, no exclusivity, empty_bias 1 / 2 / 3 | 0.9765 / 0.9768 / 0.9769 |
| **soft exclusivity + expected-F, empty_bias 2** | **0.9772** |
| oracle (all in-candidate true links) | 0.9931 |
OOF AUC 0.99984 (every fold) — AUC is saturated; decide on macro F0.5. Top gain: `comp_gap` ≫ `comp_is_best` ≫ `a_tset`, `prio`, `p_key_hit`, `len_ratio`, `n_tsort`, `n_partial`, `n_tset`, `idf_cand` — the **competition (exclusivity) features dominate**. Caveats: one Indian state only; US, LOCO and France not yet measured; the remaining gap to the oracle (0.016) and the 2.3% blocking loss are the next targets (er-error-analysis). Reproduce on any labelled slice with `python eval_slice.py --s1 ... --pool ... --cands ... --gt ... --cap 30 --out eval.json` (this folder) and log the result in `work/experiments.md`.

## Procedure
1. Features for all train candidates (per country, complete blocks) → concatenate countries.
2. `bundle, oof = train_matcher(feats, ModelConfig())` → `save_bundle(bundle, "work/model")`, `oof.write_parquet(...)`.
3. Tune the decision on `oof` over **all** train S1 ids (**er-f05-decisions**); log OOF macro F0.5 per country in `work/experiments.md`.
4. LOCO: `loco_eval({"US": f_us, "India": f_in})` → score each direction with the same decision rule. A large drop from in-country OOF means features are language-specific — the risk for France. Prefer feature changes that close the LOCO gap even at a tiny in-country cost.
5. Test: `predict_matcher(bundle, test_feats)` per country (France included, pooled parameters).
6. On the laptop, if RAM is short: `ModelConfig(neg_frac=0.3)` (weights 1/0.3 keep calibration in expectation) or train on 2–3 complete states per country; measured LightGBM throughput 2.8M row-rounds/s, 2M×40 in 0.8 GB. Full-scale training → AWS (**See also:** er-compute-and-aws).

## Optional neural add-ons (only if they raise OOF macro F0.5)
Allowed only as local MIT/Apache ≤ 8B models listed in `models_manifest.json` (**REQUIRED SUB-SKILL:** er-compliance-licensing). Feasible at full scale on CPU: model2vec static embeddings (`potion-multilingual-128M`, MIT, ~2k strings/s) as an extra cosine feature. `intfloat/multilingual-e5-small` (MIT, zero-shot cross-script AUC 0.967) only for the native-script subset or on a GPU box. Cross-encoders run 14–16 pairs/s on this CPU → only for a few thousand uncertain pairs, never all candidates.

## Common mistakes
| Mistake | Fix |
|---|---|
| Random pair-level split (an S1's other candidates leak through competition/group features) | GroupKFold by S1 (`fold_of`) |
| Validation pool restricted to GT-linked records | full pools (distractors are what makes precision hard) |
| Judging by AUC (≈0.9998 here) | macro F0.5 per S1 after decisions; AUC saturates |
| Tuning on the public leaderboard | OOF only |
| Using fold models' in-sample predictions to tune the decision | OOF `p` only |
| Country as a feature / per-country models only | pooled model; LOCO to check transfer; France uses the pooled model |
| Forgetting to retrain on all folds for test | `train_matcher` returns a final all-data booster |
| Hosted LLM / API as a "judge" | forbidden (er-compliance-licensing) |
