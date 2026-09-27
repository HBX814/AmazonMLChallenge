# Plan v5 — from LB 0.981499 towards the top (2026-09-27)

## 0. Reality check
- OOF oracle ceiling of the v4 candidate lists = **0.99540** (perfect decisions on every generated candidate). 0.997 is
  impossible without new candidates; 0.995 is the ceiling today. Top team LB 0.9907. Realistic target: **LB >= 0.991**.
- Current: LB 0.981499 (ce_hsr_rule) · OOF 0.98752 · HB 0.98755. The CE moved LB +0.0091 for +0.0018 OOF -> the CE mostly
  fixes test-shift errors; it is the lever that transfers.
- Public LB is a subset; final ranking = private LB -> choose by OOF/HB, use the 2 remaining uploads only to confirm.

## 1. Where the remaining loss is (OOF error ladder, err_full_oof.md + tags)
- recall-model (true candidate not selected) ~48% of lost points; blocking misses ~31%; false links ~22%.
- Micro precision 0.9956 vs recall 0.951 -> the model is unsure on hard true copies, F0.5 keeps them out.
- Dominant cause: **name-only records** — empty candidate address (US: 72% of FN-model, 68% of FN-blocking, 42% of FP;
  lift ~80x), alias/DBA names (lift 4-7.5x, esp. blocking India), domain-form names (India blocking lift 1.9).
  Data structure: empty-address pool records are true copies of SOME S1 97.7% of the time; DBA 99.98%; domains 95.7%.
- Test shift: ~1.9x distractors/S1 (copy-like, house number shifted 3-100 / one digit +-3..9); France concept shift
  (generic short names, legal forms "Et Fils", 15 cities).

## 2. Research take-aways
- Foursquare Location Matching (Kaggle 2022, closest public task: business name+address): every top solution =
  embedding/kNN + text candidates -> GBDT ranker -> **large multilingual transformer pair classifiers**
  (xlm-roberta-large, mdeberta-v3-base, ensembles) -> graph post-processing. The transformer stage gave the big jumps.
- Ditto (VLDB 2020): serialize pairs; **domain-knowledge injection** (mark numbers/ids with span tags, normalise spans);
  **augmentation** (span del/shuffle, attr del/shuffle, entry swap, MixDA) +2.5 F1; lr 3e-5, len 256, 10-40 epochs.
- Allowed backbones (MIT/Apache, <=8B): xlm-roberta-large (MIT, 560M), mdeberta-v3-base (MIT, 280M; fp16 unstable -> bf16),
  multilingual-e5-small/base/large (MIT), bge-m3 (MIT), **bge-reranker-v2-m3 (Apache-2.0, 568M, XLM-R-large reranker)**,
  gte-multilingual-reranker-base (Apache-2.0, 306M).
- Compute: Modal GPU needs a payment method (Starter $30/month credit; card unlocks GPUs); Kaggle free T4x2 ~30 GPU-h/week
  (P100 retired 2026-09-15), private datasets up to 200 GB.

## 3. Workstreams (ordered by expected LB gain per $)
W1 Cross-encoder v2 (largest lever)
  - GPU: bge-reranker-v2-m3 (or xlm-roberta-large) + mdeberta-v3-base, bf16; CPU fallback: e5-small 12 layers.
  - Train data: all train S1 outside OOF/HB (hash buckets 300-999, ~1.5M S1), pairs with stage-1 p >= 1e-3 + easy negatives.
  - Coverage: score ALL stage-1 p >= 1e-3 pairs (test 8.7M, OOF 1.5M, HB 1.5M), not only the 0.01-0.99 band, so
    confident copy-like false links (test shift) can be caught.
  - Ditto-style: house-number span tags; targeted augmentation = synthetic test-like hard negatives (true pair with the
    house number shifted 3-100 / one digit +-3..9 -> label 0), legal-form / '&'-'and' / word-order swaps on positives,
    entry swap.
  - Expect OOF +0.002-0.004, LB more (shift errors).
W2 Recall for name-only records
  - Measure v4 blocking recall per tag (empty address, alias, domain); add a name-only candidate pass for empty-address
    pool records (country-wide exact/near name keys, reserved slots) and domain splitting ("wilfordhancock.com").
  - Learned bi-encoder retrieval (e5 fine-tuned contrastively on S1<->copy pairs) -> top-k extra candidates.
  - Expect ceiling +0.002-0.003, realised +0.001-0.002. Needs block -> features -> train rerun (~$5 CPU).
W3 Stage-2 upgrades: extra CE features (ce8 variant running), candidate-side CE competition (needs W1 full coverage),
  LightGBM tuning, seed bagging, CE ensemble.
W4 Shift/France: refit adaptation on the new model; re-check the targeted rule on the simulated shift.
W5 Package: CE inside the pipeline (ber/crossenc.py, stages, config), README, requirements, models_manifest, docs,
  compliance audit, zip (must reproduce the uploaded file).

## 4. Uploads (2 left)
- #1 after W1 (+W3): confirms CE v2 transfer. #2 final: best OOF/HB model (+W2 if ready). Fallback kept:
  output_probes/ce_hsr_rule (0.981499).

## 5. Decisions (user, 2026-09-27) and the 24 h schedule
- GPU = Kaggle free T4x2 (kaggle CLI 1.7.4.5, user harsh0814; kernel-metadata machine_shape "NvidiaTeslaT4").
- Deadline < 24 h -> W2 (new candidates) dropped. Modal spend stays within the remaining ~$12 credit.
- Stage-2 variants on CE v1 (ce8 extra CE features, lr 0.05/255 leaves): OOF +0.0001, HB -0.00004 -> noise; the lever is the
  CE model itself.
- A (Modal CPU, ~$0.4): ce_data_v2.py -> kg_train (train S1 buckets 300-999, band 0.01-0.99, ~1.1M pairs, texts) +
  kg_eval (band rows of OOF / HB / test, ~2.25M) -> laptop -> private Kaggle dataset.
- B (Kaggle T4x2, ~2 h): BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M, XLM-R-large) fine-tuned 1 epoch, fp16, DataParallel,
  lr 2e-5, + synthetic house-number-shift hard negatives; score kg_eval. Meanwhile: CE integration into the pipeline (W5).
- C (Modal CPU, ~$1.5): stage 2 with CE v1 + CE v2 features -> OOF/HB -> ce_apply -> validate -> upload #1.
- D (if Kaggle time): second CE (different backbone) for an ensemble -> upload #2 final. E: package + docs + zip.
