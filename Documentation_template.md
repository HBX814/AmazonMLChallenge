# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Master_Bolt  
**Team Members:** Harsh Bhati, Shashvat Sharma, Krishnam Bhanwarlal Digga — Indian Institute of Technology (IIT), Indore  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
We resolve every Source-1 business against Source-2/3 records with a cascade: multi-pass blocking (exact keys + TF-IDF character-trigram top-K in both directions inside state blocks, re-ranked by a learned candidate-priority model), 98 language-agnostic pair features, a calibrated LightGBM matcher (stage 1), fine-tuned multilingual **transformer cross-encoders** that read the raw name + address text of the uncertain pairs, a **collective stage-2 LightGBM** that re-scores each pair knowing its competitors (other candidates of the same S1, other S1 claiming the same record, the cross-encoder verdicts), a label-free adaptation to the test set's distractor density, and an exclusivity-aware per-entity expected-F0.5 list selection. The stage-1 matcher doubles as the last blocking filter, so the final candidate set is only **5.02 pairs per S1** (8.69M test pairs, 11.8× fewer than blocking alone) at no measurable cost. Out-of-fold macro F0.5 on train is **0.98870** (331,205 held-out S1 incl. singletons) and **0.98864** on an honest holdout of 330,649 train S1 that no model ever saw; public leaderboard progression 0.9680 → 0.9724 → 0.9815 → **0.9837**.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full data:
- **Size:** train 2.21M S1 / 5.03M S2 / 5.29M S3; test 1.73M S1 (India 810k, US 663k, **France 259k — absent from training**) / 4.89M S2 / 5.08M S3.
- **Links:** 7.64M train links, mean 3.46 matches per S1, only **5.58% singletons** (an empty list on a matched S1 costs a full 1.0 of that entity's F0.5, so recall matters despite β = 0.5).
- **Structure (diagnosed on train):** every S2/S3 record belongs to **at most one** S1 (0 violations); each matched S1 has 0-5 S2 and 0-6 S3 noisy copies (a star: copies resemble their S1 more than each other; names are noised per copy, addresses per source); unlinked records are single distractors. An empty-address record is a true copy of *some* S1 97.7% of the time, a DBA/alias name 99.98%, a domain-form name 95.7% — the difficulty is *which* S1.
- **Scripts:** in India ~24% of S2 and ~13% of S3 names are in Indic scripts (9 scripts) while S1 is always Latin.
- **Noise patterns:** legal-form changes (Inc/LLC/Pvt Ltd/SARL/"Et Fils"), token re-ordering, typos, bracketed junk, injected accents, "NULL"/"N/A", house-number edits (1804 → 1804A, 3412 → 3411), shuffled address components, abbreviations, domain/handle forms, alias names with the correct address, and **empty addresses** (3-4% of S2/S3).
- **Test shift (label-free diagnosis):** the test pools hold ~1.9× more distractors per S1 than train, concentrated in copy-like *hard negatives* — same name and street with the house number shifted by 3-100 or one digit changed by ±3..9 (pairs of that relation class per S1: 1.84× train in the US). France adds a concept shift: short generic names ("Club", "Societe", "SARL"), 29.8% of S1 sharing name + city with another S1, only 3 regions / 15 cities. This explained our first leaderboard gap (OOF 0.9815 vs LB 0.9680).

### 2.2 Solution Strategy
**Approach Type:** Hybrid — blocking + learned candidate priority + pairwise GBDT + fine-tuned transformer cross-encoders + collective (competition-aware) second-stage GBDT + label-free shift adaptation + exclusivity-aware per-entity decision.  
**Core Innovation:** (1) exploiting "each record has at most one owner" everywhere — competition features, a collective stage-2 model that sees every S1 claiming a record, and an exclusivity-aware expected-F0.5 decision; (2) cross-encoders applied only to the uncertain band of stage-1 probabilities (16% of the re-scored pairs, ~92% of the expected errors) and trained on train S1 that no other model uses, so stage 2 learns how much to trust them from honest out-of-fold scores; (3) synthetic test-like hard negatives (house number shifted) for the cross-encoders and a label-free density adaptation of the house-number-relation classes, both aimed at the diagnosed test shift; (4) a learned candidate-priority model and city-voted alternate blocks that recover links lost by the per-entity cap; (5) a native-script → Latin token dictionary learned from training links only.

---

## 3. Candidate Generation (Blocking)
Blocking runs per country (a hard partition) and per state/region block; the country value itself is never a feature, and France goes through the same code path.

- **Blocking keys used:**
  - Exact-key passes, country-wide, keys kept only when rare (≤ 30 pool records, ≤ 150 inside the state block): normalized name, the 3 rarest name tokens, house number × rare address word, consonant skeleton, glued/compact name (domain forms), normalized address; plus an **address-key** and a **street-name** pass with 5 reserved slots beyond the cap.
  - TF-IDF (hashed character 3-grams, own IDF, `sparse_dot_topn`) on name, address and name+address, **both directions**: S1 → pool top-20 and pool → S1 top-3. Pool records with an empty address form a fallback block queried by every S1.
  - City-voted alternate blocks (label-free): a pool record whose city is dominated on the S1 side by another state also joins that state's block.
  - Learned candidate priority: a 300-tree LightGBM on blocking signals only, fitted on TRAIN labels of 20% of each train country's S1 (whole state blocks, excluding the models' training S1), ranks each S1's union before keeping 60 candidates per S1 (+ up to 5 reserved exact-key slots).
  - **Learned filter (last candidate-generation step):** the stage-1 LightGBM matcher scores those ~59 candidates per S1 and keeps only pairs with calibrated p ≥ 0.001. Everything downstream — cross-encoders, stage-2 model, adaptation, decision — runs on this filtered set only, and it is exactly what `candidate_pairs.tsv` contains. On the OOF sample the filter keeps 1,129,964 of the 1,130,088 true links present among the 60 (99.99%), with no change in macro F0.5 (0.98870) and an oracle ceiling of 0.99536 (vs 0.99540 unfiltered).
- **Candidate pairs generated (`candidate_pairs.tsv`):** test **8,689,809 pairs = 5.02 per S1** (France 1,505,613 = 5.80 per S1, India 3,847,851 = 4.75, US 3,336,345 = 5.03), after blocking produced 102,281,392 pairs (~59 per S1) — a 11.8× reduction by the learned filter at no measurable cost. Against all same-country S1 × S2/S3 pairs (France 259k × 1.43M, India 810k × 4.72M, US 663k × 3.82M ≈ 6.7 × 10¹²) the reduction ratio is 99.9999%.
- **How we ensured true matches were not lost:** every change was measured as pair recall and as oracle F0.5 (perfect matcher on the candidates) on geo slices and then on the full training data. Full-train pair recall at 60 candidates: **US 0.9882, India 0.9799** (hand-made priority: 0.9861 / 0.9724); after the learned filter (5 per S1) 99.99% of those true links remain. Oracle macro F0.5 on the OOF population: **0.9954** — the ceiling for any matcher on these candidates. An error report lists every true link that never became a candidate, tagged by cause (empty address, alias, domain form, cross-state record, typo).

---

## 4. Matching Model

**Features used (98, float32, language-agnostic; stage 1):**
- Name: ratio / token-set / token-sort / partial / Jaro-Winkler on the normalized and core name, normalized Levenshtein, sorted-token key equality, glued-name ratio + containment, consonant skeleton, token Jaccard, IDF-weighted overlap and containment, legal-form agreement / conflict, legal-form-aware equality, acronym match, lengths.
- Address: ratio / token-set / partial, empty flags, house-number exact / base / numeric distance and the **house-number relation class** of the best number pair (exact, letter/digit added or dropped, ±1-2, shifted 3-10 / 11-100, one digit substituted by 1-2 or 3-9, …), street-only overlap, city and state agreement.
- Context: source (S2/S3), script, domain form, every blocking pass flag / score / rank and the learned priority; per-S1 ranks and gaps; **competition** over all S1 of the country (the record's best score over all S1, gap to the best other S1, number of S1 claiming it); group similarity to the S1's other top candidates; name ambiguity (log counts of S1 / pool records sharing a name key); percentile ranks.

**Stage 1 — LightGBM matcher (MIT):** 127 leaves, learning rate 0.1, trained on a 15% hash sample of train S1 (331,205 S1, 19.6M pairs, all negatives) with 5-fold **GroupKFold by S1** for honest out-of-fold (OOF) probabilities, isotonic calibration, final booster on all sampled S1 (607 rounds). OOF AUC 0.99995 — AUC saturates, so every decision was made on macro F0.5.

**Cross-encoders (transformers, fine-tuned by us):** input `"S1 name | S1 address" [SEP] "candidate name | candidate address"` (raw strings), one logit, binary cross-entropy, AdamW with linear warm-up/decay, length-bucketed batches, mixed precision. Trained only on pairs whose final-model stage-1 p is in [0.01, 0.99], from train S1 in hash buckets disjoint from the stage-1 sample; applied to the same band of the OOF sample and of the test set.
| name | backbone (license, params) | training data | notes |
|---|---|---|---|
| ce | intfloat/multilingual-e5-small, first 6 of 12 layers (MIT, 107.0M) | 236k pairs (15% of train S1), 1.5 epochs | CPU-trained; val AUC 0.916 vs stage-1 0.942 on the same pairs |
| ce2 | BAAI/bge-reranker-v2-m3 (Apache-2.0, 567.8M) | 1.13M pairs (70% of train S1) + 32,913 synthetic hard negatives, 1 epoch, Kaggle 2 × T4 (5.2 h incl. scoring) | word-embedding table frozen; val AUC **0.966** vs stage-1 0.944 on the same pairs; ranks the true pair above its shifted-number twin 99.2% |

A third cross-encoder (intfloat/multilingual-e5-large, MIT, 560M, val AUC 0.965) added nothing on top of ce + ce2 (OOF +0.00004) and is not used.
Synthetic hard negatives: for 25% of the true pairs whose addresses share a house number, the candidate's number is shifted by 3-100 or one digit is changed by ≥ 3 and the pair is added as a non-match — exactly the distractor type found ~1.9× more often in test.

**Stage 2 — collective LightGBM:** re-scores every pair with stage-1 p ≥ 1e-3 (1.50M OOF rows) from the stage-1 features + within-S1 context of the stage-1 probabilities (rank, gaps, number above 0.5/0.2, sums) + the stage-1 decision (selected?, optimal list length, empty-list probability) + competition over ALL train S1 claiming the record + sibling similarities to the S1's other likely copies + the cross-encoder logits with their within-S1 rank and gaps. Same frozen folds, isotonic calibration. Guard G1: pairs whose house numbers disagree may only go *down* versus stage 1.

**Threshold selection method:** no global threshold. Probabilities are adjusted for exclusivity (soft at-most-one-owner posterior over the S1 claiming each record); for each S1 the list length maximising the **expected F0.5** (with a Poisson term for true links blocking missed and an empty-list bias) is chosen; the decision parameters are picked by OOF macro F0.5, and a repair step guarantees each record goes to at most one S1. Before the decision, the **label-free shift adaptation** re-estimates, per country and house-number-relation class, the prior of the hard classes from the density of such pairs per S1 in the test pool (odds factor capped at 4, only downward); after it, a targeted rule drops a shifted-number link when the same S1 already has an exact-number link and p < 0.99. France uses the pooled parameters.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, OOF on train, 331,205 S1 incl. singletons; same folds throughout):**
| version | change | OOF | holdout (330,649 S1) | public LB |
|---|---|---|---|---|
| v3 | 72 features, stage 1 only | 0.98152 | 0.98189 | 0.968038 |
| v3 + stage 2 (G1) | collective re-scoring | 0.98360 | 0.98401 | 0.969604 |
| + density adaptation | label-free shift prior | — | — | 0.971152 |
| + targeted rule | | — | — | 0.972409 |
| v4 stage 1 | 98 features, address-key / street passes, France normalization fixes | 0.98473 | 0.98506 | — |
| v4 + stage 2 | | 0.98569 | 0.98592 | — |
| v4 + stage 2 + ce | 6-layer e5-small cross-encoder | 0.98752 | 0.98755 | **0.981499** |
| v5 | + ce2: bge-reranker-v2-m3 cross-encoder (4.9× more training data, synthetic hard negatives) | **0.98870** | **0.98864** | **0.983714** |
| **final (submitted)** | v5 + learned candidate filter (stage-1 p ≥ 0.001): 5.02 candidates per S1 instead of ~59; 6 of 5.8M links differ from v5 | **0.98870** | — | final upload |
The cross-encoder moved the leaderboard ~5× more than the OOF (+0.0091 vs +0.0018): it mainly repairs errors caused by the test shift, which the OOF does not contain. Attribution of v4: new features +0.0030, blocking/normalization +0.0002 (v3 features on v4 candidates: 0.98170).
- **Unseen country check (no labels):** links per S1 on test — France 3.37, India 3.34, US 3.37 (OOF predictions: India 3.346, US 3.359; train truth 3.46); empty lists 5.5% / 5.8% / 5.8% (train singletons 5.58%). India and US test behave like OOF, so the remaining leaderboard gap (OOF 0.9887 vs LB 0.9837) points to France (~0.958 implied): its errors are wrong assignments inside clusters of same-named businesses (29.8% of France S1 share name + city with another S1; on train such S1 score 0.980-0.982 vs 0.989-0.991 for the rest).
- **Where the remaining OOF loss is:** oracle on the candidates 0.9954 (17,387 of 1,147,475 true links never become candidates); in-candidate misses 20,506 (61% of them empty-address copies) vs only 1,257 false links (precision 0.9989); 95% of the fixable loss sits in the uncertain band the cross-encoders cover.
- **Common false positives (wrong merges):** records whose true owner is another S1 with a near-identical name; copy-like distractors with a shifted house number (4.1× lift among false merges; the target of the synthetic negatives, the adaptation and the rule); address-less duplicates of generic names.
- **Common false negatives (missed matches):** candidates with an **empty address** (57-72% of model misses in the error reports — name-only evidence, several S1 share the name), alias names with the correct address (4-8× lift), house-number noise, and true links never generated as candidates (the 0.0046 gap between 0.9954 and 1).

---

## 6. Conclusion
A carefully measured blocking stage, competition-aware GBDT scoring, fine-tuned multilingual cross-encoders on the uncertain pairs, a collective second stage and an exclusivity-aware expected-F0.5 decision reach 0.98870 OOF macro F0.5 (0.983714 on the public leaderboard). The biggest gains came from (1) modelling the at-most-one-owner structure, (2) diagnosing the test shift without labels and targeting it (synthetic hard negatives, density adaptation), and (3) letting a transformer read the raw strings of only the uncertain pairs — cheap enough for 100M candidate pairs. The remaining loss is dominated by address-less records whose name is shared by several businesses and by the 1-2% of true links that never become candidates (worth 0.0046 of macro F0.5).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run_pipeline.py` — single entry point; stages `prepare → prio → block → features → train → crossenc → predict → write`, each resumable; `--splits/--countries` shard a stage across machines; `candidate_prune_p1` (config) is the learned last blocking filter.
- `src/ber/` — `io.py` (safe TSV reading, parquet cache), `normalize.py`, `normalize_frame.py` (transliteration, normalization, native-token dictionary), `blocking.py` (candidate generation, learned priority), `features.py`, `model.py` (LightGBM + calibration), `stage2.py` (collective model), `crossenc.py` (cross-encoders: recipe, synthetic negatives, scoring, features), `adapt.py` (shift adaptation, targeted rule), `decide.py` (exclusivity + expected-F0.5), `metric.py` (exact macro F0.5), `outputs.py`, `config.py`, `memory_guard.py`.
- `configs/submission_v5.json` — the configuration of the submitted outputs; `tools/check_submission.py`; `models_manifest.json`; `README.md`; `requirements.txt`.

Reproduce:
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python src/run_pipeline.py --data-dir <dataset> --work-dir <work> --out-dir <output> --config configs/submission_v5.json --stage all --n-jobs 14
```
CPU stages were run on Modal CPU containers (16-32 cores, 64-128 GB RAM); the large cross-encoders on Kaggle GPUs (2 × T4) and the small one on a 32-core CPU. Cross-encoder fine-tuning on GPUs is not bit-deterministic; every other stage is.

### B. Additional Results
**Fair play.** No external databases, APIs, geocoders, LLM services or internet data were used to resolve entities. The only downloads are the open-weights pretrained backbones (MIT / Apache-2.0), which are fine-tuned on the provided training data only. All learned artifacts (native-token dictionary, candidate priority, matchers, cross-encoders, adaptation priors) come from the provided data via the submitted code; test data is used only label-free (candidate generation, competition features, density adaptation). Cloud CPUs and Kaggle GPUs were used purely as compute.

**Licenses** (library and model audit):

| component | version | license |
|---|---|---|
| polars | 1.44.2 | MIT |
| pyarrow | 25.0.1 | Apache-2.0 |
| numpy | 2.2.6 | BSD-3-Clause |
| scipy | 1.15.3 | BSD-3-Clause |
| scikit-learn | 1.7.2 | BSD-3-Clause |
| rapidfuzz | 3.14.5 | MIT |
| lightgbm | 4.7.0 | MIT |
| sparse-dot-topn | 1.2.0 | Apache-2.0 |
| psutil | 7.2.2 | BSD-3-Clause |
| torch | 2.5.1 | BSD-3-Clause |
| transformers / tokenizers / safetensors | 4.46.3 / 0.20.3 / 0.4.5 | Apache-2.0 |

| model | kind | license | parameters |
|---|---|---|---|
| master-bolt/lightgbm-matcher (stage 1) | LightGBM, 607 rounds | MIT (library), trained by us | 153,571 |
| master-bolt/lightgbm-stage2 (final probabilities) | LightGBM | MIT (library), trained by us | 30,866 |
| master-bolt/lightgbm-candidate-priority | LightGBM, 300 trees | MIT (library), trained by us | 37,500 |
| ce: fine-tuned intfloat/multilingual-e5-small (6 layers) | transformer cross-encoder | MIT | 107,007,361 |
| ce2: fine-tuned BAAI/bge-reranker-v2-m3 | transformer cross-encoder | Apache-2.0 | 567,755,777 |
