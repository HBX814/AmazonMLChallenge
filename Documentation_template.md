# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Master Bolt  
**Team Members:** Harsh  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary
We resolve every Source-1 business against Source-2/3 records with a four-stage pipeline: multi-pass blocking (exact keys + TF-IDF character-trigram top-K in both directions inside state blocks, re-ranked by a learned candidate-priority model), 72 language-agnostic pair features including competition and name-ambiguity signals, a calibrated LightGBM matcher trained with GroupKFold by S1, and an exclusivity-aware per-entity expected-F0.5 list selection. Out-of-fold macro F0.5 on the training data is **0.9815** (US 0.9836, India 0.9784; 331,205 held-out S1 including singletons), and the predicted-links-per-entity rate on the unseen country (France) matches the train rate.

---

## 2. Methodology

### 2.1 Problem Analysis
Measured on the full training data:
- **Size:** train 2.21M S1 / 5.03M S2 / 5.29M S3; test 1.73M S1 (India 810k, US 663k, **France 259k — absent from training**) / 4.89M S2 / 5.08M S3.
- **Links:** 7.64M train links, mean 3.46 matches per S1, only **5.58% singletons** (an empty list on a matched S1 costs a full 1.0 of that entity's F0.5, so recall matters despite β = 0.5).
- **Exclusivity:** every S2/S3 record belongs to **at most one** S1 (0 violations in 7.64M links); S2 and S3 each contain several noisy duplicates of the same entity; ~25% of S2/S3 records are unlinked distractors. Country never crosses between linked records.
- **Scripts:** in India ~24% of S2 and ~13% of S3 names are written in Indic scripts (9 scripts) while S1 is always Latin.
- **Noise patterns seen:** legal-form changes (Inc/LLC/Pvt Ltd/SARL), token re-ordering, character typos ("Gdmarf" for "Graf"), bracketed junk ("[[Group]]"), accents injected, "NULL"/"N/A" tokens, house-number edits (1804 → 1804A, 3412 → 3411), shuffled address components ("DC, WASHINGTON, 3080 STANTON RD"), abbreviations (R/BD for Rue/Boulevard), domain/handle forms, gibberish alias names with the correct address, and **empty addresses** (3.6% of US and ~3% of India S2/S3 records).
- **Geography:** every non-empty address carries a state (US, India) or region (France: only 3 regions in test). 1.2% of India train links point to a record in a *different* state block — 97% of those are Hyderabad records still labelled "Andhra Pradesh" whose S1 is in Telangana. Test pools are ~23% denser per S1 than train pools.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — blocking + learned candidate priority + pairwise GBDT classifier + constrained (exclusivity-aware) per-entity decision.  
**Core Innovation:** (1) competition features and an exclusivity-aware expected-F0.5 decision that exploit "each record has at most one owner"; (2) a learned candidate-priority model plus label-free city-voted alternate blocks that recover true links lost by the per-entity candidate cap (India blocking recall 0.9724 → 0.9796); (3) label-free name-ambiguity counts that tell a unique address-less duplicate from a generic name (+0.0016 OOF F0.5); (4) a native-script → Latin token dictionary learned from training links only.

---

## 3. Candidate Generation (Blocking)
Blocking runs per country (a hard partition) and per state/region block; the country value itself is never used as a feature, and unseen countries (France) go through exactly the same code path.

- **Blocking keys used:**
  - Exact-key passes, country-wide, keys kept only when ≤ 30 pool records carry them (≤ 150 inside the state block): normalized name, the 3 rarest name tokens, house number × rare address word, consonant skeleton, glued/compact name (domain forms), normalized address.
  - TF-IDF (hashed character 3-grams, own IDF, frequent-n-gram query pruning, `sparse_dot_topn`) on name, address and name+address, **both directions**: S1 → pool top-20 and pool → S1 top-3 (the reverse direction exploits exclusivity). Pool records with an empty address form a fallback block queried by every S1 (top-5).
  - City-voted alternate blocks (label-free): a pool record whose city is, on the S1 side of the same split, dominated (≥ 80% of ≥ 20 S1) by another state also joins that state's block.
  - Learned candidate priority: a 300-tree LightGBM on the blocking signals only (pass flags, cosines, ranks, within-S1 context), fitted on TRAIN labels of 20% of each train country's S1 (whole state blocks, excluding the matcher's training S1), ranks each S1's candidate union before keeping **60 candidates per S1**.
- **Candidate pairs generated:** test 102,218,920 (US 38.9M, India 48.1M, France 15.2M; 59.0 per S1); train 130.2M (US 77.8M, India 52.4M).
- **How we ensured true matches were not lost:** every change was measured as pair recall and oracle F0.5 (perfect matcher on the candidates) on geographic dev slices and then on the full training data. Full-train pair recall at 60 candidates/S1: **US 0.9881, India 0.9796** (hand-made priority: 0.9861 / 0.9724; uncapped India union 0.9833 at 210 candidates/S1). Blocking ceiling macro F0.5 on the OOF population: 0.9953. The error report lists every true link that was not a candidate, tagged by cause (empty address, cross-state record, typo, alias).

---

## 4. Matching Model

**Features used (72, all float32, language-agnostic):**
- Name features: ratio / token-set / token-sort / partial / Jaro-Winkler on the normalized and core name, normalized Levenshtein, sorted-token key equality, glued-name ratio + containment, consonant-skeleton ratio (transliteration vowel loss), token Jaccard, IDF-weighted overlap and containment, legal-form agreement / conflict, lengths.
- Address features: ratio / token-set / partial on the normalized address, empty-address flag, house-number exact / base-digit / numeric distance, street-word Jaccard, city and state agreement with unknown flags.
- Other: record source (S2/S3), non-Latin script, domain form; every blocking pass flag / score / rank and the learned priority; per-S1 context (rank by a base score, gap to the S1's best, number of candidates); competition over all S1 of the country (the record's best score over all S1, is-best, gap to the best other S1, number of S1 claiming it); group similarity to the S1's other top candidates (duplicates resemble each other); name ambiguity (log count of S1 and of pool records sharing the candidate's / the S1's normalized name key).

Text normalization: Indic scripts are transliterated first (rule-based transliterator + a native-token dictionary of 1,347 entries learned from 163,001 train link pairs), then accents are folded; legal forms, honorifics, NULL tokens and bracketed junk are removed; street types, directionals and state names are canonicalized with hand-written maps.

**Model type:** LightGBM binary classifier (MIT), 127 leaves, learning rate 0.1, early stopping on an inner S1-group split, trained on a 15% hash sample of train S1 (331,205 S1, 19.5M pairs) with all negatives; 5-fold **GroupKFold by S1** gives honest out-of-fold probabilities, which are isotonically calibrated; the final booster (826 trees, 208,978 parameters) is trained on all sampled S1.  
**Threshold selection method:** no global threshold. Probabilities are adjusted for exclusivity (soft at-most-one-owner posterior p/(1−p) normalised over the S1 claiming each record), then for each S1 the list length maximising the **expected F0.5** (with a Poisson term for true links blocking missed and an empty-list bias) is chosen; the decision parameters (exclusivity mode, empty-list bias, vs. plain thresholds) are picked by OOF macro F0.5 over all train S1 in the sample, and a final repair step guarantees each record is given to at most one S1. The same parameters are applied to France.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98152** out-of-fold on train (331,205 S1 incl. singletons; US 0.98364, India 0.97836; micro precision 0.9954, micro recall 0.9559). Progression on the same folds: v1 0.98006 (hand-made priority) → v2 0.98134 (learned priority + city-voted blocks; 661,854-S1 population) → v3 0.98152 (+ name-ambiguity features, cheaper training; its A/B baseline without them was 0.97997). Leave-one-country-out (geo slices, France proxy): India→US 0.9545 vs 0.9750 in-country, US→India 0.9053 vs 0.9708; the decision parameters transferred almost perfectly (the loss is model shift), which is why the final model is trained on both countries and uses only language-agnostic features.
- **Unseen country check (no labels):** predicted links per S1 on test — US 3.42, India 3.33, France 3.39 (train truth 3.46, OOF predictions 3.33); empty lists — US 5.5%, India 5.7%, France 5.2% (train singletons 5.58%); France's score distribution matches the OOF distribution.
- **Common false positives (wrong merges):** records whose true owner is another S1 with a near-identical name; address-less duplicates with a legal-form-only difference; house-number mismatches with otherwise identical names and streets (house_number_mismatch has 4.1× lift among false merges vs correct links).
- **Common false negatives (missed matches):** candidates with an **empty address** (57% of model misses and 55% of blocking misses in the v1 error report — name-only evidence), gibberish alias names with the correct address (5.7× lift), house-number noise (3.2×), and cross-state records the geo blocks do not reach. Loss decomposition (v2 OOF): decision 0.0127, blocking 0.0047, ranking 0.0013 of F0.5.

---

## 6. Conclusion
A carefully measured blocking stage plus competition-aware GBDT scoring and an exclusivity-aware expected-F0.5 decision reach 0.9815 OOF macro F0.5 with a 209k-parameter model that runs on CPUs only. The biggest gains came from features and decisions that exploit the at-most-one-owner structure and from measuring at full scale rather than on slices (full-scale OOF was 0.003–0.006 above the geo-slice estimates — US 0.9768 → 0.9827, India 0.9726 → 0.9761 — consistent with the competition features seeing every S1, while full-scale blocking recall was lower than on slices and needed its own fixes); the remaining loss is dominated by address-less records whose name is shared by several businesses.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
- `src/run_pipeline.py` — single entry point; stages `prepare → prio → block → features → train → predict → write`, each resumable; `--splits/--countries` shard a stage across machines.
- `src/ber/` — `io.py` (safe TSV reading, parquet cache with row-count verification), `normalize.py`, `normalize_frame.py` (transliteration, normalization, native-token dictionary), `blocking.py` (candidate generation, learned priority), `features.py`, `model.py` (LightGBM + calibration), `decide.py` (exclusivity + expected-F0.5), `metric.py` (exact macro F0.5), `outputs.py` (TSV writers), `config.py`, `memory_guard.py`.
- `configs/submission_v3.json` — the configuration that produced the submitted outputs; `tools/check_submission.py` — streaming output validator; `models_manifest.json`; `README.md`; `requirements.txt`.

Reproduce:
```bash
cd code/business_entity_resolution
pip install -r requirements.txt
python src/run_pipeline.py --data-dir <dataset> --work-dir <work> --out-dir <output> --config configs/submission_v3.json --stage all --n-jobs 14
```
Measured on Modal CPU containers (16–32 cores, 64–160 GB RAM; the five split/country shards of prepare/block/features run in parallel): prepare 10 min, prio 9 min, block ≤ 14 min per shard, features ≤ 12 min per shard, train ~27 min, predict 15 min, write 1 min; peak memory 20–40 GB per shard.

### B. Additional Results
**Fair play.** No external databases, APIs, geocoders, LLM services or internet data were used. All learned artifacts (native-token dictionary, candidate-priority model, matcher) are derived from the provided training data by the submitted code. Hand-written normalization maps encode general domain knowledge (legal-form and street-type abbreviations, state and region names). Cloud CPU containers were used only as compute.

**Licenses** (from `audit_compliance.py`, RESULT: PASS):

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

| model | kind | license | exact parameters |
|---|---|---|---|
| master-bolt/lightgbm-matcher (final model) | LightGBM, 826 trees | MIT (library), trained by us | 208,978 |
| master-bolt/lightgbm-candidate-priority | LightGBM, 300 trees | MIT (library), trained by us | 37,500 |

**Experiment log (OOF macro F0.5 on train):**

| version | change | overall | US | India |
|---|---|---|---|---|
| slices | Illinois / Karnataka geo slices, cap 30 → 60 | — | 0.9750 → 0.9768 | 0.9708 → 0.9726 |
| v1 | full scale, 60 candidates/S1, 30% train sample | 0.98006 | 0.98272 | 0.97607 |
| v2 | + learned candidate priority + city-voted alternate blocks | 0.98134 | 0.98313 | 0.97868 |
| v3 A/B | baseline (15% sample, lr 0.1) → + name-ambiguity features | 0.97997 → **0.98152** | 0.98364 | 0.97836 |
