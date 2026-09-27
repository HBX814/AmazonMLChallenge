# Experiments log — Master Bolt ER (append-only; newest at the bottom)

Rules: one change per entry · same frozen folds (5-fold, seed 0) · numbers are OOF on train unless marked LB (leaderboard) · never delete entries · note RAM/runtime and the machine.

| # | date | change (one thing) | config diff | blocking pair recall / cands per S1 | OOF macro F0.5 all / US / India | LOCO US→IN / IN→US | test links/S1 (US/IN/FR) | LB | runtime / peak RAM / machine | keep? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 2026-09-25 | baseline, Karnataka geo slice (58,993 train S1, full same-country pool; prior session) | defaults; soft excl + expected-F, empty_bias 2 | 0.990 union / 0.977 @cap30 | 0.9772 / — / 0.9772 (oracle 0.9931) | — | — | — | blocking 288 s, 1.63 GB peak / laptop | yes |
| 1 | 2026-09-26 | parquet cache on full real data (gate: rows = wc -l − 1) | — | — | — | — | — | — | 46 s total / laptop | yes |

| 2 | 2026-09-26 | full-data prepare (cache + native dict + normalized frames, all splits/countries) | defaults, --n-jobs 14 | — | — | — | — | — | 580 s / 10.7 GB / Modal 16c | yes |
| 3 | 2026-09-26 | blocking measured on US_IL (75,633 S1, pool 562,562) and IN_KA (69,984 S1, pool 440,919) geo slices, pipeline-normalized | default BlockingConfig | US_IL union 0.9932 @126.7 · cap30 0.9824 · cap60 0.9900 · cap80 0.9915; IN_KA union 0.9885 @206.9 · cap30 0.9723 · cap60 0.9810 · cap80 0.9824 | oracle@cap30 US 0.9945 / IN 0.9915; @cap60 0.9970 / 0.9942 | — | — | — | blocking 34 s (3.3 GB) / 51 s (4.5 GB) / Modal 16c | yes |

| 4 | 2026-09-26 | matcher OOF on slices, cap 30 (features -> LGBM 5-fold GroupKFold-by-S1 -> isotonic -> decision grid) | cap 30; grid thr x {expected_f} x excl {none,soft} x bias | US_IL 0.9824 / IN_KA 0.9723 | US_IL **0.9750** (soft, bias 2; oracle 0.9945, AUC 0.99973) · IN_KA **0.9708** (oracle 0.9915, AUC 0.99963) | — | — | — | 251 s / 336 s / Modal 16c | yes |
| 5 | 2026-09-26 | leave-one-country-out, cap 30 (model + calibrator + decision from train slice only) | — | — | — | US→IN **0.9053** (own 0.9708, gap 0.066, AUC 0.9977) · IN→US **0.9545** (own 0.9750, gap 0.021, AUC 0.9991) | — | — | ~3 min / Modal 8c | info |
| 6 | 2026-09-26 | per-S1 candidate cap 30 -> 60 (slices) | blocking.max_cands_per_s1 60 | US_IL 0.9900 @58.3 / IN_KA 0.9810 @59.4 | US_IL **0.9768** (+0.0018; oracle 0.9970, AUC 0.99983) · IN_KA **0.9726** (+0.0018; oracle 0.9942, AUC 0.99979) | — | — | — | 324 s / 440 s / Modal 16c | **yes** (submission_v1: cap 60) |
| 7 | 2026-09-26 | FULL-scale blocking (all S1, full pools), submission_v1 (cap 60) | configs/submission_v1.json | **train/US 0.9861 @58.8 (77.8M pairs) · train/India 0.9724 @59.4 (52.4M)**; test US 38.9M (58.6/S1), India 48.1M, France 15.2M (58.5/S1) | — | — | — | — | block 522 s (26 GB) / 839 s (23 GB) / 367 s / 700 s / 193 s, 5 parallel Modal 16c | yes |
| 8 | 2026-09-26 | **city-voted alternate blocks** (blocking.city_alt_blocks: pool row joins the state block its city votes for, city->state from same-split S1, n>=20, share>=0.8) — slice IN_TG = Telangana S1 (55,915), pool telangana ∪ andhra pradesh ∪ empty | city_alt_blocks false -> true | union 0.9856 -> 0.9901; **cap60 0.9557 -> 0.9786** (50,771 pool rows gained an alt block) | IN_TG **0.9628 -> 0.9711 (+0.0083)**; oracle 0.9849 -> 0.9931; AUC 0.99981 both | — | — | — | 475 / 487 s / Modal 16c | **yes (v2)** |
| 9 | 2026-09-26 | learned candidate priority (LGBM on blocking columns, fit on 50% of slice S1, eval other 50%; re-rank within saved top-80) | hand prio -> learned | US_IL @30 0.9824->**0.9900**, @60 0.9898->0.9910 · IN_KA @30 0.9723->0.9788, @60 0.9810->0.9817 · IN_TGA @30 0.9702->0.9763, @60 0.9784->0.9793 | — (lower bound; full union headroom India ~1 pt at cap 60) | — | — | — | 77 s / Modal 16c | **yes (v2)** |
| 7b | 2026-09-26 | **v1 FULL OOF** (submission_v1: cap 60, train_s1_frac 0.3 -> 661,854 S1 / 39.05M pairs / 2.25M pos; 5-fold LGBM lr 0.05, best_iter 2349-2873, final 2901 rounds) | configs/submission_v1.json | train US 0.9861 / India 0.9724 | **0.98006 / US 0.98272 / India 0.97607** (soft excl + expected-F, bias 2.0; none+bias2 0.97982; AUC 0.99994; micro P 0.9956 R 0.9510) | — | **US 3.42 / IN 3.30 / FR 3.35** (empty 5.55 / 5.80 / 5.27%) | (v1 = output/matching_results.tsv, validated) | train 6801 s / 40 GB / Modal 32c | yes |
| 10 | 2026-09-26 | **v2 full-scale blocking** = v1 + city_alt_blocks + learned prio (stage prio: 20% of each train country's S1 by whole geo blocks, main-train S1 excluded; 55.1M rows, 1.08M pos, 143 s) | configs/submission_v2.json | **train/US 0.9861 -> 0.9881 · train/India 0.9724 -> 0.9796** (same 58.8 / 59.4 cands/S1); test US 38.9M, India 48.1M, France 15.2M pairs | **0.98134 (+0.0013) / US 0.98313 (+0.0004) / India 0.97868 (+0.0026)** same 661,854-S1 OOF population as v1; soft excl + expected-F bias 2.0; lam_missing 0.0667 -> 0.0528; AUC 0.99993 | — | **US 3.42 / IN 3.32 / FR 3.37** (empty 5.57 / 5.69 / 5.27%) | (v2 = output/matching_results.tsv, all validators PASS) | prio 513 s; block 756 / 852 / 442 / 837 / 158 s; train 5511 s | **yes** |
| 11 | 2026-09-26 | **name-ambiguity features** (amb_s1_c/q, amb_pool_c/q = log1p #S1 / #pool records sharing the normalized name key of cand / S1; label-free, per split/country) — A/B on v2 candidates, cheap training (train_s1_frac 0.15 -> 331,205 S1 / 19.5M pairs, lr 0.1), same folds | 68 -> 72 features | (v2 blocking) | **A 0.97997 -> B 0.98152 (+0.0016)**; AUC 0.99992 -> 0.99993 | — | — | — | add_amb 3 min; 2 x ~27 min / Modal 32c | **yes (v3)** |
| 12 | 2026-09-26 | **v3 submission** = v2 blocking + 72 features, matcher = A/B model B (train_s1_frac 0.15, lr 0.1, 826 trees, 208,978 params) | configs/submission_v3.json (+ prio_exclude_frac 0.3 so a from-scratch run reproduces v2's prio model) | ceiling 0.99531 | **0.98152 / US 0.98364 / India 0.97836** (331,205 S1; oracle-cut 0.99398; micro P 0.9954 R 0.9559) | — | **US 3.43 / IN 3.33 / FR 3.39** (empty 5.52 / 5.69 / 5.19%) | (v3 = output/matching_results.tsv; all validators PASS) | predict+write+validate ~25 min / Modal 16c | **yes (current submission)** |
| 14 | 2026-09-26 | **stage-2 "collective" LightGBM** on stage-1 p (diag, not in pipeline): rescores pairs with p1 >= 1e-3 (1.55M of 19.5M OOF pairs, 1,129,774 of 1,129,911 positives) with 72 feats + within-S1 p1 context + stage-1 decision + p1-competition over ALL train S1 + sibling sims; same frozen folds, lr 0.1, 127 leaves, isotonic | work/infra/diag_stage2_*.py; decision soft + expected-F, **empty_bias 1.5** | (v3 blocking; ceiling 0.99531) | **0.98411 (+0.00259) / US 0.98586 (+0.0022) / India 0.98149 (+0.0031)**; honest holdout HB (buckets 150-299, final models): 0.98189 -> **0.98452 (+0.00263)**; guarded G1 (hn-mismatch pairs may only go down): OOF 0.98361 (+0.0021), HB 0.98397 (+0.0021) | — | G0 US 3.49 / IN 3.37 / FR 3.44; **G1 US 3.44 / IN 3.35 / FR 3.42** | not uploaded (files validated: /vol/diag/stage2/matching_results_stage2.tsv, matching_results_s2g1.tsv) | train 133 s (5 folds) + 24 s final; test apply 253 s all countries, 20 GB peak / Modal 16c | **candidate (G1)** |

## Notes per entry
### #12
- 5,851,233 links, 102,218,920 candidate pairs. check_submission PASS (--check-ids), official validator PASS (NONE + --check-ids) and PASS (with candidate file). v2 files archived in work/submissions/v2/.
- Compliance audit PASS (work/compliance_report.md); models_manifest.json lists matcher (208,978 params) + prio model (37,500), both LightGBM/MIT.
- Modal spend by 12:10 UTC: $22.61 metered, all covered by credits (billed $0.00).
- Zip built 12:30 UTC: submission/Master_Bolt_submission.zip (602,444,478 B, 21 files; all package gates G1-G7 PASS; CRC re-checked). It contains the code that produced the v3 outputs.
### LB 1 (reported by user, 2026-09-26)
- **v3 submission: leaderboard 0.968038** vs train OOF 0.98152 (gap 0.0135). Top team: 0.9907.
- If US/India test = OOF, France would be ~0.90 (0.968 - 0.38*0.9836 - 0.47*0.9784 = 0.134 -> /0.15).
- Hypothesis H-density: pool records per S1 — train US 4.68 / India 4.68; test US 5.76 / India 5.82 / France 5.53 (+23% everywhere), while predicted links/S1 are ~3.4 on both -> test has ~1.1 extra distractors per S1 the model never trained against (possibly copies of S1 removed from the test S1 file = orphans). Would depress US/India too, not only France. Test on train: drop ~19% of train S1 (their copies become orphan distractors, density 4.68 -> 5.75), re-block/featurize, measure OOF drop; fix = train under test-like density. Diagnostics launched: France deep-dive, data-structure/leak audit, stage-2 collective prototype (workflow wf_03d3c7ef-dc9).
### Diagnostics after LB 1 (workflow wf_03d3c7ef-dc9; full text work/reports/diag_results.md)
- No row-order / id leak (AUC 0.4993-0.5006). HB holdout (330,649 train S1, buckets 150-299, final-model p): stage-1 0.98189 vs OOF 0.98152 -> OOF honest; LB gap = test shift.
- TEST SHIFT: copies per S1 unchanged (name+hn+state key 1.543 test US vs 1.544 train) but test linked share of pool ~0.60 vs 0.74 -> ~2.3 distractors/S1 vs 1.22. Excess concentrated in HARD house-number relations (same name/street/city, number shifted 3-100 or one digit changed by 3-9; train P_true 0.02-0.13): selected HARD links 0.072/S1 test US vs 0.021 OOF; same-name+city HARD pairs 1.84x train density (US), 1.12x (India). sum p per S1 test US 3.564 vs OOF truth-among-cands 3.426 -> ~0.05-0.09 FP/S1 on test US.
- France: label-free estimators put test US ~0.977-0.978, India ~0.976-0.977, France ~0.969 -> LB implies France ~0.91-0.93. France-only feature PSI: hn_diff 1.28, idf_cand 0.66, st_inter 0.55, a_tset 0.49. Specific: tied full-name copies (legal forms / fils / freres stripped from name_core; empty-address unique-full-name copies 78-89% true but p~0.5, 5-9% selected), 29.8% of France S1 share name+city with another S1, 22.9% share an address key, acronyms, blocking misses same-address duplicates (0.0131/S1), normalization bugs (ch->chau, appt, '12 B').
- STAGE-2 collective (competition on stage-1 p over ALL train S1 + within-S1 + siblings + 72 feats): OOF 0.98152 -> **0.98411** (US 0.98586, India 0.98149), HB 0.98189 -> 0.98452; guard G1 (house-number-mismatch pairs may only go down) 0.98361 / 0.98397. Probe file output_probes/s2g1/matching_results.tsv (validated).
- Ruled out: decision tweaks on stage-1 p, name-key sibling propagation (harmful), pool-pool clustering (0), city swaps / accents / region map.
### v4 implementation (2026-09-26 18:03-18:25 UTC agents, interrupted by the weekly usage limit; continued inline)
- Code left by the agents (reviewed, all unit tests pass: normalize 42, frame 15, blocking 20, adapt 10, stage2 6, model 3, features 2):
  normalize.py (France: ch=chemin, appt/apt unit words, '12 B' -> 12b, alle), features.py (+26 v4 columns: hn_rel best-pair relation
  + one-hots + rdiff/ldiff + multiset; st2_* street-only overlap; norm_eq, amb_s1n_*, base_n + comp_n_*; acr_match; n_s1_akey_*;
  pct_* ranks; optional record flags), blocking.py (akey / stname exact-key passes, reserve 5 slots beyond the cap, prio backward
  compat), ber/stage2.py (collective model, guard G1), ber/adapt.py (label-free EM prior shift per house-number class + density + rule).
- run_pipeline integration (lead): config stage2 {enabled, guard, floor, empty_bias_grid}, adapt {method, p_min}; stage_train = stage-1
  + stage-2 (all-S1 stage-1 p via stage1_predict_stream, build_frame on the sampled S1, fit, grid) + adapt source; stage_predict =
  stage-1 -> stage-2 -> adapt -> decision.
- expB (stage2 module check on v3 artifacts): frame features equal to the prototype on 1.55M OOF / 1.55M HB rows except c_is_best on
  0.14% (deterministic tie-break now); stage-1 OOF 0.98152 / HB 0.98189 reproduced; prototype model via module: HB 0.98452 (G0).
### #14 stage-2 module + shift adaptation validation (expB_abc, v3 artifacts, same OOF / HB populations)
- ber/stage2.py own fit (G1): **OOF 0.98152 -> 0.98360** (US 0.98534 / IN 0.98099; eb 1.0), **HB 0.98189 -> 0.98401**; AUC on re-scored rows 0.99703 -> 0.99803, logloss 0.0579 -> 0.0470; fit 143 s. Test apply per country ~1-2 min, <= 13 GB; links agree with the prototype 99.7%.
- ber/adapt.py, HB with final-model p (no shift):   s1: none 0.98189 | em 0.98189 | em_hard 0.98190 | density 0.98185 | density_hard 0.98189 | rule 0.98158
                                                     s2: none 0.98401 | em 0.98400 | em_hard 0.98401 | density 0.98397 | density_hard 0.98400 | rule 0.98364
- Simulated test-like shift on HB (+0.8x HARD negatives per S1 drawn from OOF negatives):
  uniform draw  s1: none 0.98146 | em_hard 0.98168 | density_hard 0.98168 | rule 0.98140 ; s2: none 0.98368 | em_hard 0.98383 | density_hard 0.98383 | rule 0.98348
  **tilted draw (negatives that look like copies, p-weighted; HARD links/S1 US 0.056 ~ test US 0.072)**
                s1: none 0.97105 | em 0.97104 | density_hard **0.97450** | rule **0.97690**
                s2: none **0.97570** | em 0.97602 | density_hard **0.97898** | rule **0.97947**
  -> EM cannot see copy-like negatives (label-shift assumption fails); density (pair counts per S1 per class) can. Stage-2 + G1 alone
     is +0.0047 more robust than stage-1 under the tilted shift. Default: density_hard (free without shift, +0.003 with it).
### v4 data stages (Modal, 2026-09-26 19:19-20:15 UTC) — DONE; training BLOCKED
- /vol/work_v4: prepare with France normalize fixes 626 s; prio (akey/stname flags) 55.1M rows 156 s; blocking train US 0.9882 @58.8
  (v3 0.9881), India 0.9799 @59.4 (v3 0.9796); test pairs US 38.90M, India 48.18M, France 15.20M (+8.4k vs v3); features 98 columns
  (train US / India / test US / India / France done: 686-848 s per train shard).
- **Modal workspace disabled 20:17 UTC: $30.00 free credits exhausted (billed $0.48 beyond).** Volume still readable. v4 train/predict
  (work/infra/jobs/v4_train_predict.txt, v4a_train.txt) not run. Small artifacts copied to work/artifacts/ (stage-2 model + priors,
  expB results, v4 prio model).
### LB probes (v3 stage-1 + stage-2 module; all pass the official validator with --check-ids)
| file | change | test links/S1 FR / IN / US | LB |
|---|---|---|---|
| output_probes/s2g1/ | stage-2 G1 (prototype) | 3.418 / 3.355 / 3.442 | **0.969604** (+0.0016 vs v3 0.968038) |
| output_probes/s2g1_density/ | + density_hard (HARD-class priors x0.25-0.49; FR diff_11_100 pairs 0.51/S1 vs 0.097 source) | 3.408 / 3.350 / 3.422 | **0.971152** (+0.0015) |
| output_probes/s2g1_rule/ | + rule (drop HARD links if S1 has an exact-hn link, p<0.99) | 3.400 / 3.340 / 3.399 | **0.972409** (+0.0013 vs density; best so far) |
- **v4 stage-1 (98 features, France normalize fixes, akey/stname blocking, retrained prio; same 331,205 S1 / folds / lr 0.1):
  OOF 0.98473 (v3 0.98152, +0.0032), AUC 0.99995, 607 final rounds, 1328 s** (soft excl + expected-F, bias 2.0, lam 0.0525).
- **v4 + stage-2 G1: OOF 0.98569** (+0.0010 over v4 stage-1; soft excl + expected-F, bias 1.5); re-scored rows 1.50M (US 922,883
  + India 574,607); adapt source priors from 19.56M OOF pairs.
- v4 test outputs (both pass the official validator with --check-ids; 102,281,392 candidate pairs):
  | file | adapt | links/S1 FR / IN / US | links | LB |
  |---|---|---|---|---|
  | output_probes/v4_density/ | density_hard | 3.47 / 3.35 / 3.41 | 5,880,533 | (pending) |
  | output_probes/v4_hsr_rule/ | density_hs + rule (rule dropped FR 2,793 / IN 7,552 / US 19,509 HARD links) | 3.46 / 3.34 / 3.38 | 5,843,534 | (pending) |
- 2026-09-27: new Modal workspace `sotaloss` ($30 credit); dataset re-uploaded to its volume mbolt-data (32 s); v4 rebuilt from raw
  (old workspace data not reachable). AWS: bucket master-bolt-mlc-343218218669 + role mlc-ec2-s3 created (empty, free), quota
  requests 32 vCPU pending; not used (user chose Modal).
### #13 determinism (2026-09-26 12:40 UTC)
- Clean-container reproduction (pinned requirements) on the mini set: pipeline runs end-to-end with submission_v3.json and passes the official validator, BUT two runs were not byte-identical (prio gain 65160.97 vs 65140.27; mini OOF 0.93866 vs 0.93876). Cause: polars group_by / left-join output order is not guaranteed -> row order fed to LightGBM and to ordinal tie-breaks varied.
- Fix: union sorted by (qi, pi); candidates sorted by (s1, cand); maintain_order="left" on the label / feature joins; stable tie-break (row index) in the group-feature top-N; training rows sorted by (s1, cand) before every LightGBM fit. Unit tests pass (blocking 12, features 2).
- After fix: two independent runs -> matching_results.tsv and candidate_pairs.tsv BYTE-IDENTICAL (cmp), prio gain identical (65164.65098248574), official validator PASS.
- Consequence: the submitted v3 outputs came from the pre-fix code (statistically equivalent re-runs, not byte-identical). Byte-exact reproducibility of the submission needs one full re-run with the fixed code (~$10 Modal, ~1.5 h) and a zip rebuild (G5 stale gate will otherwise fail, since src/ is now newer than output/).
### #11
- Motivated by the v1 full-OOF report: empty_address = 57% of model misses (e.g. unique "Evans Prime Renewable Partners" vs address-less "... LLC" p 0.58 not selected, while generic "Department of Veterans Affairs" correctly low).
- Mean amb_s1_c (log1p): test France 1.63, test India 1.60, test US 0.95, train India 1.72, train US 1.23 -> France not unusually ambiguous by this measure.
### #7b v1 submission (2026-09-26 11:00 UTC)
- Test: 1,732,544 rows, 5,803,752 links (3.35/S1), 102,190,086 candidate pairs (58.98/S1, max 60, 4 empty rows). check_submission PASS (official + strict, --check-ids); OFFICIAL validator PASS (--candidate NONE --check-ids) and PASS (with candidate_pairs.tsv); local re-validation of downloaded matching file PASS.
- Full-OOF ladder: blocking 0.0062 (ceiling 0.99377) · ranking 0.0012 (oracle-cut 0.99257) · decision 0.0125 -> decision still the largest (report work/reports/err_full_oof.md).
- France shift (label-free, vs pooled OOF): links/S1 3.35 vs 3.28/3.33, empty 5.3% vs 5.7-5.8%, max_p mean 0.951 vs 0.944 -> no collapse / false-merge signature. Only flag conflict_rate 0.0030 vs 0.0010, but test India shows the same 0.0029 -> denser test pools, not France-specific.
### #10 v2 submission (2026-09-26 ~12:00 UTC)
- Test: 5,830,369 links (v1 5,803,752), 102,218,920 candidate pairs; check_submission PASS (--check-ids), official validator PASS (NONE + --check-ids) and PASS (with candidate file); downloaded copy re-validated locally PASS. v1 files archived in work/submissions/v1/.
- Full-OOF ladder v2: blocking 0.0047 (ceiling 0.99533; v1 0.0062) · ranking 0.0013 (oracle-cut 0.99406) · decision 0.0127.
- all-S1 recall slightly flattering (~14% of train S1 were prio-model training rows); the OOF population (main 30% sample) is excluded from prio training.
### #8
- Baseline union already 0.9856 (country-wide key passes reach the "Andhra Pradesh"-labelled Hyderabad records) but those key-only hits get a low hand-made prio and are cut by the cap -> a LEARNED prio is the general lever (India full: union 0.9833 vs cap60 0.9724).
- Overall effect small (Telangana ~6% of India S1): ~+0.0003 expected overall; generic & label-free, no-op where cities are unambiguous.
### #7
- train/India FULL uncapped (measure_blocking, 883k S1): **union 0.9833 @209.7 cands/S1** (oracle 0.9949); cap curve 30 0.9608 · 60 0.9723 · 80 0.9749 · 100 0.9763 · 150 0.9789 · 200 0.9805 -> cap costs 1.1 pts, deep-ranked by the hand-made prio (learned prio = lever).
  Geo: true links same state block 0.9487, pool address empty 0.0392, **OTHER state block 0.0120** (invisible on geo slices; TF-IDF can't reach, only country-wide key passes). -> lever: state-confusion edges learned from TRAIN links (query confused neighbour blocks); generic, no-op for unseen countries.
  key_hn alone = 133 cands/S1 (118M pairs) but +0.127 marginal recall. TF-IDF passes 74/194/269 s at full India scale.
- Cross-block links (infra/state_confusion.py, TRAIN links): India 36,786 (1.20%) of which **35,861 = S1 telangana -> pool "..., HYDERABAD, Andhra Pradesh"** (old state name; 18.5% of all Telangana links!). US 5,352 (0.12%): mostly **dc<->wa** (S2/S3 shuffle components: "DC, WASHINGTON, 3080 STANTON RD" -> last state-like comp = wa), de->oh (Delaware OH), or->oh.
  v2 idea (generic, label-free): extra block memberships for pool records = state inferred from city via a city->state map built from the SAME split's S1 (clean addresses; works for test/France) + every state-like component when several exist.
- Full-scale recall is BELOW the slices (US 0.9861 vs IL 0.9900; India 0.9724 vs KA 0.9810): big state blocks (Maharashtra 192k S1 / 870k pool, Delhi 121k, Texas 133k) make TF-IDF top-K and the per-S1 cap more competitive. India blocking / candidate prioritisation = next lever (generic, no per-country cap).
### #6 error analysis (work/reports/err_{US_IL,IN_KA}_cap60.md)
- Loss ladder US_IL / IN_KA: blocking 0.0030 / 0.0058 · ranking 0.0022 / 0.0027 · **decision 0.0180 / 0.0189** (true candidates ranked right but not selected).
- Biggest component: partial_recall_model 0.0113 / 0.0108 (micro P 0.994, micro R 0.943 on US_IL).
- Tag with the lift: **empty_address** = 72% (US) / 46% (IN) of FN_model (lift 85x / 33x) and 42% / 37% of FP. Candidate with empty address + same name gets p 0.5-0.8.
  Suspected partly a GEO-SLICE ARTEFACT: the slice pool contains ALL same-country empty-address records (222k US) but only the state's S1, so same-name S1 from other states (true owners) are absent from competition features. Full-scale features (all S1) should help; re-check on full OOF.
- Other lifts: house_number_mismatch (FN 2.3x US / 4.4x IN), alias_name (gibberish alias, same address; 4-4.8x). native_script NOT a driver (lift 0.5) -> transliteration + dict OK.
### #4
- Pred links/S1 3.26 vs true 3.46 (both). Decision surface flat: top-6 settings within 0.0004 (US_IL). lam_missing 0.061 (US_IL).
### #5
- Best decision tuned on the EVAL slice with the transferred model: US→IN 0.9097, IN→US 0.9551 -> the loss is model shift, not decision transfer.
- France is Latin script -> IN→US (gap ~0.02) is the closer proxy; final model trains on both countries (more diverse than either LOCO direction).
### #3
- Slice ceilings (link target inside slice pool): US_IL 0.9997, IN_KA 0.9997; singleton share 5.5% both. True links in same state block 95.2% / 96.1%, rest have an EMPTY pool address (fallback block).
- Strongest single pass: TF-IDF name+address (both_fwd alone 0.972 US / 0.915 IN). India key_hn alone = 132 cands/S1 (9.3M pairs) -> heavy but +0.122 marginal recall.
- Cap loses 1.6 pts of India recall at 30 (1.1 at 60) -> candidate prioritisation before the cap is a lever (e.g. cheap first-stage ranker).
- Features: 110k pairs/s on Modal 16c (2.27M pairs in 21 s).
### #0
- Measured in the skill-building session (numbers copied from er-challenge-playbook status section). OOF AUC 0.99984.
### #1
- train S1 2,206,821 · S2 5,034,616 · S3 5,285,603 · GT 7,638,365 links · test S1 1,732,544 · S2 4,887,273 · S3 5,082,316. Parsed rows == naive newline count for all 7 files (io verify gate).
- Package assembled into code/business_entity_resolution/src/ber (11 modules + run_pipeline.py); all import in package mode.
- Same cache rebuilt on Modal (identical row counts). Native-token dict from TRAIN links: 163,001 (latin, native) pairs -> 1,347 entries (13 s).
- Full-data normalization on Modal (16 cores, --n-jobs 14): train/India 883k S1 + 4.13M pool 2.0 min (7.7 GB RSS); train/US 1.32M + 6.19M 2.5 min (10.7 GB); test/France 259k + 1.43M 0.7 min.
- Data checks: every NON-EMPTY US/India address yields a state (state is always its own comma component; the "TX 77002" parse gap never occurs). Unknown state = EMPTY address only: US S2 111,121 + S3 110,968 (3.6% of pool), India S2 57,846 + S3 64,948 (~3%); these go to the fallback block. (First sample said 0% because it hash-sorted on the address string, which clusters identical empty addresses out of the sample.) France test: only 3 regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) -> 3 huge geo blocks; S2 3.2% unknown region (mostly empty addresses); names very generic (Club, Societe, SARL, Comite).
- Infra: work/infra/modal_app.py (volume mbolt-data), slices_from_norm.py (geo slices from pipeline-normalized frames), slice_eval.py (feats/oof/loco). Container psutil sees the HOST (339 GiB, 18 cores) -> threads pinned via OMP/POLARS env; memory must be requested explicitly.
### diag_struct (dataset-generation structure; diagnostics only, no model change; scripts work/infra/diag_struct_{a..k}.py, logs work/logs/modal_diag_struct_*.log)
- NO row-order / id artifacts: AUC(linked | S2/S3 row or numeric id) 0.4997-0.5006, AUC(singleton | S1 row/id) 0.4993/0.4996, Spearman(S1 row, linked row) |r|<0.0014, siblings never adjacent, ids uniform random; test identical.
- Copies: n2 0..5, n3 0..6, independent among matched S1 (corr -0.019); singletons 5.58% are their own class (independent zero draws would give 1.57%). Star structure: copy-copy name ratio 80 vs S1-copy 88; names noised per copy (typo sharing 1-1.5%), ADDRESSES noised per source (same-source siblings share the raw address 21-26% vs 0.9-1.0% across sources).
- Distractors: single records (not clusters), ~0.3% empty address vs 3.9-4.7% for copies (P(linked | empty addr) 0.977); DBA only on copies (P 0.9998), domain names P 0.957. US hard negatives = same name_core+city, house number shifted 3-100 or one digit +-3..9 (P_true 0.02-0.13) vs copy ops letter/digit add/drop (P_true 0.96-0.99996).
- TEST shift (label-free): copies per S1 equal to train (per-S1 #pool sharing name+hn+state 1.543 vs 1.544; mixture linked fraction US 0.598 / India 0.593) but ~1.9x distractors per S1. v3 test US selects 3.433 links/S1 vs 3.342 on OOF, sum p/S1 3.564 vs 3.427 (OOF truth among cands 3.426); the excess sits in hard-negative house-number relations (HARD 0.0723/S1 test vs 0.0213 OOF, SMALL +-1/2 0.039 vs 0.022) and label-free same-name+city HARD pairs per S1 are 1.84x train (US), 1.12x (India). Likely ~0.07-0.09 extra FP/S1 on test US.
- OOF rule sims (tune folds 0-2 / eval 3-4): sibling-twin add rules <= +0.00001; empty-address argmax add -0.00095; global odds rescale to match sum-p costs -0.0034 (w 0.23) / -0.0006 (w 0.54) on OOF; 'drop HARD link if S1 has an exact-hn link & p<0.99' costs -0.00032 on OOF and drops 0.0437/S1 on test US (LB probe candidate).
### diag_fr (France deep-dive; diagnostics only, no model change; scripts work/infra/diag_fr_{a..j}.py, reports work/reports/diag/{inspect_*,norm_audit,shift,segments,collisions,city_swap,reweight,street_hn,street_hn_core,empty_addr}.md)
- Label-free test F estimates (both validated on OOF within +-0.003): self-estimated E[F0.5] from calibrated p: US 0.9774 / India 0.9762 / **France 0.9698**; covariate-reweighted OOF (same-name S1 in city, same-address S1, #uncertain cands, empty): US 0.9784 / India 0.9770 / **France 0.9687**. LB 0.968038 then implies France ~0.913 (or ~0.93 if US/India lose another 0.003 to the hard-negative density found by diag_struct) -> ~0.04-0.055 France drop NOT explained by the covariate mix = concept shift (cf. LOCO slice gaps 0.021 / 0.066).
- Feature PSI France vs train (all pairs): hn_diff 1.28, idf_cand 0.66, st_inter 0.55, a_tset 0.49, grp_max_sim 0.35, comp_gap 0.27; test US/India <= 0.04 on all top-20. Causes: small house numbers (1.9 vs 3.5 digits), region name + 'rue/des' tokens in every address, short generic names (49% of FR S1 cores have <= 1 non-generic token), only 15 cities.
- France structure: 29.8% of S1 share name_key+city with another S1 (US 2.1%, India 24%), 22.9% share an address key (US/IN 9-12%), 54% have >= 1 candidate with p in [0.05,0.95) (test US/IN 40%, OOF 31-33%), acronym-like candidates 0.080/S1 (US 0.003).
- Empty-address copies whose FULL name_norm is unique among S1 but whose core is shared: OOF true 0.784 (US) / 0.886 (IN) yet p ~0.5 and only 4.6% / 9.3% selected; France has 2-3x the rate (0.0225/S1). Fix = amb on name_norm + norm-equal flag (measurable on OOF).
- Blocking proxy: same address-key + name ts>=80 non-candidates France 0.0131/S1 vs US 0.0001, India 0.005 ('Bordeaux College' = 'Bordeaux College' same address not a candidate).
- Normalization OK overall (legal forms 0-0.02% in core, 0 non-ASCII, city/state 100% consistent on confident pairs); bugs: 'Ch' -> 'chau' (FR map has 'ch' under chemin AND chaussee; 0.42% of confident pairs), 'Appt/Apt' not unit words (become house numbers / city candidates), '12 B' / '12 T' not glued as bis/ter.
- Ruled out: city swaps (FR same hn+street other-city similar-name 0.0037/S1 vs US 0.36), accent folding, department->region mapping, digit deletion (FR 0.0086/S1).
- Inspection (80 FR / 40 US / 40 IN, seed 20260926): correct 49 / 28 / 26-27; false link 3 / 0 / 0; likely missed link 14 / 1 / 1-2; empty but should not 2 / 0 / 0; unsure 12 / 11 / 12. Modal spend for diag_fr < $0.5.
### #14 stage-2 collective model (diag_stage2; scripts work/infra/diag_stage2_{lib,a_prep,b_train,c_decide,d_test,e_changes,f_orphan,g_write,h_profile,i_guard}.py, results work/reports/diag/stage2/*.json, volume /vol/diag/stage2/)
- Setup: stage-1 p for ALL 130.2M train pairs = OOF p for the 15% sample (grp0, 331,205 S1) + final-model p for the rest (110.7M rows predicted in 497 s, 16 threads). HB = hash buckets 150-299 (330,649 S1) are outside BOTH the stage-1 and the prio training -> honest test-like holdout. Stage-1 rebuilt reproduces decision.json (0.9815249) and the v3 test links exactly (878,389 / 2,696,268 / 2,276,576).
- Ablation (OOF, best decision each): stage-1 0.98152 · within+decision only 0.98246 · no_comp 0.98298 · comp from the 15% OOF frame only 0.98301 · no 72 orig feats 0.98321 · no_sib 0.98357 · **full 0.98411**. => full-frame p1 competition +0.00113 (vs OOF-only competition +0.0011 more), siblings +0.00054. Gain share: competition 0.873 (c_margin, c_soft), within 0.095, orig72 0.025, sibling 0.006.
- OOF-only limitation measured: mean other claimants per rescored pair 44.4 (full train frame) vs 6.7 (OOF frame); negatives with p1>=0.3 having another S1 at p>=0.5: 19.7% vs 3.3%.
- Ladder OOF: decision 0.01246 -> 0.01015, ranking 0.00133 -> 0.00106, blocking 0.00469 unchanged; partial_recall_model 0.00718 -> 0.00546, partial_fp 0.00287 -> 0.00241, singleton_fp 0.00154 -> 0.00115, empty_model 0.00223 -> 0.00221 (not fixed). AUC on rescored rows 0.99703 -> 0.99803, logloss 0.0579 -> 0.0470.
- Decision-only ideas on stage-1 p (OOF / HB): soft exclusivity against ALL train S1 +0.00033 / +0.00031 (= OOF under-estimates test-time exclusivity; not a test gain); lam 0..0.2, p^g, hard margins, L=24 all within +-0.0001; caps <=6 and thr+fallback lose 0.001-0.03; name-key sibling propagation LOSES 0.003-0.09 (added-link precision 0.22-0.51) -> generic names make name-key siblings unreliable.
- Orphan emulation (19% of S1 removed from the competition frame): stage-2 still +0.0019 (OOF) / +0.0020 (HB); training on the reduced frame is more robust (-0.0004 instead of -0.0006) and costs -0.00016 when nothing is missing.
- TEST shift warning: stage-2 adds 0.071 links/S1 on test US vs 0.031 on OOF; 69% of the test-US additions have a house-number MISMATCH (OOF 31%) = the hard-negative relation diag_struct found 3.4x more often on test US -> their OOF precision (0.90) is unlikely to hold. India 25% / France 31% look OOF-like. Guard G1 (mismatch rows: p = min(p1, p2)) costs 0.0005 on OOF/HB and brings test-US additions to 0.022/S1 (OOF G1 0.021).
- Files (validated with the official validator incl. candidate subset + check_submission --check-ids): /vol/diag/stage2/matching_results_stage2.tsv (G0, 5,936,043 links) and /vol/diag/stage2/matching_results_s2g1.tsv (G1, 5,886,713 links). v3 candidate_pairs.tsv stays valid (stage 2 only re-scores v3 candidates). Modal spend for diag_stage2 ~ $1.0.
### v4a attribution (2026-09-26 23:20-23:49 UTC, Modal sotaloss, 1730 s)
- v4 blocking + France normalize fixes + retrained prio, but the **v3 72-feature set** (same 331,205 S1, folds, lr 0.1):
  OOF **0.98170** (soft, bias 3.0; AUC 0.99993; 810 final rounds). -> of v4's +0.0032 over v3 (0.98152 -> 0.98473):
  blocking/normalize/prio **+0.0002**, the 26 v4 features **+0.0030**. Model: /vol/exp/v4a/model.
### #15 cross-encoder stage-2 feature (in progress, 2026-09-27)
- Plan: multilingual-e5-small (MIT, 118M; first 6 of 12 layers kept) fine-tuned as a pair classifier on raw
  "name | address" texts of TRAIN S1 buckets [300, 450) (unused by every other model), only pairs with stage-1 p in the
  uncertain band [0.01, 0.99] (India OOF: 16% of the p>=1e-3 rows, ~92% of the expected errors). Its logit + within-S1
  rank/gaps are added to stage 2; kept only if stage-2 OOF AND honest holdout HB (buckets 150-299) both improve.
- Compute: Modal refuses GPU functions without a payment method -> CPU. 32-core benchmark (ce_bench.py): 12 layers bf16
  175 pairs/s inference, 6 layers bf16 481/s, train 184/s — but only on AVX512-BF16 hosts; on AVX2 hosts bf16 falls back
  to a naive single-threaded gemm (first attempt stalled) -> scripts pick bf16 only when supported, else fp32 (80 pairs/s train).
- Train set: 236,091 band pairs (71,402 pos), val 12,555 (S1-hash split); 1.5 epochs, bs 128, lr 5e-5.
- CE training done (AVX2 host, fp32, 80 pairs/s, 4478 s): 6-layer model 107.0M params (MIT). Validation on held-out S1
  band pairs: **AUC 0.9155 / logloss 0.3254 vs stage-1 LightGBM on the same pairs AUC 0.9418 / logloss 0.2671** -> weaker
  alone; value only if complementary in stage 2. Scoring OOF/HB band pairs: 2 shards x 32 cores, ~180-220 pairs/s each.
- **Stage-2 comparison (ce_stage2.py, same frames/folds; population OOF = buckets <150, HB = 150-299; soft excl + expected-F):**
  | model | OOF all / India / US (best eb) | HB all / India / US (eb 1.0 / 1.5) | re-scored-row AUC / logloss |
  |---|---|---|---|
  | stage-1 v4 | 0.98473 / 0.98316 / 0.98578 (eb 2.0) | 0.98506 (eb 2.0) | 0.99752 / 0.05004 |
  | stage-2 v4 base (reproduces the pipeline) | 0.98569 / 0.98433 / 0.98660 (eb 1.5) | 0.98592 / 0.98443 / 0.98692 | 0.99832 / 0.04166 |
  | **stage-2 + CE (4 cols)** | **0.98752 / 0.98705 / 0.98784 (eb 1.0; +0.0018)** | **0.98755 / 0.98687 / 0.98800 (+0.0016)** | **0.99887 / 0.03309** |
  CE coverage 15.4% (India) / 17.3% (US) of the stage-2 rows; top gain c_soft, c_margin, **ce** (#3), p1, p1_logit.
  Leakage check: CE trained on other S1 (buckets 300-449); pool records can be shared across S1 populations but label
  memorisation of a record would hurt, not help, a different S1 -> gain judged genuine. Test effect unknown (shift) -> LB probe.
- Test (ce_apply.py; test band pairs scored: FR 341.6k / IN 686.5k / US 718.2k, 5 x 32-core containers ~28 min; CE
  coverage of stage-2 rows FR 22.7% / IN 17.8% / US 21.5% vs 15-17% on train = test shift toward uncertain pairs).
  Decision expected-F soft, lam 0.0525, empty_bias 1.5; adapt priors refit on the CE stage-2 OOF.
  | file | adapt | links/S1 FR / IN / US | links | differs from v4_hsr_rule / s2g1_rule (S1) | LB |
  |---|---|---|---|---|---|
  | output_probes/ce_density/ | density_hard | 3.412 / 3.345 / 3.388 | 5,841,048 | — | (pending) |
  | output_probes/ce_hsr_rule/ | density_hs + rule (dropped FR 2,192 / IN 4,340 / US 10,716) | 3.402 / 3.338 / 3.367 | 5,819,384 | 69,650 / 136,814 | (pending) |
  Both pass the official validator (--candidate NONE --check-ids); 1,732,544 rows each. Modal spend to here ~$17 of $30.
  If kept: the final package must include the CE training/scoring code + model (MIT, 107.0M params) in the pipeline.
