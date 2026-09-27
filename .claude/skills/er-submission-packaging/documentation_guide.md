# Filling Documentation_template.md (evidence map)

Copy `student_resource/Documentation_template.md` to the project root first; never edit the original.
Every number must come from `work/experiments.md` (OOF on train, stated as such) — never invent or round up.

| Template section | What to write | Source |
|---|---|---|
| Header | Team Name: Master Bolt; members; submission date | — |
| 1 Executive summary | 2–3 sentences: multi-pass blocking + LightGBM + exclusivity-aware expected-F0.5 decisions; headline OOF macro F0.5 | experiments log |
| 2.1 Problem analysis | the EDA facts: 5.58% singletons, mean 3.46 matches, exclusivity (each S2/S3 → ≤1 S1), ~25% distractors, Indic scripts (share), noise operators, France test-only, test pools 23% denser | `er-challenge-playbook/reference-data-facts.md` |
| 2.2 Strategy | Approach Type: Blocking + Classifier + constrained assignment (Hybrid). Core innovation: e.g. learned native-script token dictionary + exclusivity-aware per-S1 expected-F0.5 selection | skills |
| 3 Blocking | passes table (keys + TF-IDF directions), K/caps, geo blocks, total candidate pairs (train/test), pair recall + oracle F0.5 per country, how misses were checked | `measure_blocking.py` output |
| 4 Matching model | feature families, LightGBM params, GroupKFold by S1, calibration, threshold/decision method (OOF macro F0.5), LOCO result as France proxy | model report |
| 5 Results & error analysis | OOF macro F0.5 overall/US/India, precision, recall, leaderboard score(s), top FP and FN categories with counts | `error_report.py` |
| 6 Conclusion | what worked, what did not (with numbers) | log |
| Appendix A | code map + exact reproduction command + runtime/RAM + machine | README |
| Appendix B | licenses table (library, version, license; models with params) + the audit report + fair-play statement | `audit_compliance.py --report` |

Fair-play statement to include (edit to truth): "No external databases, APIs, geocoders, LLM services or internet data were used. All learned artifacts (native-token dictionary, model) are derived from the provided training data by the submitted code. Hand-written normalization maps encode general domain knowledge (legal-form and street-type abbreviations, state names)."
