## Compliance audit (2026-09-26 17:44, audit_compliance.py)

- Tree: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution` (22 files, 14 Python)
- requirements: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution\requirements.txt` | manifest: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution\models_manifest.json`
- Models listed: 2; combined exact params 246,478 (cap 8,000,000,000; licenses allowed: MIT, Apache-2.0)
- **Overall: PASS** (0 FAIL, 0 WARN)

| # | Check | Result | Findings |
|---|---|---|---|
| 1 | No external APIs, hosted LLMs or network calls | PASS | 0 |
| 2 | No external data (geocoders, gazetteers, scraping, pretrained external parsers) | PASS | 0 |
| 3 | No API keys, tokens or credentials | PASS | 0 |
| 4 | Library licenses (requirements.txt) | PASS | 0 |
| 5 | Model licenses and exact size (models_manifest.json) | PASS | 0 |
| 6 | Every model id / model file used is in the manifest | PASS | 0 |
| 7 | Hygiene (reviewer-visible) | PASS | 0 |

### Libraries (requirements.txt)

| package | version | license | verdict |
|---|---|---|---|
| polars | 1.44.2 | MIT | PASS |
| pyarrow | 25.0.1 | Apache-2.0 | PASS |
| numpy | 2.2.6 | BSD-3-Clause | PASS |
| scipy | 1.15.3 | BSD-3-Clause | PASS |
| scikit-learn | 1.7.2 | BSD-3-Clause | PASS |
| rapidfuzz | 3.14.5 | MIT | PASS |
| lightgbm | 4.7.0 | MIT | PASS |
| sparse-dot-topn | 1.2.0 | Apache-2.0 | PASS |
| psutil | 7.2.2 | BSD-3-Clause | PASS |

### Models (models_manifest.json)

| model | role | license | exact params | counted from files | revision | verdict |
|---|---|---|---|---|---|---|
| master-bolt/lightgbm-matcher | final pairwise matcher (the submitted model): calibrated p(same business) for each (s1, cand) pair | MIT | 208,978 | 208,978 | - | PASS |
| master-bolt/lightgbm-candidate-priority | blocking helper: ranks each S1's candidate union before the 60-per-S1 cap (inputs = blocking pass flags / scores / ranks only) | MIT | 37,500 | 37,500 | - | PASS |

RESULT: PASS
