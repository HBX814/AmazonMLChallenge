## Compliance audit (2026-09-27 18:32, audit_compliance.py)

- Tree: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution` (29 files, 17 Python)
- requirements: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution\requirements.txt` | manifest: `C:\Users\harsh\Downloads\AmazonMLChallenge\code\business_entity_resolution\models_manifest.json`
- Models listed: 7; combined exact params 1,360,395,124 (cap 8,000,000,000; licenses allowed: MIT, Apache-2.0)
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
| torch | 2.5.1 | BSD-3-Clause | PASS |
| transformers | 4.46.3 | Apache-2.0 | PASS |
| tokenizers | 0.20.3 | Apache-2.0 | PASS |
| safetensors | 0.4.5 | Apache-2.0 | PASS |

### Models (models_manifest.json)

| model | role | license | exact params | counted from files | revision | verdict |
|---|---|---|---|---|---|---|
| master-bolt/lightgbm-stage2 | final pairwise model: collective stage-2 re-scoring of every pair with stage-1 p >= 1e-3 (stage-1 features + within-S1 context + competition over all S1 + sibling similarities + cross-encoder logits); its calibrated p drives the decision | MIT | 30,866 | - | - | PASS |
| master-bolt/lightgbm-matcher | stage-1 pairwise matcher: calibrated p(same business) for every candidate pair (98 features) | MIT | 153,571 | - | - | PASS |
| master-bolt/cross-encoder-ce | cross-encoder 'ce': logit for (S1 text, candidate text) pairs with stage-1 p in [0.01, 0.99]; stage-2 feature | MIT | 107,007,361 | - | - | PASS |
| master-bolt/cross-encoder-ce2 | cross-encoder 'ce2': logit for (S1 text, candidate text) pairs with stage-1 p in [0.01, 0.99]; stage-2 feature | Apache-2.0 | 567,755,777 | - | - | PASS |
| master-bolt/lightgbm-candidate-priority | blocking helper: ranks each S1's candidate union before the 60-per-S1 cap (inputs = blocking pass flags / scores / ranks only) | MIT | 37,500 | - | - | PASS |
| intfloat/multilingual-e5-small | pretrained backbone of cross-encoder 'ce' (first 6 of 12 layers kept, then fine-tuned by the team) | MIT | 117,654,272 | - | 614241f622f5 | PASS |
| BAAI/bge-reranker-v2-m3 | pretrained backbone of cross-encoder 'ce2' (fine-tuned by the team) | Apache-2.0 | 567,755,777 | - | 953dc6f6f85a | PASS |

RESULT: PASS
