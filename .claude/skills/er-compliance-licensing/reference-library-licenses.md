> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# R1 — Entity-resolution methods & libraries for THIS scale on THIS hardware

Amazon ML Challenge 2026, Business Entity Resolution, team Master Bolt. Written 2026-09-25 (resumed run).
Audience: agents writing `.claude/skills/*/SKILL.md` for this project. Everything marked **[measured]** was run on this
laptop (i5-1235U, 10C/12T, 15.7 GB RAM with only ~1-3 GB free, no CUDA) on 2026-09-25, often while other agents were
using the CPU too, so timings are conservative.

Related reports (parallel agents): R2 normalization (`research/R2_normalization.md`, `normalize_ref.py`),
R5 metric/decision (`research/R5_metric_decision.md`, `metric_ref.py`, `decide_ref.py`), R8 full-pool blocking
experiment (`research/R8_blocking_experiment.md`, `research/blocking_exp/`), R9 matching prototype.
R1 does NOT repeat R8's full-pool numbers; R1 benchmarks methods on two realistic-density state shards.

STATUS: in progress (sections are appended as they finish).

## 1. License & Windows/py3.10 wheel table [measured: PyPI JSON API + installed dist-info, 2026-09-25]

How checked: `research/scripts/pypi_check.py` (latest version, license fields) and `research/scripts/pypi_py310.py`
(newest non-yanked release with a wheel whose tags are installable on CPython 3.10 / win_amd64, honouring
`requires_python`). Raw outputs: `research/r1_pypi_latest.tsv`, `research/r1_pypi_py310.tsv`. Where PyPI metadata was
empty, the license was read from the installed `*.dist-info/licenses/LICENSE` (lightgbm, sparse_dot_topn, rapidfuzz, duckdb).

**Pin the "py3.10-win newest" column in requirements.txt** — several latest releases dropped Python 3.10 (numpy 2.3+,
pandas 3.x, scipy 1.16+, scikit-learn 1.8+, rapidfuzz 3.14.6, xgboost 3.3+, networkx 3.5+, onnxruntime 1.24+).

| library | installed (venv) | newest with py3.10 win_amd64 wheel | license | Windows wheel | role / verdict |
|---|---|---|---|---|---|
| polars (+ polars-runtime-32) | 1.44.2 | 1.44.2 (runtime is cp310-abi3 wheel) | MIT | yes | main dataframe engine |
| duckdb | 1.5.5 | 1.5.5 | MIT | yes | out-of-core joins/group-bys with `memory_limit` + spill |
| pyarrow | 25.0.1 | 25.0.1 | Apache-2.0 | yes | parquet IO |
| pandas | 2.3.3 | 2.3.3 (3.x needs py>=3.11) | BSD-3-Clause | yes | only for small frames / library interop |
| numpy | 2.2.6 | 2.2.6 (2.3+ needs py>=3.11) | BSD-3-Clause (+bundled permissive) | yes | |
| scipy | 1.15.3 | 1.15.3 | BSD-3-Clause | yes | CSR sparse matrices |
| scikit-learn | 1.7.2 | 1.7.2 | BSD-3-Clause | yes | HashingVectorizer / TfidfTransformer, metrics |
| sparse-dot-topn | 1.2.0 | 1.2.0 | Apache-2.0 (bundled LICENSE) | yes (cp310) | multithreaded sparse top-k cosine = TF-IDF blocking workhorse |
| rapidfuzz | 3.14.5 | 3.14.5 (3.14.6 needs py>=3.11) | MIT | yes | all string similarity features, `process.cdist/cpdist` with `workers=-1` |
| jellyfish | 1.2.1 | 1.2.1 | MIT | yes (cp310) | phonetic codes (metaphone, NYSIIS, soundex, match rating) |
| datasketch | 2.0.0 | 2.0.0 | MIT | pure | MinHash + MinHashLSH (see §3: slower and lower recall than TF-IDF top-k here) |
| faiss-cpu | 1.15.1 (global py: 1.10.0) | 1.15.1 | MIT | yes (cp310) | dense ANN (HNSW / IVF-PQ) if embeddings are used |
| hnswlib | - | **none** (0.8.0 is sdist-only, needs MSVC Build Tools) | Apache-2.0 | **no** | use faiss `IndexHNSWFlat` or `chroma-hnswlib` instead |
| chroma-hnswlib | global py 0.7.6 | 0.7.6 | Apache-2.0 | yes (cp310) | drop-in hnswlib fork with wheels |
| usearch | - | 2.26.2 | Apache-2.0 | yes | alternative HNSW, supports int8/f16 vectors |
| pynndescent | - | 0.6.0 | BSD-2-Clause | pure (needs numba) | kNN-graph; slower build than faiss, skip |
| lightgbm | 4.7.0 | 4.7.0 (py3-none-win_amd64) | MIT | yes | **pairwise matcher** (recommended) |
| xgboost | - | 3.2.0 (3.3+ needs py>=3.12) | Apache-2.0 | yes | alternative GBDT |
| catboost | - | 1.2.10 | Apache-2.0 | yes (cp310) | alternative GBDT (slower on CPU, big wheel) |
| splink | - | 4.0.17 | MIT | pure (duckdb backend) | Fellegi-Sunter baseline; not the final matcher (see §4) |
| recordlinkage | - | 0.16 (2023, unmaintained) | BSD-3-Clause | pure | pandas-based; too slow/memory-hungry at 10M rows |
| dedupe | - | 3.0.3 | MIT | yes (cp310) | active-learning dedupe; human-labelling loop, not needed (we have 7.6M labels) |
| sentence-transformers | global py 3.2.1 | 6.1.0 | Apache-2.0 | pure | only if a small MIT/Apache encoder is used |
| torch (CPU) | global py 2.6.0 | 2.14.0 | BSD-3-Clause-style (PyPI says Apache-2.0 AND BSD...) | yes | |
| onnxruntime | - | 1.23.2 (1.24+ needs py>=3.11) | MIT | yes | faster CPU inference if an encoder is used |
| model2vec | 0.9.0 | 0.9.0 | MIT | pure | static distilled embeddings, ~1-3k texts/s on this CPU (see §3.6) |
| anyascii | 0.3.3 | 0.3.3 | ISC | pure | transliteration Indic->Latin + accent folding (R2 owns details) |
| indic-transliteration | 2.3.82 | 2.3.82 | MIT | pure | alternative transliteration |
| joblib | 1.6.0 | 1.6.0 | BSD-3-Clause | pure | loky process pool (Windows-safe) |
| psutil | 7.2.2 | 7.2.2 | BSD-3-Clause | yes (abi3) | RSS monitoring |
| numba | - | 0.67.0 | BSD-2-Clause | yes (cp310) | optional JIT for custom loops (union-find, assignment) |
| networkx | - | 3.4.2 (3.5+ needs py>=3.11) | BSD-3-Clause | pure | connected components on small graphs only |
| rustworkx | - | 0.18.1 | Apache-2.0 | yes (abi3) | fast graph algorithms if needed |
| scipy.sparse.csgraph | (scipy) | - | BSD-3-Clause | yes | **connected_components on 10M-node graphs — use this** |
| scikit-network | - | 0.33.5 | BSD-3-Clause | yes | Louvain etc. if needed |
| optuna | - | 5.0.0 | MIT | pure | hyper-parameter search (optional) |
| pyahocorasick | - | 2.3.1 | BSD-3-Clause | yes | fast multi-pattern dictionary replacement (optional) |
| symspellpy | - | 6.10.0 | MIT | pure | spelling correction of city names (optional) |
| cleanco | - | 2.3 | MIT | pure | legal-suffix lists (reference only) |
| usaddress | - | 0.5.16 | MIT | pure | US address CRF parser (US only; slow in Python) |

**Do NOT use (license or platform)**: `unidecode` (GPL-2.0+), `python-Levenshtein`/`Levenshtein` (GPL-2.0+; use
`rapidfuzz.distance.Levenshtein`), `aksharamukha` (AGPL-3.0), `zingg` (AGPL-3.0), `igraph`/`python-igraph` and
`leidenalg` (GPL), `text-unidecode` (Artistic/GPL), `hnswlib` (no Windows wheel), `postal`/libpostal (needs native C lib
+ 2 GB model download, sdist only on Windows — and it is external data, arguably violating the "no external data" rule).

