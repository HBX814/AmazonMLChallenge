# Shared contract for all Master Bolt ER skills (MUST follow exactly)

## Where things live (project root = C:/Users/harsh/Downloads/AmazonMLChallenge)
```
AmazonMLChallenge/
├── CLAUDE.md                              # short project memory → points to er-challenge-playbook
├── .claude/skills/<skill-name>/SKILL.md   # the skills (+ supporting files)
├── student_resource/                      # organiser files — NEVER modify
│   ├── dataset/{train,test}/*.tsv
│   ├── utils/validate_submission.py
│   └── Documentation_template.md
├── code/business_entity_resolution/       # the deliverable pipeline (zipped)
│   ├── src/
│   │   ├── run_pipeline.py                # single entry point (argparse); stages: prepare, block, features, train, predict, write, all
│   │   └── ber/                           # the package
│   │       ├── __init__.py
│   │       ├── config.py                  # dataclass Config: paths, K per pass, thresholds, seeds, n_jobs, memory caps
│   │       ├── io.py                      # safe TSV readers/writers, parquet cache, GT explode
│   │       ├── normalize.py               # transliteration + name/address normalization (country-keyed maps + generic fallback)
│   │       ├── blocking.py                # candidate generation (multi-pass union) + recall measurement
│   │       ├── features.py                # pairwise + context + competition features; FEATURE_COLUMNS
│   │       ├── model.py                   # LightGBM training (GroupKFold by S1), calibration, save/load, predict
│   │       ├── decide.py                  # exclusivity assignment + per-entity selection (threshold / expected-F0.5)
│   │       ├── metric.py                  # exact macro F0.5 + breakdowns (CLI too)
│   │       └── outputs.py                 # write matching_results.tsv / candidate_pairs.tsv exactly to spec
│   ├── README.md                          # exact reproduction steps
│   └── requirements.txt                   # pinned versions
├── output/                                # matching_results.tsv, candidate_pairs.tsv (zipped)
├── Documentation_template.md              # filled copy at root of the zip (copy from student_resource and fill)
├── work/                                  # intermediates (parquet caches, candidates, features, models, logs) — NOT zipped
│   └── experiments.md                     # append-only experiment log (date, change, val F0.5 per country, notes)
└── .venv/                                 # python 3.10 venv (exists)
```
Run from `code/business_entity_resolution/`:
`python src/run_pipeline.py --data-dir ../../student_resource/dataset --work-dir ../../work --out-dir ../../output --stage all`

## Tech choices (pinned in requirements.txt; all permissive licenses)
- polars (MIT) for dataframes, duckdb (MIT) for big SQL joins, pyarrow (Apache-2.0) parquet, numpy/scipy/scikit-learn (BSD), rapidfuzz (MIT), lightgbm (MIT), anyascii (ISC) and/or indic-transliteration (MIT) for transliteration, psutil (BSD). Optional: sparse_dot_topn (Apache-2.0/MIT — verify), faiss-cpu (MIT), sentence-transformers (Apache-2.0) with an MIT/Apache model ≤ 8B.
- NOT allowed in the pipeline: any remote API (Groq, OpenAI, geocoders...), unidecode (GPL — avoid for license hygiene; use anyascii), Llama/Gemma weights (non-MIT/Apache).
- Python 3.10, Windows-compatible (multiprocessing must use `if __name__ == "__main__":` guards; spawn start method).

## Canonical column names (polars, all ids Utf8)
Source frames (after io.read_source): `entity_id, business_name, business_address, country` — all Utf8, missing -> "" (never null, never NaN).
Derived source column `source` in {"S1","S2","S3"} from the id prefix.
Normalized columns added by normalize.add_normalized_columns(df):
- `name_norm` (lowercase, transliterated, accents folded, punctuation→space, junk/brackets removed, spaces collapsed)
- `name_core` (name_norm minus legal forms / honorifics / generic filler; falls back to name_norm if it would become empty)
- `name_compact` (name_core with spaces removed — for domain-form names like "unifiedpinnacleesports")
- `addr_norm` (address normalized, NULL/None tokens removed, abbreviations canonicalized, state canonicalized)
- `house_nums` (List[Utf8] of house/door/plot numbers, digits+optional letter suffix)
- `city_key`, `state_key` (best-effort canonical strings, "" if unknown)
- `script` ("latin" | "indic" | "other"), `name_is_domain` (bool)
Ground truth long form: `s1, mid` (one row per link). Singletons are S1 ids with no rows.
Candidates: `s1, cand` + per-pass flags/scores/ranks (e.g. `p_name_score, p_name_rank, p_addr_score, p_addr_rank, p_key_hit`).
Features: `s1, cand` + FEATURE_COLUMNS (float32) (+ `label` Int8 in training).
Scored pairs: `s1, cand, p` (calibrated probability, float32).
Final links: `s1, mid`.

## Function signatures (reference modules must implement these names)
```python
# io.py
def read_source(path: str) -> pl.DataFrame            # safe TSV read: sep='\t', no quoting, all Utf8, "" for missing
def read_ground_truth(path: str) -> pl.DataFrame      # long form s1, mid
def load_split(data_dir: str, split: str) -> dict[str, pl.DataFrame]   # {"s1":..., "s2":..., "s3":..., "gt": (train only)}
# normalize.py
def transliterate(text: str) -> str
def normalize_name(name: str, country: str) -> str
def name_core(name: str, country: str) -> str
def normalize_address(addr: str, country: str) -> str
def address_parts(addr: str, country: str) -> dict    # house_nums, street_tokens, city_key, state_key
def add_normalized_columns(df: pl.DataFrame, n_jobs: int = -1) -> pl.DataFrame
# blocking.py
def generate_candidates(s1: pl.DataFrame, pool: pl.DataFrame, cfg) -> pl.DataFrame   # per country; pool = S2∪S3 normalized
def candidate_recall(cands: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series | None = None) -> dict
# features.py
FEATURE_COLUMNS: list[str]
def compute_features(cands: pl.DataFrame, s1: pl.DataFrame, pool: pl.DataFrame, cfg) -> pl.DataFrame
# model.py
def train_matcher(feats: pl.DataFrame, cfg) -> tuple[object, pl.DataFrame]   # returns model(s), OOF scored frame (s1, cand, p, label)
def predict_matcher(model, feats: pl.DataFrame) -> pl.DataFrame              # s1, cand, p
# decide.py
def assign_exclusive(scored: pl.DataFrame) -> pl.DataFrame                   # keeps, for each cand, only its best s1 (with tie/margin rule)
def select_links(scored: pl.DataFrame, method: str = "expected_f", **kw) -> pl.DataFrame   # -> s1, mid
# metric.py
def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids) -> float
def f05_breakdown(pred, gt, s1_meta) -> pl.DataFrame                          # by country, by true-k bucket
# outputs.py
def write_id_lists(s1_ids, pairs: pl.DataFrame, path: str, list_col: str) -> None   # list_col in {"matched_entity_ids","candidate_entity_ids"}
```
Country handling: every function takes the country as a plain string and MUST work for unseen countries (France) via generic fallbacks. Never one-hot/encode the country value as a model feature. Partition work by country only as a hard block (true links never cross countries).

## Skill file conventions
- Path: `.claude/skills/<name>/SKILL.md`, YAML frontmatter with `name` (letters/digits/hyphens) and `description` starting "Use when ..." (triggering conditions only, no workflow summary, third person, < 500 chars, include searchable keywords).
- Body: Overview (core principle), When to use / not, Quick reference table, the procedure/pattern, pointers to supporting files, Common mistakes (from the baseline failure catalog), measured numbers where available (cite "measured on train, 2026-09-25").
- Keep SKILL.md focused (~150-350 lines max); move heavy reference to `reference-*.md` and runnable code to `*.py` in the same folder. Reference code must be runnable, tested on the mini datasets, and follow the contract above. Invoke scripts as `python <path>` in prose.
- Cross-reference other skills by name with **REQUIRED SUB-SKILL:** / **See also:** markers; never @-include files.
- Never put API keys or credentials in any skill file.
