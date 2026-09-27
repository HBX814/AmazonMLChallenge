# Amazon ML Challenge 2026 — Business Entity Resolution (team Master Bolt)

Match every Source 1 business (`S1-…`) to all Source 2/3 records (`S2-…`, `S3-…`) of the same real business.
Score: per-S1 F0.5, macro-averaged over all test S1 (singletons count). Deliverables: `output/matching_results.tsv`,
`output/candidate_pairs.tsv`, `code/business_entity_resolution/`, filled `Documentation_template.md`, `Master_Bolt_submission.zip`.

**Before doing anything on this project, load the `er-challenge-playbook` skill** — it has the roadmap, the hard rules,
the measured data facts, the repo layout and which skill to load for each step.

Non-negotiables (details in the skills):
- Never modify `student_resource/` (official data, validator, blank template).
- Only provided data: no external APIs/LLM APIs (Groq, OpenAI, Bedrock…), no geocoders, no downloaded gazetteers. Final model MIT/Apache-2.0 and ≤ 8B params. Never store or use API keys.
- `country` is an open set (France appears only in test) — never hard-code {US, India} or encode country as a feature.
- Read TSVs as TAB-separated with quoting and NA-parsing disabled; write outputs as UTF-8 (no BOM), LF, exact headers.
- Laptop: Windows 11, 15.7 GB RAM (often 1–4 GB free), no GPU → chunk/stream, one heavy process at a time; use AWS (see `er-compute-and-aws`) for full-scale runs. Python: `.venv\Scripts\python` (3.10).
- Log every experiment with its OOF macro F0.5 in `work/experiments.md`; validate with `check_submission.py` + the official validator before any upload.
