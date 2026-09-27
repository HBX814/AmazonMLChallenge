> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# R3: Model license audit and Groq API verdict
Amazon ML Challenge 2026, Business Entity Resolution. Team: Master Bolt.
Written 2026-09-25. Status: IN PROGRESS (sections are appended as they are finished).

## 0. TL;DR (verdict first)
- **Do NOT use the Groq API (or any hosted LLM API) in the pipeline or at any stage that touches the data.** Three independent reasons, each enough on its own:
  1. **Rule text.** README "Academic Integrity and Fair Play": *"Participants are STRICTLY NOT ALLOWED to use external databases, APIs, or services to look up business identities or resolve entities."* Sending S1/S2/S3 record pairs to a hosted LLM so it decides "same business?" is, word for word, using an external API/service to resolve entities. Penalty: "immediate disqualification". The code is audited ("We use it to reproduce your results, audit your blocking, and check the fair-play and model-license rules").
  2. **No Groq model qualifies anyway** (details in section 2). Every model Groq hosts for free or Developer tier is either over 8B parameters (gpt-oss-20b is 20.9B total, gpt-oss-120b 116.8B, qwen3.8-27b 27B) or not MIT/Apache-2.0 (Llama 3.1 8B: Llama 3.1 Community License, 8.03B; Prompt Guard 22M/86M: Llama 4 license, and they are jailbreak classifiers). `llama-3.1-8b-instant` was also **removed from free and Developer tiers on 2026-08-16** (Enterprise contracts only).
  3. **Not reproducible or feasible.** The audited zip must regenerate the outputs "using only what is in this folder". A key-dependent remote call cannot be re-run by the auditors, the hosted model can be deprecated at any time (Groq deprecated 6 LLM IDs in 2026), and at ~10M candidate pairs the free tier (gpt-oss-20b: 1K requests/day, 200K tokens/day) would need years.
- **SAFE alternative:** run an open-weights MIT/Apache-2.0 model of 8B parameters or fewer **locally** (laptop, or the team's own AWS instance, which is compute you control, not an entity-lookup service). For this task the best value is a GBDT (LightGBM MIT / XGBoost Apache-2.0 / CatBoost Apache-2.0) on string-similarity features, plus optionally a small fine-tuned multilingual encoder or cross-encoder (e.g. `intfloat/multilingual-e5-small` MIT 118M, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` Apache-2.0 118M) used only on the uncertain pairs. See sections 1, 4 and 5.

## Exact rule text (from student_resource/README.md)
- Constraint 5 (line 186): "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."
- Fair play (lines 240-254): "STRICTLY NOT ALLOWED to use external databases, APIs, or services to look up business identities or resolve entities. This includes but is not limited to: Using commercial entity resolution APIs or services; Looking up business registrations from government databases; Using geocoding APIs to normalize addresses; Any external data augmentation from internet sources." Enforcement: "Any evidence of external data lookup will result in immediate disqualification." "This challenge is designed to test your machine learning and data science skills using only the provided training data."
- Package (lines 169-175): "Anyone should be able to regenerate both output files from the training/test data using only what is in this folder."


---
## 1. Verified model table: MIT / Apache-2.0 and <=8B, useful for this task

### 1.1 How this was verified
- **License:** Hugging Face Hub API `https://huggingface.co/api/models/<id>`, fields `tags: license:*` and `cardData.license` / `license_name`. Scripts: `hf_license_audit.py` (pass 1, 05:29, 100 ids) and `hf_license_audit2.py` (pass 2, 11:05, 57 ids incl. 2026 releases and Groq-hosted models). Outputs: `hf_license_audit.json`, `hf_license_audit2.json`, `hf_license_audit2.txt`. GitHub LICENSE files were checked for code-distributed models and for the libraries (`github_licenses.txt`). Only public model ids were sent, never any data.
- **Params:** `safetensors.total` from the same API (total parameters incl. embeddings, vision towers and LM head). Where a repo has no safetensors, the count is estimated as fp32 `.bin` bytes / 4 and marked "est.".
- **Commit SHAs** of every audited repo are in the JSON files (`sha`). Pin these as `revision=` in code (section 4).
- **CPU throughput:** measured on this laptop (i5-1235U, torch 2.6 CPU, 10 threads, batch 256, `max_seq_length=64`, 3,000 real S2/S3 strings from the mini set) by `bench_encoders.py` / `bench_model2vec.py` (results in `bench_encoders_results.json`, `bench_model2vec_results.txt`, `bench_log.txt`). **Caveat: all transformer numbers were measured while other agents' Python jobs held the CPU at 100%, so treat them as pessimistic lower bounds.** The thread sweep (`bench_threads_results.txt`) was equally noisy, ranging 36-89 strings/s. "name" = business_name only (mean 8-9 tokens); "full" = "name | address" (mean 26-30 tokens). Rows without a measurement are scaled by transformer FLOPs (about 2 x non-embedding params x tokens) against the measured MiniLM rows and are marked "est.".
- **Cross-script quality (measured 2026-09-25, `bench_indic_auc.py`, outputs `r3_indic_auc_m2v.txt`, `r3_indic_auc_st.txt`):** zero-shot AUC of cosine similarity for 600 true pairs (Latin S1 name vs native-script S2/S3 name of the same entity) against 600 negatives (the same S1 name vs a random other native-script name).

| method (zero-shot, no training) | AUC Latin vs native-script names |
|---|---|
| `intfloat/multilingual-e5-small` (MIT, 118M), prefix `"query: "` | **0.967** |
| `minishlab/potion-multilingual-128M` model2vec (MIT) on raw text | 0.939 |
| rapidfuzz `token_set_ratio` after `anyascii` transliteration (MIT + ISC) | 0.932 |
| potion-multilingual-128M on anyascii-transliterated text | 0.869 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) | 0.685 (trained on translations, weak at transliteration) |
| rapidfuzz `token_set_ratio` raw (no transliteration) | 0.626 |

### 1.2 Sentence embedders (bi-encoders) for blocking and similarity features
| model | params | license (HF tag, verified) | rule OK? | languages / scripts | measured CPU (strings/s, contended) | 10M strings on laptop | notes |
|---|---|---|---|---|---|---|---|
| `minishlab/potion-base-8M` (model2vec static) | 7.6M | MIT | YES | English vocab | 3,748 name / 2,786 full | **~1.0 h** | no attention, pure lookup + mean; `pip install model2vec` (MIT) |
| `minishlab/potion-multilingual-128M` (model2vec, distilled from bge-m3) | 128.1M | MIT | YES | 101 langs incl. hi, bn, ta, te, kn, ml, gu, pa, mr, ur + fr | 2,189 name / 1,958 full | **~1.4 h** | best speed/quality for blocking at 10M scale; cross-script AUC 0.939 |
| `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | Apache-2.0 | YES | English only; **15.7% [UNK] on Indic names** | 78 name / 56 full | 36-50 h | do not use for native script |
| `sentence-transformers/all-MiniLM-L12-v2` | 33.4M | Apache-2.0 | YES | English | est. ~40 name / ~28 full | est. ~70-100 h | |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 117.7M (96M of it is the 250k-vocab embedding) | Apache-2.0 | YES | 50 langs (card lists gu, hi, mr, ur, fr); XLM-R tokenizer, 0% UNK on Indic | 105 name / 35 full | 26-79 h | cross-script AUC only 0.685 zero-shot; good fine-tuning base |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278.0M | Apache-2.0 | YES | 50 langs | est. ~25 / ~9 | est. 110-300 h | |
| `sentence-transformers/distiluse-base-multilingual-cased-v2` | 134.7M | Apache-2.0 | YES | 50 langs | est. ~50 / ~18 | | |
| `sentence-transformers/LaBSE` | 470.9M | Apache-2.0 | YES | 109 langs incl. as, bn, gu, hi, kn, ml, mr, ne, or, pa, ta, te, ur, fr | est. ~25 / ~9 (BERT-base compute) | est. 110-300 h | strong for translation pairs, heavy |
| `intfloat/multilingual-e5-small` | 117.7M | MIT | YES | 94 langs incl. all major Indic + fr | 81 name / 33 full | 34-84 h | **best zero-shot cross-script (AUC 0.967)**. Needs `"query: "` prefix. Encode only the ~1.7M test India names that are native-script or their Latin S1 counterparts (~6 h contended) |
| `intfloat/multilingual-e5-base` | 278.0M | MIT | YES | 94 langs | est. ~25 / ~9 | est. 110-300 h | |
| `intfloat/multilingual-e5-large` (and `-instruct`) | 559.9M | MIT | YES | 94 langs | est. ~7 / ~3 | infeasible on laptop | cloud only |
| `BAAI/bge-m3` | 568M (est. from 2,271 MB fp32) | MIT | YES | 100+ langs, 8192 ctx, dense + sparse + ColBERT | est. ~7 / ~3 | infeasible on laptop | XLM-R-large compute |
| `BAAI/bge-small-en-v1.5` / `bge-base-en-v1.5` | 33.4M / 109.5M | MIT | YES | English | est. ~40 / ~25 name | | |
| `Alibaba-NLP/gte-multilingual-base` | 305.4M | Apache-2.0 | YES | 75 langs incl. bn, gu, hi, kn, ml, mr, ne, pa, ta, te, ur, fr | est. ~25 / ~9 | | **needs `trust_remote_code=True`** (hub code executes). Pin `revision`, or avoid for audit hygiene |
| `Snowflake/snowflake-arctic-embed-m-v2.0` / `-l-v2.0` | 305.4M / 567.8M | Apache-2.0 | YES | 74 langs incl. Indic | est. ~25 / ~7 | | m-v2.0 also uses custom code |
| `nomic-ai/nomic-embed-text-v2-moe` | 475.3M | Apache-2.0 | YES | ~100 langs | | | custom code (MoE) |
| `Qwen/Qwen3-Embedding-0.6B` / `-4B` / `-8B` | 595.8M / 4.02B / 7.57B | Apache-2.0 | YES (8B = 7.57B total, under cap) | 100+ langs | est. <5 | infeasible on CPU | GPU only |

### 1.3 Character/byte-level and multilingual encoder backbones (for fine-tuning a matcher)
| model | params | license | rule OK? | scripts | notes |
|---|---|---|---|---|---|
| `google/byt5-small` / `byt5-base` | ~300M / ~582M (est.) | Apache-2.0 | YES | any (UTF-8 bytes) | typo- and script-robust, but Indic chars are 3 bytes each, so sequences are long and slow |
| `google/canine-c` / `canine-s` | 132.1M | Apache-2.0 | YES | any (Unicode code points) | char-level, no tokenizer; good fit for noisy names |
| `FacebookAI/xlm-roberta-base` / `-large` | 278.9M / 561.2M | MIT | YES | 100 langs incl. Indic + fr | standard cross-encoder base |
| `microsoft/Multilingual-MiniLM-L12-H384` | ~118M (est.) | MIT | YES | XLM-R vocab | small multilingual cross-encoder base |
| `microsoft/mdeberta-v3-base` | ~276M (card: 86M backbone + 190M embeddings) | MIT | YES | 100 langs | strong fine-tuning base, slower than MiniLM |
| `google-bert/bert-base-multilingual-cased` / `distilbert/distilbert-base-multilingual-cased` | 178.6M / 135.4M | Apache-2.0 | YES | 104 langs | |
| `google/muril-base-cased` | ~238M (est.) | Apache-2.0 | YES | 17 Indian langs incl. transliterated text | `ai4bharat/MuRIL` id returns 401. Use the google/ id |
| `ai4bharat/IndicBERTv2-MLM-only` | ~279M (est.) | MIT | YES | 23 Indic langs + en (no fr) | |
| `ai4bharat/indic-bert` (v1, ALBERT) | ~34M (est.) | MIT | YES | 12 Indic langs + en | HF repo is click-through gated (`gated=auto`) |
| `ai4bharat/IndicXlit` (transliteration, Indic <-> Latin) | ~11M (GitHub README) | MIT (GitHub LICENSE verified: "Copyright (c) 2022 AI4Bhārat") | YES | 21 Indic langs, both directions | weights come from GitHub releases, not HF safetensors. pip `ai4bharat-transliteration` 1.1.3 (MIT) depends on **fairseq 0.12.2, whose PyPI wheels exist only for cp36-cp38 Linux/macOS**, so Python 3.10 on Windows needs a source build (fragile). Prefer rule-based `indic-transliteration` (MIT) or `anyascii` (ISC) on the laptop, and IndicXlit only on a Linux box |
| `ai4bharat/indictrans2-indic-en-dist-200M` | 228.3M | MIT | YES | 22 Indic -> en | gated=auto; translation (not transliteration). Overkill for names |

### 1.4 Cross-encoders / rerankers (pairwise scorers)
| model | params | license | rule OK? | languages | measured CPU (pairs/s, contended, max_len 128) | notes |
|---|---|---|---|---|---|---|
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | 4.4M | Apache-2.0 | YES | en | est. ~150 | tiny, good distillation target |
| `cross-encoder/ms-marco-MiniLM-L2-v2` / `-L4-v2` | 15.6M / 19.2M | Apache-2.0 | YES | en | est. ~50 / ~30 | |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22.7M | Apache-2.0 | YES | en | **16** (10M pairs = 175 h) | zero-shot is trained for query-passage relevance, not record identity: must fine-tune |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 33.4M | Apache-2.0 | YES | en | est. ~8 | |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 117.6M | Apache-2.0 | YES | 15 langs (hi, fr, ...) | **14** (10M pairs = 205 h) | multilingual CE base |
| `cross-encoder/stsb-distilroberta-base` / `quora-distilroberta-base` | 82.1M | Apache-2.0 | YES | en | est. ~5 | quora = duplicate-question detection, the closest zero-shot task |
| `BAAI/bge-reranker-base` | 278.0M | MIT | YES | en, zh | est. ~4 | |
| `BAAI/bge-reranker-v2-m3` | 567.8M | Apache-2.0 | YES | multilingual (bge-m3 base) | est. ~1-2 | GPU only |
| `Alibaba-NLP/gte-multilingual-reranker-base` | 306.0M | Apache-2.0 | YES | 75 langs incl. Indic | est. ~4 | custom code |
| `mixedbread-ai/mxbai-rerank-base-v2` | 494.0M | Apache-2.0 | YES | 100+ langs | est. ~2 | Qwen2-based |

MS MARCO itself is a non-commercial research dataset, but these model cards carry Apache-2.0 and the rule is about the model license, so they are fine. They still must be fine-tuned on the provided training pairs to be useful.

### 1.5 Small LLMs (decoder-only), license and exact size
**Parameter cap check uses the safetensors total. Several models named "8B" are over 8.0B.**
| model | total params | license | rule OK? | notes |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M/360M/1.7B-Instruct` | 134.5M / 361.8M / 1.71B | Apache-2.0 | YES | English-centric |
| `HuggingFaceTB/SmolLM3-3B` | 3.08B | Apache-2.0 | YES | card langs: en, fr, es, it, pt, zh, ar, ru (no Indic) |
| `Qwen/Qwen2.5-0.5B-Instruct` / `1.5B-Instruct` | 494.0M / 1.54B | Apache-2.0 | YES | |
| `Qwen/Qwen2.5-3B-Instruct` | 3.09B | **`qwen-research` (other)** | **NO** | the exception in the Qwen2.5 family |
| `Qwen/Qwen2.5-7B-Instruct` | 7.62B | Apache-2.0 | YES | largest safe Qwen2.5 |
| `Qwen/Qwen3-0.6B` / `1.7B` / `4B` / `4B-Instruct-2507` | 751.6M / 2.03B / 4.02B / 4.02B | Apache-2.0 | YES | |
| `Qwen/Qwen3-8B` | **8.19B** (8,190,735,360) | Apache-2.0 | **RISKY / NO** | over "up to 8 Billion" if counted literally |
| `Qwen/Qwen3.5-0.8B` / `2B` / `4B` (2026, multimodal) | 873.4M / 2.27B / 4.66B | Apache-2.0 | YES | vision tower counted in total |
| `Qwen/Qwen3.5-9B` | 9.65B | Apache-2.0 | NO | over cap |
| `microsoft/Phi-3-mini-4k-instruct` / `Phi-3.5-mini-instruct` | 3.82B | MIT | YES | |
| `microsoft/Phi-4-mini-instruct` / `-reasoning` / `-flash-reasoning` | 3.84B / 3.84B / 3.85B | MIT | YES | Phi-4-mini lists 24 langs incl. fr. `microsoft/phi-4` (14.7B) is over cap |
| `mistralai/Mistral-7B-v0.3` / `-Instruct-v0.3` | 7.25B | Apache-2.0 | YES | |
| `mistralai/Ministral-3-3B-Instruct-2512` | 3.85B | Apache-2.0 | YES | |
| `mistralai/Ministral-3-8B-Instruct-2512` | **8.92B** | Apache-2.0 | NO | over cap (includes vision encoder) |
| `mistralai/Ministral-8B-Instruct-2410` | 8.02B | **MRL (Mistral Research License)** | NO | |
| `ibm-granite/granite-4.0-350m` / `4.0-1b` / `4.0-micro` / `4.1-3b` / `4.2-3b` | 352M / 1.63B / 3.40B / 3.40B / 3.66B | Apache-2.0 | YES | |
| `ibm-granite/granite-3.3-2b-instruct` | 2.53B | Apache-2.0 | YES | |
| `ibm-granite/granite-3.3-8b-instruct` / `granite-4.1-8b` / `granite-4.2-8b` | 8.17B / 8.79B / 8.79B | Apache-2.0 | NO | over cap |
| `allenai/OLMo-2-1124-7B-Instruct` | 7.30B | Apache-2.0 | YES | fully open data |
| `google/gemma-4-E2B-it` (Gemma 4, released April 2026 under **Apache-2.0**: Google Open Source Blog "Gemma 4: Expanding the Gemmaverse with Apache 2.0"; HF card `license: apache-2.0`) | 5.12B total (2B effective) | Apache-2.0 | YES | |
| `google/gemma-4-E4B-it` | **7,996,156,490** total (4.5B effective) | Apache-2.0 | YES, just barely | 3.84M params under the cap. Any LoRA/adapter/head of more than 3.84M params pushes it over. Avoid |
| `google/gemma-2-2b-it`, `gemma-3-1b-it`, `gemma-3-4b-it`, `google/embeddinggemma-300m` | 2.61B / 1.0B / 4.30B / 303M | **Gemma Terms of Use** | NO | only Gemma 4 is Apache |
| `meta-llama/Llama-3.1-8B-Instruct`, `Llama-3.2-1B/3B-Instruct`, Llama 3.3/4, Prompt-Guard-2 | 8.03B / 1.24B / 3.21B / ... | Llama community licenses | NO | not MIT/Apache (and 3.1-8B is 8.03B) |
| `openai/gpt-oss-20b` | 20.9B | Apache-2.0 | NO | over cap (MoE total, not active params) |
| `sarvamai/sarvam-1` | 2.53B | no license tag on HF | NO (unverifiable) | |
| `jinaai/jina-embeddings-v3`, `jinaai/jina-reranker-v2-base-multilingual` | 572M / 278M | CC-BY-NC-4.0 | NO | non-commercial |
| `l3cube-pune/indic-sentence-similarity-sbert`, `indic-sentence-bert-nli` | n/a (no safetensors) | CC-BY-4.0 | NO | not MIT/Apache, although they cover 10 Indic langs |

**Throughput reality check for LLMs** (FLOPs estimate, not measured: 10M pairs x ~140 tokens/pair x 2 x non-embedding params):
- Qwen2.5-0.5B (about 0.36B non-embedding) needs about 1.0e18 FLOP for one pass. On this laptop (about 50-150 GFLOP/s effective) that is **about 2,000-5,500 h**, so infeasible. On one L4/A10G GPU (about 30-40 TFLOP/s effective in batched prefill, scoring one "yes" logit and generating nothing) it is **about 7-10 h**.
- Qwen2.5-7B / Mistral-7B (about 6.5B non-embedding) needs about 1.8e19 FLOP, **about 130+ GPU-hours**. At most viable on a small uncertain subset (for example 5% of pairs, about 6-7 h on a g6.xlarge at $0.80/h).
- In short: LLMs are not the tool for 10M pairs on this budget. A GBDT plus a fine-tuned small encoder is.

---
## 2. Groq API: current catalog, licenses, limits, feasibility, compliance

### 2.1 Sources (fetched 2026-09-25; raw captures saved next to this report)
- https://console.groq.com/docs/models (captured `groq_models.html` / `groq_models.txt` at 05:31, re-fetched live ~11:00 with identical content)
- https://console.groq.com/docs/rate-limits (`groq_rl.html` / `groq_rl.txt`)
- https://console.groq.com/docs/deprecations (`groq_depr.html` / `groq_depr.txt`)
- https://console.groq.com/docs/your-data, https://console.groq.com/docs/batch
- License and parameter counts: Hugging Face Hub API (`hf_license_audit.json`, `hf_license_audit2.json`; `params` = safetensors total, including embeddings). No Groq key was used and no data was sent to Groq.

### 2.2 Everything Groq hosts today, with license and size
| Groq model ID | Tier / status (2026-09-25) | What it is | Open-weights license | Total params | <=8B? | MIT/Apache? | Usable for text-pair scoring? |
|---|---|---|---|---|---|---|---|
| `openai/gpt-oss-20b` | Production, free + Developer ($0.075 in / $0.30 out per 1M) | reasoning LLM, MoE | Apache-2.0 | **20.9B** (3.6B active) | NO | yes | yes, but too big |
| `openai/gpt-oss-120b` | Production, free + Developer ($0.15 / $0.60) | reasoning LLM, MoE | Apache-2.0 | **116.8B** (5.1B active) | NO | yes | yes, but too big |
| `llama-3.1-8b-instant` | Production but **"Enterprise / Contact Sales" only**. Deprecated for free and Developer tiers, shutdown **2026-08-16** (replacement: gpt-oss-20b) | LLM | **Llama 3.1 Community License** | **8.03B** (8,030,261,248) | NO | **NO** | n/a |
| `llama-3.3-70b-versatile` | Enterprise only (deprecated for free/dev 2026-08-16) | LLM | Llama 3.3 Community License | 70.6B | NO | NO | n/a |
| `qwen/qwen3.8-27b` | Preview, free + Developer ($0.80 / $4.00) | LLM (multimodal) | Apache-2.0 | **27.8B** | NO | yes | yes, but too big |
| `openai/gpt-oss-safeguard-20b` | Preview | safety-policy classifier LLM | Apache-2.0 | 21.5B | NO | yes | not designed for it; too big |
| `minimaxai/minimax-m2.7` | Preview, Enterprise only | LLM, MoE | "other" (MiniMax license) | 228.7B | NO | NO | n/a |
| `meta-llama/llama-prompt-guard-2-22m` / `-86m` | Preview | jailbreak/prompt-injection classifier (512 ctx) | Llama 4 Community License | 70.8M / 278.8M | yes | **NO** | no (fixed-label classifier) |
| `whisper-large-v3` / `whisper-large-v3-turbo` | Production | speech-to-text | Apache-2.0 / MIT | 1.54B / 0.81B | yes | yes | **no, audio input only** |
| `canopylabs/orpheus-v1-english` / `orpheus-arabic-saudi` | Preview | text-to-speech | (open orpheus-3b-0.1-ft card: Apache-2.0, 3.78B; Groq's v1 variants not separately carded) | ~3.8B | yes | probably | **no, audio output only** |
| `groq/compound`, `groq/compound-mini` | **Shut down 2026-09-21** | agentic system with built-in web search | n/a (closed system) | n/a | n/a | n/a | n/a (and web search = external lookup) |
| Recently removed: `qwen/qwen3.6-27b` (2026-09-14), `qwen/qwen3-32b` + `llama-4-scout-17b-16e` (2026-07-17), `kimi-k2-instruct-0905` (2026-04-15), `llama-4-maverick` (2026-03-09), `llama-guard-4-12b` (2026-03-05) | gone | | | | | | |

**Answer: no Groq-hosted text model is both <=8B parameters and MIT/Apache-2.0.** The only models that meet both conditions are the Whisper speech-recognition models (and possibly the Orpheus TTS voices). They take audio in or give audio out, so they cannot score business-record pairs. The "8B model" the user has in mind is `llama-3.1-8b-instant`. It fails on every count: its license is the Llama 3.1 Community License (not MIT/Apache), it has 8.03B parameters (just over the cap), and since 2026-08-16 free and Developer accounts cannot call it. The recommended replacement, gpt-oss-20b, is Apache-2.0 but has 20.9B total parameters. The rule counts the model's parameters, and for an MoE that means total parameters, not active ones.

### 2.3 Rate limits (public docs, 2026-09-25)
Free plan (the table on /docs/rate-limits):

| model | RPM | RPD | TPM | TPD |
|---|---|---|---|---|
| openai/gpt-oss-20b, gpt-oss-120b, gpt-oss-safeguard-20b, qwen/qwen3.8-27b | 30 | 1K | 8K | 200K |
| llama-prompt-guard-2-22m / 86m | 30 | 14.4K | 15K | 500K |
| orpheus (both) | 10 | 100 | 1.2K | 3.6K |
| whisper-large-v3 / turbo | 20 | 2K | - | - (ASH 7.2K, ASD 28.8K) |

Developer plan (shown in the models table): gpt-oss-20b / gpt-oss-120b / qwen3.8-27b 250K TPM, 1K RPM; gpt-oss-safeguard-20b 150K TPM, 1K RPM. Batch API: 50% discount, 24h-7d window, JSONL of up to 50,000 lines / 200 MB per file, does not consume the per-model rate limits. Batch also needs data retention: enabling Zero Data Retention disables Batch.
Data handling (/docs/your-data): inputs and outputs are not retained by default but "may be temporarily stored for up to 30 days" for reliability and abuse investigation, and "All customer data is retained in Google Cloud Platform (GCP) buckets located in the United States."

### 2.4 Feasibility at this data scale (measured token counts, local tokenizers only)
Measured with the gpt-oss-20b o200k tokenizer from the local HF cache (`r3_token_budget.py` -> `r3_token_budget.json`; 3,000 random S1 and 3,000 random S2/S3 records from the mini train set; nothing was sent anywhere):
- S1 record ("name | address"): mean 24.2 tokens (p95 41). S2/S3 record: mean 26.1 (p95 48). Indic-script S2/S3 records: 39.6 (Qwen2.5 tokenizer: 58.5). A minimal instruction is 76 tokens.
- One pair per request: 138 input tokens per pair, so 1.38B input tokens for 10M pairs.
- Grouped by S1 (1 S1 + 5.77 candidates per request = 10M / 1.73M): 286 input tokens per request, 49.5 per pair, 1.73M requests, **495M input tokens**. Assume about 100 output tokens per request: gpt-oss always emits reasoning tokens, even at low effort, plus the 0/1 JSON. That adds about 173M output tokens, **about 670M tokens in total** for one pass over test only.

| tier | binding limit | time for one test pass (10M pairs, grouped) | cost |
|---|---|---|---|
| Free, gpt-oss-20b | 200K TPD (and 1K RPD) | 670M / 200K = **~3,350 days (about 9 years)**. Even the RPD cap alone gives 1.73M / 1K = 1,733 days | $0 |
| Developer, gpt-oss-20b sync | 250K TPM, 1K RPM | 670M / 250K = ~2,700 min, **about 45 h** of saturated calls | 495M x $0.075 + 173M x $0.30 = **about $89** (Batch: about $45) |
| Developer, qwen3.8-27b | 250K TPM | about 45 h | 495M x $0.80 + 173M x $4.00 = **about $1,090** |
| Developer, llama-3.1-8b-instant | not available (Enterprise contract only) | - | Contact Sales |

Train-side validation and threshold tuning (2.2M train S1, about 1.3x the test pair count) would multiply these figures by roughly 2-3x. **So on the Developer tier the obstacle is not cost or throughput. It is the rules and the model-size cap.** On the free tier it is infeasible by three orders of magnitude.

### 2.5 Rule-compliance assessment
| Risk | Assessment |
|---|---|
| "External APIs or services to ... resolve entities" | **Direct violation.** A hosted LLM deciding whether two records are the same business is resolving entities through an external API. The clause lists "commercial entity resolution APIs or services" only as an example ("includes but is not limited to"). The penalty is immediate disqualification. There is no carve-out for "the API only runs an allowed model". |
| Model license and size (constraint 5) | Fails: no Groq text model is <=8B and MIT/Apache (section 2.2). |
| Reproducibility of the audited zip | Fails: "Anyone should be able to regenerate both output files ... using only what is in this folder." A pipeline that needs a `GROQ_API_KEY` cannot be re-run by auditors, and the team must never ship a key. Hosted models are also deprecated at short notice: Groq retired 6 LLM IDs between March and September 2026, including the 8B one. |
| Data leaving the machine | Competition data goes to a US third party with up to 30-day retention. This conflicts with the project HARD RULE ("never send dataset content to any external API/service") and is evidence an auditor can see (network calls, `groq` / `openai` imports, API-key env vars). |
| "Only used for labeling / training data / feature engineering, not inference" | Still disallowed. Using an external LLM to label or augment the provided data is "external data augmentation" plus "services to resolve entities", and it still sends data out. Keep all modelling offline. |
| Using Groq with **no data at all** (for example asking a hosted model for general coding help, or reading Groq docs) | Not a data lookup and not part of the pipeline, so it does not touch the rule. It is also pointless for the submission and should not be mixed into the project codebase. |

### 2.6 Verdict
**No. Do not use Groq, or any hosted LLM API, anywhere in the pipeline, for labeling, or for any step that sends dataset rows off the machine.** The rule prohibits it, no Groq model meets the license and size limits, and it would make the audited zip non-reproducible.

**Safe alternative:** do all matching locally with open weights you download once and ship or pin: a GBDT on handcrafted features as the main model, and optionally a small MIT/Apache transformer (bi-encoder or cross-encoder of 118M-300M parameters) fine-tuned on training pairs. Run it on the laptop or on the team's own AWS EC2/SageMaker instance. That is your own compute, not an entity-resolution service, provided no third-party API is called from it. Section 5 has concrete recipes.

Pre-submission hygiene:
- Keep `groq`, `openai`, `anthropic`, `requests` and `httpx` out of `requirements.txt` unless they are really needed. The global Python here has `groq 0.25.0` and `openai 1.93.0` installed, so do not generate requirements from `pip freeze` of the global interpreter.
- `grep -rniE "groq|openai|api[_-]?key|requests\.(get|post)|https?://" code/` must come back clean, apart from documented HF model download URLs.
- Set `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) in the final run so it is provably offline after the one-time model download.
