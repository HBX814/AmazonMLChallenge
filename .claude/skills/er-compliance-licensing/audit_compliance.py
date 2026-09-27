# -*- coding: utf-8 -*-
"""audit_compliance.py -- fair-play + license audit of the Master Bolt submission code tree.

Purpose
    Before every leaderboard upload and before zipping, prove that the audited deliverable
    (code/business_entity_resolution/) follows the challenge rules:
      1. no external API / hosted LLM / geocoder / web lookup (README "Academic Integrity and Fair Play"),
      2. no downloaded gazetteer and no address parser pretrained on external data,
      3. no API key, token or credential anywhere,
      4. requirements.txt pins only permissive libraries (no GPL/AGPL, no API clients),
      5. every model used is listed in models_manifest.json as MIT or Apache-2.0 with an EXACT integer parameter
         count <= 8,000,000,000 (README constraint 5).
    Prints a Markdown PASS/FAIL report that can be pasted into Documentation_template.md.

    Pure standard library (Python >= 3.10, nothing to install). It never opens a network connection and never
    prints a secret: every match is redacted to its first 4 characters.

Contract functions
    audit_tree(root, manifest_path=None, requirements_path=None, exclude=(), use_installed_metadata=True) -> AuditResult
    render_report(result) -> str            Markdown report
    count_params_file(path) -> int | None   exact parameter count of a .safetensors or LightGBM text model file
    load_manifest(path) -> dict

CLI
    python audit_compliance.py [ROOT] [--manifest P] [--requirements P] [--exclude GLOB ...] [--strict]
                               [--json OUT.json] [--report OUT.md] [--no-installed-metadata]
    python audit_compliance.py --count-params work/models/lgbm_fold0.txt [more files ...]
    python audit_compliance.py --self-test [DIR]
    ROOT defaults to code/business_entity_resolution (cwd first, then the project root above .claude/).
    Exit code: 0 = PASS (no FAIL; with --strict also no WARN), 1 = FAIL, 2 = usage error.

Usage example (from the project root, PowerShell or Git Bash)
    .venv/Scripts/python .claude/skills/er-compliance-licensing/audit_compliance.py code/business_entity_resolution --report work/compliance_report.md
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import fnmatch
import glob
import json
import os
import re
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

PARAM_CAP = 8_000_000_000  # README constraint 5: "up to 8 Billion parameters" -> inclusive cap on the exact count
ALLOWED_MODEL_LICENSES = ("MIT", "Apache-2.0")
ALLOWED_ARTIFACT_ORIGINS = ("hand-written", "learned-from-provided-train", "generated-by-code")
DEFAULT_EXCLUDES = (".git", "__pycache__", ".venv", "venv", "env", "node_modules", ".ipynb_checkpoints",
                    ".pytest_cache", ".mypy_cache", "audit_compliance.py")
MAX_TEXT_BYTES = 5_000_000

CODE_EXTS = {".py", ".pyw", ".ipynb", ".sh", ".bat", ".cmd", ".ps1", ".cfg", ".ini", ".toml", ".yaml", ".yml",
             ".json", ".sql", ".env"}
DOC_EXTS = {".md", ".txt", ".rst"}
MODEL_EXTS = {".safetensors", ".onnx", ".gguf", ".ggml", ".pt", ".pth", ".ckpt", ".h5", ".keras", ".tflite",
              ".msgpack", ".lgb", ".lgbm", ".cbm", ".xgb", ".ubj", ".joblib", ".pkl", ".pickle", ".bin"}
DATA_EXTS = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".parquet", ".feather", ".arrow", ".npy", ".npz", ".db",
             ".sqlite", ".duckdb", ".zip", ".gz", ".bz2", ".xz", ".7z", ".tar", ".pbf", ".shp", ".dbf", ".xml"}
CACHE_EXTS = {".parquet", ".feather", ".arrow", ".npy", ".npz", ".duckdb", ".db", ".sqlite"}
SMALL_TEXT_OK = {"requirements.txt", "readme.md", "readme.txt", "license", "license.txt", "models_manifest.json",
                 "pyproject.toml", "setup.cfg"}

# ----------------------------------------------------------------------------------------------------------------
# Rule tables (licenses verified 2026-09-25: PyPI JSON API, installed dist-info, GitHub LICENSE files, HF Hub API)
# ----------------------------------------------------------------------------------------------------------------
# import name (top-level or dotted prefix) -> (rule, reason). All of these are FAIL.
DENY_IMPORTS = {
    # hosted LLM / AI APIs: "external databases, APIs, or services to ... resolve entities" -> disqualification
    "groq": ("IMP-API", "Groq hosted-LLM client (external service; no Groq text model is <=8B AND MIT/Apache)"),
    "openai": ("IMP-API", "OpenAI hosted-LLM client"),
    "anthropic": ("IMP-API", "Anthropic hosted-LLM client"),
    "cohere": ("IMP-API", "Cohere hosted API client"),
    "mistralai": ("IMP-API", "Mistral hosted API client"),
    "google.generativeai": ("IMP-API", "Google Gemini API client"),
    "google.genai": ("IMP-API", "Google Gemini API client"),
    "vertexai": ("IMP-API", "Google Vertex AI client"),
    "together": ("IMP-API", "Together AI hosted API client"),
    "replicate": ("IMP-API", "Replicate hosted API client"),
    "litellm": ("IMP-API", "LLM API router"),
    "langchain": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "langchain_openai": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "langchain_groq": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "langchain_community": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "langchain_anthropic": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "llama_index": ("IMP-API", "LLM orchestration (hosted LLM calls)"),
    "ollama": ("IMP-API", "LLM server client (models are mostly Llama/Gemma licensed; not needed)"),
    "fireworks": ("IMP-API", "hosted LLM API client"),
    "voyageai": ("IMP-API", "hosted embedding API client"),
    # copyleft libraries (license hygiene of the audited zip)
    "unidecode": ("IMP-GPL", "Unidecode is GPL-2.0+; use anyascii (ISC) or normalize.transliterate"),
    "text_unidecode": ("IMP-GPL", "text-unidecode is Artistic/GPL"),
    "Levenshtein": ("IMP-GPL", "python-Levenshtein / Levenshtein is GPL-2.0+; use rapidfuzz.distance.Levenshtein (MIT)"),
    "aksharamukha": ("IMP-GPL", "aksharamukha is AGPL-3.0; use indic-transliteration (MIT) or normalize.transliterate"),
    "pykakasi": ("IMP-GPL", "pykakasi is GPL-3.0+ (pulled in by aksharamukha)"),
    "zingg": ("IMP-GPL", "zingg is AGPL-3.0"),
    "igraph": ("IMP-GPL", "igraph is GPL-2.0+; use scipy.sparse.csgraph"),
    "leidenalg": ("IMP-GPL", "leidenalg is GPL-3.0"),
    # pretrained parsers trained on EXTERNAL data = external data augmentation
    "postal": ("IMP-ADDR", "libpostal: native lib + ~2 GB model trained on external OSM/OpenAddresses data"),
    "deepparse": ("IMP-ADDR", "address parser pretrained on external address data (LGPL-3.0)"),
    "usaddress": ("IMP-ADDR", "CRF address parser pretrained on external labelled US addresses"),
    "probablepeople": ("IMP-ADDR", "CRF name parser pretrained on external labelled data"),
    # geocoders / gazetteers
    "geopy": ("IMP-GEO", "geocoding API client (README: 'Using geocoding APIs to normalize addresses')"),
    "googlemaps": ("IMP-GEO", "Google Maps / geocoding API client"),
    "geocoder": ("IMP-GEO", "geocoding API client"),
    "opencage": ("IMP-GEO", "OpenCage geocoding API client"),
    "mapbox": ("IMP-GEO", "Mapbox geocoding API client"),
    "herepy": ("IMP-GEO", "HERE geocoding API client"),
    "pgeocode": ("IMP-GEO", "downloads the GeoNames postal-code gazetteer at runtime"),
    "uszipcode": ("IMP-GEO", "downloads a US ZIP database at runtime"),
    "geonamescache": ("IMP-GEO", "bundled GeoNames city gazetteer (external data)"),
    "zipcodes": ("IMP-GEO", "bundled US ZIP gazetteer (external data)"),
    "pyzipcode": ("IMP-GEO", "bundled US ZIP gazetteer (external data)"),
    "reverse_geocoder": ("IMP-GEO", "bundled GeoNames gazetteer (external data)"),
    "reverse_geocode": ("IMP-GEO", "bundled GeoNames gazetteer (external data)"),
    "geotext": ("IMP-GEO", "bundled GeoNames city list (external data)"),
    # web scraping / search = external data augmentation from internet sources
    "bs4": ("IMP-SCRAPE", "HTML scraping"), "scrapy": ("IMP-SCRAPE", "web scraping"),
    "selenium": ("IMP-SCRAPE", "browser automation / scraping"), "playwright": ("IMP-SCRAPE", "browser automation"),
    "mechanicalsoup": ("IMP-SCRAPE", "web scraping"), "requests_html": ("IMP-SCRAPE", "web scraping"),
    "pyppeteer": ("IMP-SCRAPE", "browser automation"), "trafilatura": ("IMP-SCRAPE", "web scraping"),
    "newspaper": ("IMP-SCRAPE", "web scraping"), "wikipedia": ("IMP-SCRAPE", "Wikipedia lookup"),
    "wikipediaapi": ("IMP-SCRAPE", "Wikipedia lookup"), "googlesearch": ("IMP-SCRAPE", "web search"),
    "duckduckgo_search": ("IMP-SCRAPE", "web search"), "serpapi": ("IMP-SCRAPE", "web search API"),
    "tavily": ("IMP-SCRAPE", "web search API"), "firecrawl": ("IMP-SCRAPE", "web scraping API"),
    "exa_py": ("IMP-SCRAPE", "web search API"),
}
# import name -> (rule, reason). WARN: allowed only with a written justification.
WARN_IMPORTS = {
    "requests": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "httpx": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "aiohttp": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "urllib.request": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "urllib3": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "http.client": ("IMP-NET", "network client in the pipeline; the final run must be offline"),
    "socket": ("IMP-NET", "raw sockets in the pipeline; the final run must be offline"),
    "ftplib": ("IMP-NET", "network client"), "websocket": ("IMP-NET", "network client"),
    "websockets": ("IMP-NET", "network client"), "pycurl": ("IMP-NET", "network client"),
    "grpc": ("IMP-NET", "network client"), "paramiko": ("IMP-NET", "network client"),
    "smtplib": ("IMP-NET", "network client"),
    "boto3": ("AWS-SDK", "AWS SDK in the pipeline: the zip must reproduce without AWS; never call Bedrock/"
                         "Comprehend/Location/Translate/Entity Resolution"),
    "botocore": ("AWS-SDK", "AWS SDK in the pipeline"),
    "huggingface_hub": ("HF-HUB", "hub client: only for a one-time download of manifest-listed weights; "
                                  "run the pipeline with HF_HUB_OFFLINE=1"),
    "dotenv": ("ENV-LOADER", "credentials loader; the pipeline needs no secrets"),
    "pycountry": ("IMP-GAZ-LITE", "bundled ISO country/subdivision lists = external data; prefer hand-written maps"),
    "us": ("IMP-GAZ-LITE", "bundled US state metadata = external data; prefer the hand-written state map"),
    "hnswlib": ("IMP-PLATFORM", "no Python 3.10 Windows wheel; use faiss IndexHNSWFlat or chroma-hnswlib"),
}
NET_CALL_PREFIXES = ("requests.", "httpx.", "aiohttp.", "urllib.request.", "urllib3.", "http.client.",
                     "socket.create_connection", "socket.socket", "ftplib.", "pycurl.", "smtplib.", "paramiko.",
                     "websocket.", "websockets.")
HF_INFER_NAMES = ("InferenceClient", "AsyncInferenceClient", "InferenceApi", "InferenceEndpoint")
AWS_FORBIDDEN_SERVICES = {
    "bedrock", "bedrock-runtime", "bedrock-agent", "bedrock-agent-runtime", "bedrock-data-automation-runtime",
    "comprehend", "comprehendmedical", "location", "geo-places", "geo-maps", "geo-routes", "translate",
    "textract", "entityresolution", "kendra", "qbusiness", "lex-runtime", "lexv2-runtime",
}
LOADER_FUNCS = {"from_pretrained", "SentenceTransformer", "CrossEncoder", "SparseEncoder", "SetFitModel",
                "hf_hub_download", "snapshot_download", "pipeline", "distill", "TextEmbedding", "StaticModel"}
MODEL_KWARGS = {"model", "model_name", "model_name_or_path", "pretrained_model_name_or_path", "repo_id",
                "model_id", "path", "tokenizer"}
HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
LOCAL_PATH_HEADS = {"models", "model", "work", "data", "output", "src", "artifacts", "weights", "checkpoints", ".", ".."}
# Model ids that only exist as hosted endpoints (Groq catalogue, 2026-09-25) -> using them means calling the API.
HOSTED_ONLY_MODEL_IDS = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama3-8b-8192", "llama3-70b-8192",
                         "mixtral-8x7b-32768", "gemma2-9b-it", "groq/compound", "groq/compound-mini",
                         "qwen/qwen3-32b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b", "minimaxai/minimax-m2.7",
                         "moonshotai/kimi-k2-instruct-0905", "meta-llama/llama-4-scout-17b-16e-instruct",
                         "meta-llama/llama-4-maverick-17b-128e-instruct"}

# HF model id -> (license, exact params or None). HF Hub API 2026-09-25: tags license:* and safetensors.total.
KNOWN_MODELS = {
    'ai4bharat/indic-bert': ('MIT', None),
    'ai4bharat/IndicBERTv2-MLM-only': ('MIT', None),
    'ai4bharat/IndicBERTv2-MLM-Sam-TLM': ('MIT', None),
    'ai4bharat/indictrans2-indic-en-dist-200M': ('MIT', 228316160),
    'ai4bharat/IndicXlit': ('MIT', None),
    'ai4bharat/MuRIL': ('unknown (repo returns 401)', None),
    'Alibaba-NLP/gte-multilingual-base': ('Apache-2.0', 305369089),
    'Alibaba-NLP/gte-multilingual-reranker-base': ('Apache-2.0', 305959681),
    'allenai/OLMo-2-1124-7B-Instruct': ('Apache-2.0', 7298617344),
    'BAAI/bge-base-en-v1.5': ('MIT', 109482752),
    'BAAI/bge-m3': ('MIT', None),
    'BAAI/bge-reranker-base': ('MIT', 278044931),
    'BAAI/bge-reranker-v2-m3': ('Apache-2.0', 567755777),
    'BAAI/bge-small-en-v1.5': ('MIT', 33360512),
    'canopylabs/orpheus-3b-0.1-ft': ('Apache-2.0', 3782986752),
    'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1': ('Apache-2.0', 117641603),
    'cross-encoder/ms-marco-MiniLM-L12-v2': ('Apache-2.0', 33360897),
    'cross-encoder/ms-marco-MiniLM-L2-v2': ('Apache-2.0', 15616257),
    'cross-encoder/ms-marco-MiniLM-L4-v2': ('Apache-2.0', 19165185),
    'cross-encoder/ms-marco-MiniLM-L6-v2': ('Apache-2.0', 22714113),
    'cross-encoder/ms-marco-TinyBERT-L2-v2': ('Apache-2.0', 4386561),
    'cross-encoder/quora-distilroberta-base': ('Apache-2.0', 82119683),
    'cross-encoder/stsb-distilroberta-base': ('Apache-2.0', 82119683),
    'distilbert/distilbert-base-multilingual-cased': ('Apache-2.0', 135445755),
    'FacebookAI/xlm-roberta-base': ('MIT', 278885778),
    'FacebookAI/xlm-roberta-large': ('MIT', 561192082),
    'google-bert/bert-base-multilingual-cased': ('Apache-2.0', 178566653),
    'google/byt5-base': ('Apache-2.0', None),
    'google/byt5-small': ('Apache-2.0', None),
    'google/canine-c': ('Apache-2.0', 132099328),
    'google/canine-s': ('Apache-2.0', 132099328),
    'google/embeddinggemma-300m': ('gemma', 302863104),
    'google/gemma-2-2b-it': ('gemma', 2614341888),
    'google/gemma-3-1b-it': ('gemma', 999885952),
    'google/gemma-3-4b-it': ('gemma', 4300079472),
    'google/gemma-4-E2B-it': ('Apache-2.0', 5123178051),
    'google/gemma-4-E4B-it': ('Apache-2.0', 7996156490),
    'google/muril-base-cased': ('Apache-2.0', None),
    'HuggingFaceTB/SmolLM2-1.7B-Instruct': ('Apache-2.0', 1711376384),
    'HuggingFaceTB/SmolLM2-135M-Instruct': ('Apache-2.0', 134515008),
    'HuggingFaceTB/SmolLM2-360M-Instruct': ('Apache-2.0', 361821120),
    'HuggingFaceTB/SmolLM3-3B': ('Apache-2.0', 3075098624),
    'ibm-granite/granite-3.3-2b-instruct': ('Apache-2.0', 2533539840),
    'ibm-granite/granite-3.3-8b-instruct': ('Apache-2.0', 8170864640),
    'ibm-granite/granite-4.0-1b': ('Apache-2.0', 1631750144),
    'ibm-granite/granite-4.0-350m': ('Apache-2.0', 352379904),
    'ibm-granite/granite-4.0-micro': ('Apache-2.0', 3402836480),
    'ibm-granite/granite-4.1-3b': ('Apache-2.0', 3402836480),
    'ibm-granite/granite-4.1-8b': ('Apache-2.0', 8791592960),
    'ibm-granite/granite-4.2-3b': ('Apache-2.0', 3659737600),
    'ibm-granite/granite-4.2-8b': ('Apache-2.0', 8791592960),
    'intfloat/multilingual-e5-base': ('MIT', 278044162),
    'intfloat/multilingual-e5-large': ('MIT', 559890946),
    'intfloat/multilingual-e5-large-instruct': ('MIT', 559890432),
    'intfloat/multilingual-e5-small': ('MIT', 117654272),
    'jinaai/jina-embeddings-v3': ('cc-by-nc-4.0', 572310396),
    'jinaai/jina-reranker-v2-base-multilingual': ('cc-by-nc-4.0', 278437633),
    'l3cube-pune/indic-sentence-bert-nli': ('cc-by-4.0', None),
    'l3cube-pune/indic-sentence-similarity-sbert': ('cc-by-4.0', None),
    'meta-llama/Llama-3.1-8B-Instruct': ('llama3.1', 8030261248),
    'meta-llama/Llama-3.2-1B-Instruct': ('llama3.2', 1235814400),
    'meta-llama/Llama-3.2-3B-Instruct': ('llama3.2', 3212749824),
    'meta-llama/Llama-3.3-70B-Instruct': ('llama3.3', 70553706496),
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': ('other (Llama 4 Community License)', 108641793536),
    'meta-llama/Llama-Guard-4-12B': ('other (Llama 4 Community License)', 12001097216),
    'meta-llama/Llama-Prompt-Guard-2-22M': ('other (Llama 4 Community License)', 70830722),
    'meta-llama/Llama-Prompt-Guard-2-86M': ('other (Llama 4 Community License)', 278810882),
    'microsoft/mdeberta-v3-base': ('MIT', None),
    'microsoft/Multilingual-MiniLM-L12-H384': ('MIT', None),
    'microsoft/Phi-3-mini-4k-instruct': ('MIT', 3821079552),
    'microsoft/Phi-3.5-mini-instruct': ('MIT', 3821079552),
    'microsoft/phi-4': ('MIT', 14659507200),
    'microsoft/Phi-4-mini-flash-reasoning': ('MIT', 3852562944),
    'microsoft/Phi-4-mini-instruct': ('MIT', 3836021760),
    'microsoft/Phi-4-mini-reasoning': ('MIT', 3836021760),
    'MiniMaxAI/MiniMax-M2.7': ('other (MiniMax)', 228689764864),
    'minishlab/M2V_multilingual_output': ('MIT', 128269824),
    'minishlab/potion-base-8M': ('MIT', 7559168),
    'minishlab/potion-multilingual-128M': ('MIT', 128090368),
    'mistralai/Ministral-3-3B-Instruct-2512': ('Apache-2.0', 3849090048),
    'mistralai/Ministral-3-8B-Instruct-2512': ('Apache-2.0', 8918026716),
    'mistralai/Ministral-8B-Instruct-2410': ('other (Mistral Research License)', 8019808256),
    'mistralai/Mistral-7B-Instruct-v0.3': ('Apache-2.0', 7248023552),
    'mistralai/Mistral-7B-v0.3': ('Apache-2.0', 7248023552),
    'mixedbread-ai/mxbai-rerank-base-v2': ('Apache-2.0', 494032768),
    'moonshotai/Kimi-K2-Instruct': ('other (modified MIT)', 1026408235864),
    'nomic-ai/nomic-embed-text-v1.5': ('Apache-2.0', 136731648),
    'nomic-ai/nomic-embed-text-v2-moe': ('Apache-2.0', 475292928),
    'openai/gpt-oss-120b': ('Apache-2.0', 116829156672),
    'openai/gpt-oss-20b': ('Apache-2.0', 20914757184),
    'openai/gpt-oss-safeguard-20b': ('Apache-2.0', 21511953984),
    'openai/whisper-large-v3': ('Apache-2.0', 1543490560),
    'openai/whisper-large-v3-turbo': ('MIT', 808878080),
    'Qwen/Qwen2.5-0.5B-Instruct': ('Apache-2.0', 494032768),
    'Qwen/Qwen2.5-1.5B-Instruct': ('Apache-2.0', 1543714304),
    'Qwen/Qwen2.5-3B-Instruct': ('other (qwen-research)', 3085938688),
    'Qwen/Qwen2.5-7B-Instruct': ('Apache-2.0', 7615616512),
    'Qwen/Qwen3-0.6B': ('Apache-2.0', 751632384),
    'Qwen/Qwen3-1.7B': ('Apache-2.0', 2031739904),
    'Qwen/Qwen3-32B': ('Apache-2.0', 32762123264),
    'Qwen/Qwen3-4B': ('Apache-2.0', 4022468096),
    'Qwen/Qwen3-4B-Instruct-2507': ('Apache-2.0', 4022468096),
    'Qwen/Qwen3-8B': ('Apache-2.0', 8190735360),
    'Qwen/Qwen3-Embedding-0.6B': ('Apache-2.0', 595776512),
    'Qwen/Qwen3-Embedding-4B': ('Apache-2.0', 4021774336),
    'Qwen/Qwen3-Embedding-8B': ('Apache-2.0', 7567295488),
    'Qwen/Qwen3-Reranker-0.6B': ('Apache-2.0', 595776512),
    'Qwen/Qwen3.5-0.8B': ('Apache-2.0', 873438784),
    'Qwen/Qwen3.5-2B': ('Apache-2.0', 2274069824),
    'Qwen/Qwen3.5-4B': ('Apache-2.0', 4659865088),
    'Qwen/Qwen3.5-9B': ('Apache-2.0', 9653104368),
    'Qwen/Qwen3.6-27B': ('Apache-2.0', 27781427952),
    'Qwen/Qwen3.6-35B-A3B': ('Apache-2.0', 35951822704),
    'Qwen/Qwen3.8-27B': ('Apache-2.0', 27781427952),
    'sarvamai/sarvam-1': ('unknown (no license tag)', 2525087744),
    'sarvamai/sarvam-m': ('Apache-2.0', 23572403200),
    'sentence-transformers/all-MiniLM-L12-v2': ('Apache-2.0', 33360512),
    'sentence-transformers/all-MiniLM-L6-v2': ('Apache-2.0', 22713728),
    'sentence-transformers/all-mpnet-base-v2': ('Apache-2.0', 109486978),
    'sentence-transformers/distiluse-base-multilingual-cased-v2': ('Apache-2.0', 134734080),
    'sentence-transformers/LaBSE': ('Apache-2.0', 470927360),
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': ('Apache-2.0', 117654272),
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2': ('Apache-2.0', 278044162),
    'Snowflake/snowflake-arctic-embed-l-v2.0': ('Apache-2.0', 567754752),
    'Snowflake/snowflake-arctic-embed-m-v2.0': ('Apache-2.0', 305368320),
}
_KNOWN_LOWER = {k.lower(): (k, v) for k, v in KNOWN_MODELS.items()}
# Families whose licenses are not MIT/Apache even when a specific id is missing from KNOWN_MODELS.
FAMILY_RULES = [
    (re.compile(r"^meta-llama/", re.I), "Llama community license (not MIT/Apache)"),
    (re.compile(r"^google/(gemma-[123]|embeddinggemma|codegemma|paligemma|medgemma)", re.I), "Gemma Terms of Use (only Gemma 4 is Apache-2.0)"),
    (re.compile(r"^mistralai/Ministral-8B-Instruct-2410$", re.I), "Mistral Research License"),
    (re.compile(r"^Qwen/Qwen2\.5-(3B|72B)", re.I), "qwen-research / qwen license (not Apache)"),
    (re.compile(r"^jinaai/", re.I), "jina models are mostly CC-BY-NC-4.0 (non-commercial)"),
]
MODEL_ORGS = {k.split("/")[0].lower() for k in KNOWN_MODELS} | {
    "deepseek-ai", "tiiuae", "eleutherai", "bigscience", "facebook", "thebloke", "unsloth", "nousresearch",
    "01-ai", "stabilityai", "databricks", "mosaicml", "cohereforai", "coherelabs", "thenlper", "hkunlp", "qwen",
    "xlm-roberta-base", "sentence-transformers", "google-t5", "t5-small",
}

# PEP 503-normalised distribution name -> license (verified 2026-09-25)
ALLOW_LIBS = {
    "polars": "MIT", "polars-runtime-32": "MIT", "polars-runtime-64": "MIT", "polars-lts-cpu": "MIT",
    "duckdb": "MIT", "pyarrow": "Apache-2.0", "pandas": "BSD-3-Clause", "numpy": "BSD-3-Clause",
    "scipy": "BSD-3-Clause", "scikit-learn": "BSD-3-Clause", "sparse-dot-topn": "Apache-2.0", "rapidfuzz": "MIT",
    "jellyfish": "MIT", "datasketch": "MIT", "faiss-cpu": "MIT", "chroma-hnswlib": "Apache-2.0",
    "usearch": "Apache-2.0", "lightgbm": "MIT", "xgboost": "Apache-2.0", "catboost": "Apache-2.0",
    "sentence-transformers": "Apache-2.0", "transformers": "Apache-2.0", "torch": "BSD-3-Clause",
    "onnxruntime": "MIT", "model2vec": "MIT", "anyascii": "ISC", "indic-transliteration": "MIT",
    "joblib": "BSD-3-Clause", "psutil": "BSD-3-Clause", "numba": "BSD-2-Clause", "llvmlite": "BSD-2-Clause",
    "networkx": "BSD-3-Clause", "rustworkx": "Apache-2.0", "scikit-network": "BSD-3-Clause", "optuna": "MIT",
    "pyahocorasick": "BSD-3-Clause", "symspellpy": "MIT", "tokenizers": "Apache-2.0", "safetensors": "Apache-2.0",
    "huggingface-hub": "Apache-2.0", "hf-xet": "Apache-2.0", "threadpoolctl": "BSD-3-Clause",
    "tqdm": "MPL-2.0 AND MIT", "pyyaml": "MIT", "regex": "Apache-2.0 AND CNRI-Python",
    "packaging": "Apache-2.0 OR BSD-2-Clause", "python-dateutil": "Apache-2.0 OR BSD-3-Clause", "pytz": "MIT",
    "tzdata": "Apache-2.0", "six": "MIT", "typing-extensions": "PSF-2.0", "filelock": "MIT",
    "fsspec": "BSD-3-Clause", "jinja2": "BSD-3-Clause", "markupsafe": "BSD-3-Clause", "pytest": "MIT",
    "splink": "MIT", "recordlinkage": "BSD-3-Clause", "dedupe": "MIT", "textdistance": "MIT",
    "sentencepiece": "Apache-2.0", "indic-nlp-library": "MIT", "setfit": "Apache-2.0", "peft": "Apache-2.0",
    "accelerate": "Apache-2.0", "datasets": "Apache-2.0", "optimum": "Apache-2.0", "ctranslate2": "MIT",
    "openvino": "Apache-2.0", "unicodedata2": "Apache-2.0", "cloudpickle": "BSD-3-Clause",
    "click": "BSD-3-Clause", "colorama": "BSD-3-Clause",
}
# name -> (license / problem, fix). FAIL.
DENY_LIBS = {
    "unidecode": ("GPL-2.0+", "use anyascii (ISC) / normalize.transliterate"),
    "text-unidecode": ("Artistic / GPL-2.0+", "use anyascii (ISC)"),
    "python-levenshtein": ("GPL-2.0+", "use rapidfuzz.distance.Levenshtein (MIT)"),
    "levenshtein": ("GPL-2.0+", "use rapidfuzz.distance.Levenshtein (MIT)"),
    "aksharamukha": ("AGPL-3.0", "use indic-transliteration (MIT) / normalize.transliterate"),
    "pykakasi": ("GPL-3.0+", "dependency of aksharamukha; remove both"),
    "zingg": ("AGPL-3.0", "not needed"), "igraph": ("GPL-2.0+", "use scipy.sparse.csgraph"),
    "python-igraph": ("GPL-2.0+", "use scipy.sparse.csgraph"), "leidenalg": ("GPL-3.0", "not needed"),
    "deepparse": ("LGPL-3.0 + pretrained on external address data", "hand-written address rules"),
    "postal": ("libpostal: native lib + model trained on external data", "hand-written address rules"),
    "pypostal": ("libpostal: native lib + model trained on external data", "hand-written address rules"),
    "usaddress": ("CRF pretrained on external labelled addresses", "hand-written rules / learned from train"),
    "probablepeople": ("CRF pretrained on external labelled names", "hand-written rules / learned from train"),
    "groq": ("hosted-LLM API client", "forbidden: external service used to resolve entities"),
    "openai": ("hosted-LLM API client", "forbidden"), "anthropic": ("hosted-LLM API client", "forbidden"),
    "cohere": ("hosted API client", "forbidden"), "mistralai": ("hosted API client", "forbidden"),
    "google-generativeai": ("hosted API client", "forbidden"), "google-genai": ("hosted API client", "forbidden"),
    "google-cloud-aiplatform": ("hosted API client", "forbidden"), "together": ("hosted API client", "forbidden"),
    "replicate": ("hosted API client", "forbidden"), "litellm": ("hosted API router", "forbidden"),
    "langchain": ("LLM orchestration", "forbidden"), "langchain-openai": ("LLM orchestration", "forbidden"),
    "langchain-groq": ("LLM orchestration", "forbidden"), "langchain-community": ("LLM orchestration", "forbidden"),
    "langchain-anthropic": ("LLM orchestration", "forbidden"), "llama-index": ("LLM orchestration", "forbidden"),
    "ollama": ("LLM server client", "not needed"), "fireworks-ai": ("hosted API client", "forbidden"),
    "voyageai": ("hosted API client", "forbidden"),
    "geopy": ("geocoding API client", "forbidden"), "googlemaps": ("geocoding API client", "forbidden"),
    "geocoder": ("geocoding API client", "forbidden"), "opencage": ("geocoding API client", "forbidden"),
    "pgeocode": ("downloads GeoNames gazetteer", "forbidden"), "uszipcode": ("downloads ZIP DB", "forbidden"),
    "geonamescache": ("bundled gazetteer", "forbidden"), "zipcodes": ("bundled gazetteer", "forbidden"),
    "pyzipcode": ("bundled gazetteer", "forbidden"), "reverse-geocoder": ("bundled gazetteer", "forbidden"),
    "reverse-geocode": ("bundled gazetteer", "forbidden"), "geotext": ("bundled gazetteer", "forbidden"),
    "beautifulsoup4": ("web scraping", "forbidden"), "bs4": ("web scraping", "forbidden"),
    "scrapy": ("web scraping", "forbidden"), "selenium": ("browser automation", "forbidden"),
    "playwright": ("browser automation", "forbidden"), "mechanicalsoup": ("web scraping", "forbidden"),
    "requests-html": ("web scraping", "forbidden"), "pyppeteer": ("browser automation", "forbidden"),
    "trafilatura": ("web scraping", "forbidden"), "newspaper3k": ("web scraping", "forbidden"),
    "wikipedia": ("external lookup", "forbidden"), "wikipedia-api": ("external lookup", "forbidden"),
    "googlesearch-python": ("web search", "forbidden"), "duckduckgo-search": ("web search", "forbidden"),
    "google-search-results": ("web search API", "forbidden"), "tavily-python": ("web search API", "forbidden"),
    "firecrawl-py": ("web scraping API", "forbidden"),
}
# name -> reason. WARN.
WARN_LIBS = {
    "requests": "network client; remove unless justified (the pipeline must run offline)",
    "httpx": "network client; remove unless justified", "aiohttp": "network client; remove unless justified",
    "urllib3": "network client; remove unless justified",
    "boto3": "AWS SDK: the zip must reproduce without AWS", "botocore": "AWS SDK: the zip must reproduce without AWS",
    "python-dotenv": "credentials loader; the pipeline needs no secrets",
    "hnswlib": "no Python 3.10 Windows wheel (sdist needs MSVC)",
    "pycountry": "bundled ISO lists (external data); prefer hand-written maps",
    "us": "bundled US state metadata (external data); prefer hand-written maps",
    "cleanco": "bundled legal-suffix lists; fine as reference, prefer the hand-written list in normalize.py",
}
DIST_IMPORTS = {
    "scikit-learn": ("sklearn",), "faiss-cpu": ("faiss",), "pyyaml": ("yaml",), "sparse-dot-topn": ("sparse_dot_topn",),
    "indic-transliteration": ("indic_transliteration",), "sentence-transformers": ("sentence_transformers",),
    "huggingface-hub": ("huggingface_hub",), "chroma-hnswlib": ("hnswlib",), "python-levenshtein": ("Levenshtein",),
    "levenshtein": ("Levenshtein",), "beautifulsoup4": ("bs4",), "scikit-network": ("sknetwork",),
    "pyahocorasick": ("ahocorasick",), "python-dateutil": ("dateutil",), "text-unidecode": ("text_unidecode",),
    "google-generativeai": ("google.generativeai",), "google-genai": ("google.genai",), "python-dotenv": ("dotenv",),
    "polars-lts-cpu": ("polars",), "faiss-gpu": ("faiss",), "python-igraph": ("igraph",), "pypostal": ("postal",),
    "reverse-geocoder": ("reverse_geocoder",), "requests-html": ("requests_html",), "llama-index": ("llama_index",),
    "wikipedia-api": ("wikipediaapi",), "duckduckgo-search": ("duckduckgo_search",), "fireworks-ai": ("fireworks",),
    "firecrawl-py": ("firecrawl",), "tavily-python": ("tavily",), "google-search-results": ("serpapi",),
}
TRANSITIVE_OK = {"polars-runtime-32", "polars-runtime-64", "threadpoolctl", "joblib", "llvmlite", "tokenizers",
                 "safetensors", "huggingface-hub", "hf-xet", "tzdata", "python-dateutil", "pytz", "six", "packaging",
                 "typing-extensions", "filelock", "fsspec", "jinja2", "markupsafe", "networkx", "sympy", "mpmath",
                 "regex", "tqdm", "pyarrow", "pytest", "torch", "numpy", "scipy", "transformers", "onnxruntime"}

SECRET_PATTERNS = [
    ("Groq API key", re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9]{20,}")),
    ("Anthropic API key", re.compile(r"(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("OpenAI-style API key", re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("AWS access key id", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])")),
    ("AWS secret access key", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("Hugging Face token", re.compile(r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}")),
    ("Google API key", re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_\-]{35}")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("hard-coded secret assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|password)\b\s*[=:]\s*"
        r"['\"][A-Za-z0-9_\-/+=]{16,}['\"]")),
]
KEY_ENV_NAMES_RE = re.compile(
    r"\b(GROQ_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY|GEMINI_API_KEY|MISTRAL_API_KEY|COHERE_API_KEY|"
    r"TOGETHER_API_KEY|HF_TOKEN|HUGGING_FACE_HUB_TOKEN|HUGGINGFACEHUB_API_TOKEN|AWS_SECRET_ACCESS_KEY|"
    r"AWS_ACCESS_KEY_ID|AWS_SESSION_TOKEN)\b")
MENTION_RE = re.compile(r"(?i)\b(groq|openai|anthropic|bedrock|gemini|chatgpt|gpt-4o?|gpt-3\.5)\b")
CRED_FILE_RE = re.compile(r"(?i)^(\.env(\..*)?|credentials(\.json|\.csv)?|.*\.pem|.*\.key|id_rsa.*|id_ed25519.*|"
                          r"\.netrc|\.pypirc|token|.*accesskeys?\.csv)$")
GAZ_NAME_RE = re.compile(r"(?i)(geonames|gazetteer|allcountries|cities\d{3,}|admin[12]codes|pin_?codes?|"
                         r"zip_?codes?|uszips|zcta|postal_?codes?|postcodes?|openaddresses|libpostal|india_?post|"
                         r"laposte|code_?postal|communes|world_?cities|simplemaps|osm_|\.osm\.|\.pbf$)")
COMP_DATA_RE = re.compile(r"(?i)^(train|test)_(source[123]|ground_truth)\.tsv$")
URL_RE = re.compile(r"https?://([A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)(?::\d+)?(/[^\s\"'<>)\]}`,]*)?")
FORBIDDEN_URL_RULES = [
    (r"(^|\.)groq\.com$", "hosted LLM API (Groq)"),
    (r"(^|\.)openai\.com$|(^|\.)openai\.azure\.com$", "hosted LLM API (OpenAI)"),
    (r"(^|\.)anthropic\.com$|(^|\.)claude\.ai$", "hosted LLM API (Anthropic)"),
    (r"(^|\.)(generativelanguage|aiplatform)\.googleapis\.com$", "hosted LLM API (Google)"),
    (r"(^|\.)(mistral\.ai|cohere\.ai|cohere\.com|together\.xyz|together\.ai|replicate\.com|openrouter\.ai|"
     r"fireworks\.ai|deepinfra\.com|perplexity\.ai|cognitiveservices\.azure\.com)$", "hosted AI API"),
    (r"^(api-inference|router)\.huggingface\.co$|\.endpoints\.huggingface\.cloud$", "hosted inference (Hugging Face)"),
    (r"(^|\.)(bedrock[a-z-]*|comprehend|translate|textract|entityresolution|geo|places\.geo|maps\.geo)"
     r"\.[a-z0-9-]+\.amazonaws\.com$", "AWS AI / geo service endpoint"),
    (r"(^|\.)(maps|places)\.googleapis\.com$|^maps\.google\.[a-z.]+$", "geocoding / places API (Google)"),
    (r"(^|\.)nominatim\.[a-z.]+$|(^|\.)photon\.komoot\.io$|(^|\.)overpass-api\.de$", "geocoder (OpenStreetMap)"),
    (r"(^|\.)(opencagedata\.com|mapbox\.com|hereapi\.com|here\.com|positionstack\.com|geocode\.maps\.co|"
     r"geoapify\.com|locationiq\.com|bigdatacloud\.net|geocoder\.ca|geocod\.io|smartystreets\.com|smarty\.com|"
     r"zippopotam\.us|postalpincode\.in)$", "geocoding / postal lookup API"),
    (r"^api-adresse\.data\.gouv\.fr$|^data\.geopf\.fr$|(^|\.)geo\.api\.gouv\.fr$", "geocoder (France)"),
    (r"(^|\.)(geonames\.org|simplemaps\.com|openaddresses\.io|geofabrik\.de|openstreetmap\.org)$",
     "gazetteer / map data download"),
    (r"(^|\.)(data\.gov\.in|data\.gouv\.fr|insee\.fr|api\.gouv\.fr|mca\.gov\.in|sec\.gov|"
     r"company-information\.service\.gov\.uk|opencorporates\.com|gleif\.org|census\.gov|usps\.com|"
     r"indiapost\.gov\.in)$", "government / business-registry data"),
    (r"(^|\.)(yelp\.com|foursquare\.com|dnb\.com|zoominfo\.com|clearbit\.com|crunchbase\.com|justdial\.com|"
     r"indiamart\.com|pagesjaunes\.fr|yellowpages\.com)$", "business directory"),
    (r"(^|\.)(wikipedia\.org|wikidata\.org|dbpedia\.org)$", "external knowledge base"),
    (r"^(www\.)?(google\.[a-z.]+|bing\.com|duckduckgo\.com)$|(^|\.)(serpapi\.com|tavily\.com|firecrawl\.dev|exa\.ai)$",
     "web search / scraping service"),
]
FORBIDDEN_URL_RULES = [(re.compile(p, re.I), why) for p, why in FORBIDDEN_URL_RULES]
FORBIDDEN_URL_PATH_RE = re.compile(r"(?i)libpostal|openvenues")
LIC_ALIASES = {"mit": "MIT", "mit license": "MIT", "the mit license": "MIT", "expat": "MIT",
               "apache-2.0": "Apache-2.0", "apache 2.0": "Apache-2.0", "apache2": "Apache-2.0",
               "apache-2": "Apache-2.0", "apache license 2.0": "Apache-2.0", "apache license, version 2.0": "Apache-2.0",
               "apache license version 2.0": "Apache-2.0", "apache software license": "Apache-2.0",
               "apache 2.0 license": "Apache-2.0"}
PERMISSIVE_RE = re.compile(r"(?i)\b(MIT|BSD|Apache|ISC|PSF|Python Software Foundation|Zlib|Unlicense|CC0|MPL|"
                           r"Mozilla|HPND|0BSD|ZPL|Zope)\b")
COPYLEFT_RE = re.compile(r"(?i)(\bA?GPL|\bLGPL|GNU (Affero |Lesser )?General Public|Artistic|\bEUPL|\bSSPL|"
                         r"CC-BY-NC|Commons Clause)")
GROUPS = [
    ("No external APIs, hosted LLMs or network calls",
     {"IMP-API", "API-CALL", "NET-CALL", "IMP-NET", "HF-INFER", "HF-HUB", "AWS-SVC", "AWS-SDK", "URL-FORBIDDEN",
      "MODEL-HOSTED", "OFFLINE"}),
    ("No external data (geocoders, gazetteers, scraping, pretrained external parsers)",
     {"IMP-GEO", "IMP-ADDR", "IMP-SCRAPE", "IMP-GAZ-LITE", "DATA-GAZ", "DATA-ORIGIN", "DATA-UNDECL", "DATA-COMP"}),
    ("No API keys, tokens or credentials", {"SECRET", "ENV-KEY", "ENV-LOADER", "CRED-FILE", "KEY-MENTION"}),
    ("Library licenses (requirements.txt)",
     {"IMP-GPL", "REQ-MISSING-FILE", "REQ-DENY", "REQ-UNKNOWN", "REQ-UNPINNED", "REQ-INDEX", "REQ-URL",
      "REQ-FREEZE", "REQ-UNUSED", "REQ-NOTPINNED", "REQ-EDITABLE", "IMP-PLATFORM"}),
    ("Model licenses and exact size (models_manifest.json)",
     {"MAN-MISSING", "MAN-INVALID", "MAN-LICENSE", "MAN-PARAMS", "MAN-NEAR-CAP", "MAN-KNOWN-MISMATCH", "MAN-FINAL",
      "MAN-REVISION", "MAN-TRC", "MAN-PARAMS-MISMATCH", "MAN-DUP"}),
    ("Every model id / model file used is in the manifest",
     {"MODEL-BANNED", "MODEL-UNLISTED", "MODEL-LOCAL", "FILE-UNLISTED", "TRC", "MODEL-REV"}),
    ("Hygiene (reviewer-visible)", {"URL-CODE", "MENTION", "CACHE", "PARSE"}),
]


@dataclass
class Finding:
    level: str      # FAIL | WARN | INFO
    rule: str
    where: str
    message: str


@dataclass
class AuditResult:
    root: str
    when: str
    files_scanned: int = 0
    py_files: int = 0
    findings: list = field(default_factory=list)
    libs: list = field(default_factory=list)       # dicts: name, spec, license, verdict
    models: list = field(default_factory=list)     # dicts: id, role, license, params, revision, verdict
    artifacts: list = field(default_factory=list)  # dicts: path, origin, verdict
    manifest_path: str = ""
    requirements_path: str = ""
    total_params: int = 0          # sum of exact params over manifest 'models' (the whole predicting system)

    @property
    def n_fail(self):
        return sum(f.level == "FAIL" for f in self.findings)

    @property
    def n_warn(self):
        return sum(f.level == "WARN" for f in self.findings)

    def rules(self, level=None):
        return {f.rule for f in self.findings if level is None or f.level == level}


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def _redact(s: str) -> str:
    return f"{s[:4]}...[redacted, {len(s)} chars]"


def _scrub(text: str) -> str:
    for _, rx in SECRET_PATTERNS:
        text = rx.sub(lambda m: _redact(m.group(0)), text)
    return text


def _canon_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _canon_license(lic) -> str:
    if not isinstance(lic, str):
        return str(lic)
    s = lic.strip()
    return LIC_ALIASES.get(s.lower(), s)


def _is_key_env(name: str) -> bool:
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name.upper()) if p]
    if any(p in {"SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS", "APIKEY"} for p in parts):
        return True
    for i, p in enumerate(parts):
        if p == "KEY" and i > 0 and parts[i - 1] in {"API", "ACCESS", "SECRET", "PRIVATE"}:
            return True
    return False


def _url_verdict(host: str, url: str):
    for rx, why in FORBIDDEN_URL_RULES:
        if rx.search(host):
            return why
    if FORBIDDEN_URL_PATH_RE.search(url):
        return "libpostal model/data download"
    return None


def _match_deny(modname: str, table: dict):
    """Return (key, value) if modname equals a table key or is a dotted child of it."""
    for key, val in table.items():
        if modname == key or modname.startswith(key + "."):
            return key, val
    return None


def _looks_like_model_id(s: str) -> bool:
    if not HF_ID_RE.match(s):
        return False
    head = s.split("/")[0]
    return head.lower() in MODEL_ORGS and head.lower() not in LOCAL_PATH_HEADS


def _hf_id_from_url(host: str, path: str):
    if host.lower() not in ("huggingface.co", "www.huggingface.co", "hf.co"):
        return None
    segs = [p for p in (path or "").split("/") if p]
    if segs and segs[0] in ("api", "models"):
        segs = segs[1:]
    if len(segs) >= 2 and segs[0] not in ("docs", "datasets", "spaces", "blog", "settings", "join", "login"):
        return f"{segs[0]}/{segs[1]}"
    return None


# ----------------------------------------------------------------------------------------------------------------
# parameter counting (exact)
# ----------------------------------------------------------------------------------------------------------------
def count_safetensors_params(path) -> int:
    """Sum of prod(shape) over all tensors in the safetensors header (no torch needed)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file too short for safetensors")
        n = struct.unpack("<Q", raw)[0]
        if n <= 1 or n > 200_000_000:
            raise ValueError("implausible safetensors header length")
        header = json.loads(f.read(n).decode("utf-8"))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        c = 1
        for d in meta["shape"]:
            c *= int(d)
        total += c
    return total


def _is_lightgbm_text(path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(64)
    except OSError:
        return False
    return head.startswith(b"tree\n") or head.startswith(b"tree\r\n")


def count_lightgbm_params(path) -> int:
    """Trainable values of a LightGBM text model: one threshold per split + one value per leaf, over all trees."""
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        if f.readline().strip() != "tree":
            raise ValueError("not a LightGBM text model (first line must be 'tree')")
        for line in f:
            if line.startswith("threshold=") or line.startswith("leaf_value="):
                total += len(line.split("=", 1)[1].split())
            elif line.startswith("end of trees"):
                break
    return total


def count_params_file(path):
    """Exact parameter count for .safetensors or LightGBM text files; None when the format is not countable."""
    p = str(path)
    if p.lower().endswith(".safetensors"):
        return count_safetensors_params(p)
    if _is_lightgbm_text(p):
        return count_lightgbm_params(p)
    return None


# ----------------------------------------------------------------------------------------------------------------
# manifest + requirements
# ----------------------------------------------------------------------------------------------------------------
def load_manifest(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_requirements(path: Path, depth=0):
    """Yield (lineno, kind, name, spec, raw) with kind in {pkg, url, index, editable, include}."""
    out = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return out
    for i, raw in enumerate(lines, 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement")):
            inc = line.split(None, 1)[1].strip() if " " in line else ""
            if inc and depth < 3:
                out += _parse_requirements(path.parent / inc, depth + 1)
            continue
        if line.startswith(("-i ", "--index-url", "--extra-index-url", "-f ", "--find-links", "--trusted-host")):
            out.append((i, "index", "", "", raw))
            continue
        if line.startswith(("-e ", "--editable")):
            out.append((i, "editable", "", "", raw))
            continue
        if line.startswith("-"):
            continue
        if re.match(r"^(git\+|hg\+|svn\+|https?://|file:)", line) or " @ " in line:
            name = line.split(" @ ", 1)[0].strip() if " @ " in line else ""
            out.append((i, "url", _canon_dist(name) if name else "", line, raw))
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$", line)
        if m:
            out.append((i, "pkg", _canon_dist(m.group(1)), m.group(3).strip(), raw))
    return out


def _installed_license(dist_name: str):
    try:
        import importlib.metadata as md
        meta = md.metadata(dist_name)
    except Exception:
        return None
    le = meta.get("License-Expression") or ""
    lic = (meta.get("License") or "").strip().splitlines()[0][:60] if meta.get("License") else ""
    cls = [c.split("::")[-1].strip() for c in (meta.get_all("Classifier") or []) if c.startswith("License")]
    text = " | ".join(x for x in [le, lic] + cls if x)
    return text or None


# ----------------------------------------------------------------------------------------------------------------
# the audit
# ----------------------------------------------------------------------------------------------------------------
class _Auditor:
    def __init__(self, root, manifest_path, requirements_path, exclude, use_installed_metadata):
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path) if manifest_path else self.root / "models_manifest.json"
        self.req_path = Path(requirements_path) if requirements_path else self.root / "requirements.txt"
        self.exclude = tuple(DEFAULT_EXCLUDES) + tuple(exclude or ())
        self.use_meta = use_installed_metadata
        self.res = AuditResult(root=str(self.root), when=_dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                               manifest_path=str(self.manifest_path), requirements_path=str(self.req_path))
        self._seen = set()
        self.imports = {}            # top-level import name -> first location
        self.model_refs = {}         # model id -> first location
        self.local_model_refs = {}   # local path string -> location
        self.model_files = []        # (relpath, abspath)
        self.data_files = []         # (relpath, abspath, size)
        self.local_modules = set()
        self.manifest = None
        self.offline_flag_seen = False

    # -- bookkeeping ----------------------------------------------------------------------------------------------
    def add(self, level, rule, where, message, dedup_key=None):
        key = dedup_key or (rule, where)
        if key in self._seen:
            return
        self._seen.add(key)
        self.res.findings.append(Finding(level, rule, where, _scrub(message)))

    def rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return p.as_posix()

    def excluded(self, rel: str) -> bool:
        parts = rel.split("/")
        for pat in self.exclude:
            if any(fnmatch.fnmatch(part, pat) for part in parts) or fnmatch.fnmatch(rel, pat):
                return True
        return False

    # -- walk -----------------------------------------------------------------------------------------------------
    def run(self) -> AuditResult:
        files = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            d = Path(dirpath)
            dirnames[:] = [x for x in dirnames if not self.excluded(self.rel(d / x))]
            for fn in filenames:
                p = d / fn
                r = self.rel(p)
                if not self.excluded(r):
                    files.append((r, p))
        for r, p in files:
            if p.suffix == ".py":
                self.local_modules.add(p.stem)
                self.local_modules.add(p.parent.name)
        self.local_modules.discard("")
        self.manifest = self._load_manifest()
        for r, p in files:
            self._scan_file(r, p)
        self.res.files_scanned = len(files)
        self._check_requirements()
        self._check_manifest_and_models()
        self._check_data_files()
        if self.model_refs and not self.offline_flag_seen:
            self.add("WARN", "OFFLINE", "tree", "Hugging Face models are referenced but HF_HUB_OFFLINE is never set; "
                     "set HF_HUB_OFFLINE=1 (and TRANSFORMERS_OFFLINE=1) for the final run so it is provably offline")
        return self.res

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return None
        try:
            return load_manifest(self.manifest_path)
        except (OSError, json.JSONDecodeError) as e:
            self.add("FAIL", "MAN-INVALID", self.rel(self.manifest_path), f"manifest is not valid JSON: {e}")
            return None

    def _scan_file(self, rel: str, p: Path):
        name = p.name
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
        except OSError:
            return
        if CRED_FILE_RE.match(name) and name != "models_manifest.json":
            self.add("FAIL", "CRED-FILE", rel, "credential / secret file inside the code tree; delete it (never "
                     "ship keys) and revoke any key it held")
        if COMP_DATA_RE.match(name):
            self.add("WARN", "DATA-COMP", rel, "competition data copied into the code tree; read it from "
                     "--data-dir instead of shipping it")
        is_model = ext in MODEL_EXTS and (ext != ".bin" or size > 1_000_000)
        if not is_model and ext in {".txt", ".model", ".lgb"} and _is_lightgbm_text(p):
            is_model = True
        if is_model:
            self.model_files.append((rel, p))
            return
        if ext in DATA_EXTS and name.lower() not in SMALL_TEXT_OK and not name.lower().startswith("requirements"):
            self.data_files.append((rel, p, size))
        if size > MAX_TEXT_BYTES:
            return
        if ext in CODE_EXTS or ext in DOC_EXTS or ext == "" or name.lower().startswith(".env"):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return
            self.res.py_files += ext == ".py"
            if "HF_HUB_OFFLINE" in text or "TRANSFORMERS_OFFLINE" in text:
                self.offline_flag_seen = True
            if ext == ".py":
                self._scan_python(rel, text)
            elif ext == ".ipynb":
                self._scan_notebook(rel, text)
            self._scan_text(rel, text, is_doc=ext in DOC_EXTS, is_manifest=(p.resolve() == self.manifest_path.resolve()))

    # -- text-level scans (all text files) --------------------------------------------------------------------------
    def _scan_text(self, rel, text, is_doc, is_manifest):
        def lineno(pos):
            return text.count("\n", 0, pos) + 1
        for label, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                self.add("FAIL", "SECRET", f"{rel}:{lineno(m.start())}",
                         f"{label} found ({_redact(m.group(0))}); remove it, revoke/rotate the key, and never "
                         f"store keys in files")
        for m in URL_RE.finditer(text):
            host, path = m.group(1).lower(), m.group(2) or ""
            where = f"{rel}:{lineno(m.start())}"
            why = _url_verdict(host, m.group(0))
            hf_id = _hf_id_from_url(host, path)
            if why:
                lvl = "WARN" if is_doc else "FAIL"
                self.add(lvl, "URL-FORBIDDEN", where, f"URL to {why}: {host}" +
                         (" (in docs: remove unless it states non-use)" if is_doc else ""))
            elif hf_id:
                if not is_doc and not is_manifest:
                    self.model_refs.setdefault(hf_id, where)
            elif not is_doc and not is_manifest:
                self.add("WARN", "URL-CODE", where, f"URL literal in code/config ({host}); confirm it is never "
                         f"fetched at run time")
        if not is_manifest:
            for m in KEY_ENV_NAMES_RE.finditer(text):
                where = f"{rel}:{lineno(m.start())}"
                if rel.endswith(".py"):
                    continue  # the AST pass reports real env reads precisely
                lvl = "WARN" if is_doc else "FAIL"
                self.add(lvl, "KEY-MENTION", where, f"{m.group(1)} referenced: the pipeline must need no API key")
            hits = [m for m in MENTION_RE.finditer(text)]
            if hits:
                first = hits[0]
                self.add("WARN", "MENTION", f"{rel}:{lineno(first.start())}",
                         f"{len(hits)} mention(s) of hosted-AI vendors (first: '{first.group(0)}'); reviewers grep "
                         f"for these -- keep only explicit statements of non-use", dedup_key=("MENTION", rel))
            if not rel.endswith(".py"):
                for s in re.findall(r"[\"'\s=:]([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*)[\"'\s,]", text):
                    if _looks_like_model_id(s) and not is_doc:
                        self.model_refs.setdefault(s, rel)
                for hid in HOSTED_ONLY_MODEL_IDS:
                    if re.search(r"(?i)(?<![A-Za-z0-9_./-])" + re.escape(hid) + r"(?![A-Za-z0-9_-])", text):
                        self.add("WARN" if is_doc else "FAIL", "MODEL-HOSTED", rel,
                                 f"hosted-only model id '{hid}' (Groq catalogue) referenced")

    def _scan_notebook(self, rel, text):
        try:
            nb = json.loads(text)
            src = "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        except (json.JSONDecodeError, AttributeError, TypeError):
            return
        src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        self._scan_python(rel + "[cells]", src)

    # -- python AST scan ----------------------------------------------------------------------------------------------
    def _scan_python(self, rel, text):
        try:
            tree = ast.parse(text)
        except SyntaxError as e:
            self.add("WARN", "PARSE", rel, f"could not parse ({e.msg} line {e.lineno}); scanned as text only")
            return
        alias = {}  # name bound in this file -> fully qualified module/attribute
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self._check_import(rel, node.lineno, a.name)
                    if a.asname:
                        alias[a.asname] = a.name
                    else:
                        top = a.name.split(".")[0]
                        alias[top] = top
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                self._check_import(rel, node.lineno, node.module)
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    alias[a.asname or a.name] = full
                    # `from google import generativeai` hides the denied module in the imported name
                    if _match_deny(full, DENY_IMPORTS) and not _match_deny(node.module, DENY_IMPORTS):
                        self._check_import(rel, node.lineno, full)
        # bare string statements (docstrings) are documentation, not model references
        doc_ids = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self._check_call(rel, node, alias)
            elif isinstance(node, ast.Subscript):
                self._check_env_subscript(rel, node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc_ids:
                self._check_string(rel, node)

    def _check_import(self, rel, lineno, modname):
        top = modname.split(".")[0]
        if top in self.local_modules:
            return
        self.imports.setdefault(top, f"{rel}:{lineno}")
        self.imports.setdefault(modname, f"{rel}:{lineno}")
        hit = _match_deny(modname, DENY_IMPORTS)
        if hit:
            rule, why = hit[1]
            self.add("FAIL", rule, f"{rel}:{lineno}", f"imports `{modname}`: {why}")
            return
        hit = _match_deny(modname, WARN_IMPORTS)
        if hit:
            rule, why = hit[1]
            self.add("WARN", rule, f"{rel}:{lineno}", f"imports `{modname}`: {why}")

    @staticmethod
    def _dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
        if isinstance(node, ast.Call):  # e.g. boto3.Session().client(...)
            inner = _Auditor._dotted(node.func)
            return (inner + "()." + ".".join(reversed(parts))) if inner else ".".join(reversed(parts))
        return ".".join(reversed(parts))

    def _check_call(self, rel, node, alias):
        name = self._dotted(node.func)
        if not name:
            return
        head, _, tail = name.partition(".")
        resolved = alias.get(head, head) + ("." + tail if tail else "")
        where = f"{rel}:{node.lineno}"
        last = name.split(".")[-1]
        if _match_deny(resolved, DENY_IMPORTS) and DENY_IMPORTS[_match_deny(resolved, DENY_IMPORTS)[0]][0] == "IMP-API":
            self.add("FAIL", "API-CALL", where, f"calls `{resolved}` (hosted API client)")
        if resolved.startswith(NET_CALL_PREFIXES) or resolved in ("urllib.request.urlopen",):
            self.add("FAIL", "NET-CALL", where, f"network call `{resolved}(...)`: the pipeline must not fetch "
                     f"anything at run time")
        if last in HF_INFER_NAMES or resolved.startswith("huggingface_hub.inference"):
            self.add("FAIL", "HF-INFER", where, f"`{last}` = Hugging Face hosted inference (external service)")
        if last in ("client", "resource") and ("boto" in resolved or "session" in resolved.lower() or
                                                "boto3" in self.imports):
            svc = None
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                svc = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "service_name" and isinstance(kw.value, ast.Constant):
                    svc = kw.value.value
            if svc:
                if svc.lower() in AWS_FORBIDDEN_SERVICES:
                    self.add("FAIL", "AWS-SVC", where, f"AWS `{svc}` client: managed AI/geo/entity-resolution "
                             f"service = external service (Bedrock with data is forbidden)")
                else:
                    self.add("WARN", "AWS-SVC", where, f"AWS `{svc}` client in the pipeline; the zip must "
                             f"reproduce offline from local files")
        for kw in node.keywords:
            if kw.arg == "trust_remote_code" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.add("WARN", "TRC", where, "trust_remote_code=True executes code downloaded from the Hub; "
                         "prefer models without custom code, else pin revision= and review the code")
        if last in ("getenv",) or name.endswith("environ.get") or name.endswith("environ.setdefault") or \
                name.endswith("environ.pop"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                self._env_name(rel, node.lineno, node.args[0].value)
        if last in LOADER_FUNCS:
            vals = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            vals += [kw.value.value for kw in node.keywords if kw.arg in MODEL_KWARGS and
                     isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)]
            has_rev = any(kw.arg == "revision" for kw in node.keywords)
            for v in vals:
                if HF_ID_RE.match(v) and v.split("/")[0].lower() not in LOCAL_PATH_HEADS and \
                        not (self.root / v).exists():
                    self.model_refs.setdefault(v, where)
                    if not has_rev:
                        self.add("WARN", "MODEL-REV", where, f"`{last}('{v}')` without revision=<commit sha>; "
                                 f"pin the audited commit so the weights cannot change")
                elif ("/" in v or "\\" in v or v.startswith(".")) and not v.startswith("http"):
                    self.local_model_refs.setdefault(v, where)

    def _check_env_subscript(self, rel, node):
        name = self._dotted(node.value)
        if name.endswith("environ"):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                self._env_name(rel, node.lineno, sl.value)

    def _env_name(self, rel, lineno, var):
        if var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            self.offline_flag_seen = True
        if _is_key_env(var):
            self.add("FAIL", "ENV-KEY", f"{rel}:{lineno}", f"reads secret env var `{var}`: the pipeline must need "
                     f"no API key or token")

    def _check_string(self, rel, node):
        s = node.value
        if len(s) > 200:
            return
        low = s.strip().lower()
        if low in HOSTED_ONLY_MODEL_IDS:
            self.add("FAIL", "MODEL-HOSTED", f"{rel}:{node.lineno}", f"hosted-only model id '{s}' (Groq catalogue): "
                     f"using it means calling an external API")
        elif _looks_like_model_id(s.strip()):
            self.model_refs.setdefault(s.strip(), f"{rel}:{node.lineno}")

    # -- requirements -----------------------------------------------------------------------------------------------
    def _check_requirements(self):
        rel = self.rel(self.req_path)
        if not self.req_path.exists():
            self.add("FAIL", "REQ-MISSING-FILE", rel, "requirements.txt missing (README: the package must pin "
                     "dependencies)")
            return
        entries = _parse_requirements(self.req_path)
        pinned = set()
        for lineno, kind, name, spec, raw in entries:
            where = f"{rel}:{lineno}"
            if kind == "index":
                self.add("WARN", "REQ-INDEX", where, f"custom index / find-links `{raw.strip()}`; auditors need "
                         f"plain PyPI")
                continue
            if kind == "editable":
                self.add("WARN", "REQ-EDITABLE", where, "editable install; pin a released version instead")
                continue
            if kind == "url":
                m = URL_RE.search(spec)
                why = _url_verdict(m.group(1).lower(), m.group(0)) if m else None
                self.add("FAIL" if why else "WARN", "REQ-URL", where, f"URL requirement `{spec}`"
                         + (f" -> {why}" if why else "; prefer a PyPI pin"))
                if name:
                    pinned.add(name)
                continue
            pinned.add(name)
            ver = ""
            m = re.match(r"^===?\s*([^,;\s*]+)\s*(;.*)?$", spec)
            if m:
                ver = m.group(1)
            if name in DENY_LIBS:
                lic, fix = DENY_LIBS[name]
                self.add("FAIL", "REQ-DENY", where, f"`{name}` ({lic}): {fix}")
                verdict, lic_s = "FAIL", lic
            elif name in WARN_LIBS:
                self.add("WARN", "REQ-DENY", where, f"`{name}`: {WARN_LIBS[name]}", dedup_key=("REQ-WARN", name))
                verdict, lic_s = "WARN", ALLOW_LIBS.get(name, _installed_license(name) if self.use_meta else "") or "?"
            elif name in ALLOW_LIBS:
                verdict, lic_s = "PASS", ALLOW_LIBS[name]
            else:
                lic_s = _installed_license(name) if self.use_meta else None
                if lic_s and COPYLEFT_RE.search(lic_s) and not (" OR " in lic_s and PERMISSIVE_RE.search(lic_s)):
                    verdict = "FAIL"
                    self.add("FAIL", "REQ-DENY", where, f"`{name}` is copyleft per installed metadata ({lic_s})")
                elif lic_s and PERMISSIVE_RE.search(lic_s):
                    verdict = "PASS"
                    lic_s = f"{lic_s} (installed metadata)"
                else:
                    verdict = "WARN"
                    lic_s = lic_s or "unknown"
                    self.add("WARN", "REQ-UNKNOWN", where, f"`{name}`: license not in the verified table "
                             f"({lic_s}); check PyPI/GitHub and add it to the Documentation table")
            if not ver:
                self.add("WARN", "REQ-UNPINNED", where, f"`{name}` not pinned with ==; pin the exact version "
                         f"installed in .venv")
                verdict = "FAIL" if verdict == "FAIL" else "WARN"
            self.res.libs.append({"name": name, "version": ver or spec or "(unpinned)", "license": lic_s,
                                  "verdict": verdict})
        n_pkgs = sum(1 for e in entries if e[1] == "pkg")
        if n_pkgs > 40 or pinned & {"pip", "setuptools", "wheel"}:
            self.add("WARN", "REQ-FREEZE", rel, f"{n_pkgs} pins (or pip/setuptools/wheel pinned): looks like "
                     f"`pip freeze`; hand-write pins from the imports in src/ only")
        imported_dists = set()
        for mod in self.imports:
            top = mod.split(".")[0]
            if top in self.local_modules or top in getattr(sys, "stdlib_module_names", ()) or top == "__future__":
                continue
            imported_dists.add(mod)
        for name in sorted(pinned):
            mods = DIST_IMPORTS.get(name, (name.replace("-", "_"),))
            if not mods:
                continue
            if not any(m in self.imports or m.split(".")[0] in self.imports for m in mods) and \
                    name not in TRANSITIVE_OK and name not in DENY_LIBS:
                self.add("WARN", "REQ-UNUSED", rel, f"`{name}` is pinned but never imported in the tree "
                         f"(pip-freeze leftover?)", dedup_key=("REQ-UNUSED", name))
        import_to_dist = {}
        for dist, mods in DIST_IMPORTS.items():
            for m in mods:
                import_to_dist.setdefault(m.split(".")[0], set()).add(dist)
        for mod in sorted(imported_dists):
            if "." in mod:
                continue
            cands = import_to_dist.get(mod, set()) | {_canon_dist(mod)}
            if not cands & pinned and not _match_deny(mod, DENY_IMPORTS):
                self.add("WARN", "REQ-NOTPINNED", self.imports[mod], f"`import {mod}` but no matching pin in "
                         f"requirements.txt", dedup_key=("REQ-NOTPINNED", mod))

    # -- manifest + model references ----------------------------------------------------------------------------------
    def _manifest_globs(self, entry):
        out = set()
        for pat in entry.get("files") or []:
            for hit in glob.glob(os.path.join(str(self.root), pat), recursive=True):
                out.add(str(Path(hit).resolve()))
        lp = entry.get("local_path")
        if lp:
            base = (self.root / lp).resolve()
            if base.is_dir():
                out |= {str(x.resolve()) for x in base.rglob("*") if x.is_file()}
            elif base.exists():
                out.add(str(base))
        return out

    def _check_manifest_and_models(self):
        mrel = self.rel(self.manifest_path)
        entries, by_id = [], {}
        if self.manifest is None:
            if not any(f.rule == "MAN-INVALID" for f in self.res.findings):
                self.add("FAIL", "MAN-MISSING", mrel, "models_manifest.json missing: list the final model (LightGBM "
                         "booster) and every pretrained model with license + exact params (template in "
                         "er-compliance-licensing)")
        else:
            entries = self.manifest.get("models")
            if not isinstance(entries, list) or not entries:
                self.add("FAIL", "MAN-INVALID", mrel, "manifest needs a non-empty 'models' list")
                entries = []
        covered_files = set()
        for i, e in enumerate(entries):
            if not isinstance(e, dict) or not isinstance(e.get("id"), str):
                self.add("FAIL", "MAN-INVALID", f"{mrel}#models[{i}]", "each model entry needs a string 'id'")
                continue
            mid = e["id"]
            where = f"{mrel}#{mid}"
            if mid.lower() in by_id:
                self.add("WARN", "MAN-DUP", where, "duplicate model id in manifest")
            by_id[mid.lower()] = e
            lic = _canon_license(e.get("license"))
            params = e.get("params")
            verdict = "PASS"
            if lic not in ALLOWED_MODEL_LICENSES:
                self.add("FAIL", "MAN-LICENSE", where, f"license '{e.get('license')}' is not MIT or Apache-2.0 "
                         f"(README constraint 5)")
                verdict = "FAIL"
            if isinstance(params, bool) or not isinstance(params, int):
                self.add("FAIL", "MAN-PARAMS", where, f"params must be the exact integer count (got "
                         f"{params!r}); use safetensors.total or count_params_file()")
                verdict = "FAIL"
            elif params <= 0:
                self.add("FAIL", "MAN-PARAMS", where, f"params must be > 0 (got {params})")
                verdict = "FAIL"
            elif params > PARAM_CAP:
                self.add("FAIL", "MAN-PARAMS", where, f"{params:,} params > {PARAM_CAP:,} cap (names like '8B' "
                         f"are often over: Qwen3-8B = 8,190,735,360)")
                verdict = "FAIL"
            elif params >= 0.99 * PARAM_CAP:
                self.add("WARN", "MAN-NEAR-CAP", where, f"{params:,} params is within 1% of the cap; any adapter "
                         f"or head can push it over")
                verdict = "WARN" if verdict == "PASS" else verdict
            known = _KNOWN_LOWER.get(mid.lower())
            if known:
                kid, (klic, kparams) = known
                if klic not in ALLOWED_MODEL_LICENSES:
                    self.add("FAIL", "MAN-KNOWN-MISMATCH", where, f"verified HF license of {kid} is '{klic}' "
                             f"(2026-09-25 audit), not MIT/Apache-2.0")
                    verdict = "FAIL"
                elif lic in ALLOWED_MODEL_LICENSES and klic != lic:
                    self.add("FAIL", "MAN-KNOWN-MISMATCH", where, f"manifest says {lic}, verified HF license is {klic}")
                    verdict = "FAIL"
                if kparams is not None and isinstance(params, int) and not isinstance(params, bool) \
                        and params != kparams:
                    self.add("FAIL", "MAN-KNOWN-MISMATCH", where, f"declared params {params:,} != verified "
                             f"safetensors.total {kparams:,}")
                    verdict = "FAIL"
                if kparams is not None and kparams > PARAM_CAP:
                    self.add("FAIL", "MAN-KNOWN-MISMATCH", where, f"verified size {kparams:,} > cap")
                    verdict = "FAIL"
            for rx, why in FAMILY_RULES:
                if rx.search(mid) and not (known and known[1][0] in ALLOWED_MODEL_LICENSES):
                    self.add("FAIL", "MAN-KNOWN-MISMATCH", where, f"{mid}: {why}")
                    verdict = "FAIL"
            is_team = str(e.get("kind", "")).lower() in {"gbdt", "team-trained", "own"} or \
                mid.lower().startswith("master-bolt/")
            if not is_team and not e.get("revision"):
                self.add("WARN", "MAN-REVISION", where, "no 'revision' (commit sha); pin it in the manifest and "
                         "in code (revision=...)")
                verdict = "WARN" if verdict == "PASS" else verdict
            if e.get("trust_remote_code"):
                self.add("WARN", "MAN-TRC", where, "trust_remote_code models execute Hub code; avoid for audit "
                         "hygiene")
            files = self._manifest_globs(e)
            covered_files |= files
            counted = None
            countable = [f for f in sorted(files) if count_params_file(f) is not None]
            if countable:
                counted = sum(count_params_file(f) for f in countable)
                if isinstance(params, int) and not isinstance(params, bool) and counted != params:
                    self.add("FAIL", "MAN-PARAMS-MISMATCH", where, f"declared params {params:,} but the listed "
                             f"files contain exactly {counted:,}")
                    verdict = "FAIL"
                if counted > PARAM_CAP:
                    self.add("FAIL", "MAN-PARAMS", where, f"listed files contain {counted:,} params > cap")
                    verdict = "FAIL"
            self.res.models.append({
                "id": mid, "role": e.get("role", ""), "license": lic,
                "params": params, "params_counted_from_files": counted,
                "revision": e.get("revision") or "", "verdict": verdict})
        self.res.total_params = sum(m["params"] for m in self.res.models
                                    if isinstance(m["params"], int) and not isinstance(m["params"], bool))
        if self.res.total_params > PARAM_CAP:
            # an encoder feeding the booster is part of the predicting system, so the cap applies to the sum
            self.add("FAIL", "MAN-PARAMS", mrel, f"all listed models together hold {self.res.total_params:,} params "
                     f"> {PARAM_CAP:,}; every model that affects predictions counts toward the final model")
        final_id = (self.manifest or {}).get("final_model")
        if self.manifest is not None:
            if not final_id:
                self.add("FAIL", "MAN-FINAL", mrel, "manifest has no 'final_model' (the submitted matcher)")
            elif final_id.lower() not in by_id:
                self.add("FAIL", "MAN-FINAL", mrel, f"final_model '{final_id}' is not listed in 'models'")
        for mid, where in sorted(self.model_refs.items()):
            known = _KNOWN_LOWER.get(mid.lower())
            banned = None
            if known:
                klic, kparams = known[1]
                if klic not in ALLOWED_MODEL_LICENSES:
                    banned = f"license '{klic}'"
                elif kparams is not None and kparams > PARAM_CAP:
                    banned = f"{kparams:,} params > {PARAM_CAP:,}"
            else:
                for rx, why in FAMILY_RULES:
                    if rx.search(mid):
                        banned = why
            if banned:
                self.add("FAIL", "MODEL-BANNED", where, f"model `{mid}` is not allowed: {banned}")
            if mid.lower() not in by_id:
                self.add("FAIL", "MODEL-UNLISTED", where, f"model `{mid}` is used but not listed in "
                         f"models_manifest.json (license + exact params + revision required)")
        for lp, where in sorted(self.local_model_refs.items()):
            base = (self.root / lp)
            if base.exists():
                hits = {str(x.resolve()) for x in ([base] if base.is_file() else base.rglob("*"))}
                if hits and not hits & covered_files:
                    self.add("WARN", "MODEL-LOCAL", where, f"local model path '{lp}' is not covered by any "
                             f"manifest 'files'/'local_path'")
        for rel, p in self.model_files:
            if str(p.resolve()) not in covered_files:
                cnt = None
                try:
                    cnt = count_params_file(p)
                except (ValueError, OSError, KeyError):
                    pass
                self.add("FAIL", "FILE-UNLISTED", rel, "model/weights file not listed in models_manifest.json "
                         "'files'" + (f" (contains {cnt:,} params)" if cnt is not None else ""))

    # -- data artifacts -------------------------------------------------------------------------------------------------
    def _check_data_files(self):
        declared = {}
        for a in (self.manifest or {}).get("artifacts") or []:
            if isinstance(a, dict) and a.get("path"):
                for hit in glob.glob(os.path.join(str(self.root), a["path"]), recursive=True):
                    declared[str(Path(hit).resolve())] = a
        for rel, p, size in self.data_files:
            a = declared.get(str(p.resolve()))
            gaz = bool(GAZ_NAME_RE.search(p.name))
            if p.suffix.lower() in CACHE_EXTS:
                self.add("WARN", "CACHE", rel, "cache/intermediate file in the code tree; keep caches in work/ "
                         "(packaging allowlist)")
            if a is None:
                if gaz:
                    self.add("FAIL", "DATA-GAZ", rel, "file name looks like a downloaded gazetteer / postal list "
                             "(external data)")
                elif size > 64_000:
                    self.add("WARN", "DATA-UNDECL", rel, f"{size:,}-byte data file with no declared origin; add it "
                             f"to manifest 'artifacts' (origin: {' | '.join(ALLOWED_ARTIFACT_ORIGINS)})")
                continue
            origin = str(a.get("origin", ""))
            verdict = "PASS"
            if origin not in ALLOWED_ARTIFACT_ORIGINS:
                self.add("FAIL", "DATA-ORIGIN", rel, f"artifact origin '{origin}' not allowed (must be one of "
                         f"{', '.join(ALLOWED_ARTIFACT_ORIGINS)})")
                verdict = "FAIL"
            elif gaz:
                self.add("WARN", "DATA-GAZ", rel, f"declared as '{origin}' but the name looks like a gazetteer; "
                         f"double-check it was not downloaded")
                verdict = "WARN"
            self.res.artifacts.append({"path": rel, "origin": origin, "how": a.get("how", ""), "verdict": verdict})


def audit_tree(root, manifest_path=None, requirements_path=None, exclude=(), use_installed_metadata=True):
    """Run every check on the code tree; returns AuditResult (see render_report)."""
    return _Auditor(root, manifest_path, requirements_path, exclude, use_installed_metadata).run()


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def render_report(res: AuditResult) -> str:
    overall = "FAIL" if res.n_fail else "PASS"
    out = [f"## Compliance audit ({res.when}, audit_compliance.py)", "",
           f"- Tree: `{res.root}` ({res.files_scanned} files, {res.py_files} Python)",
           f"- requirements: `{res.requirements_path}` | manifest: `{res.manifest_path}`",
           f"- Models listed: {len(res.models)}; combined exact params {res.total_params:,} "
           f"(cap {PARAM_CAP:,}; licenses allowed: {', '.join(ALLOWED_MODEL_LICENSES)})",
           f"- **Overall: {overall}** ({res.n_fail} FAIL, {res.n_warn} WARN)", "",
           "| # | Check | Result | Findings |", "|---|---|---|---|"]
    grouped = set()
    for i, (title, rules) in enumerate(GROUPS, 1):
        fs = [f for f in res.findings if f.rule in rules]
        grouped |= rules
        lvl = "FAIL" if any(f.level == "FAIL" for f in fs) else "WARN" if any(f.level == "WARN" for f in fs) else "PASS"
        out.append(f"| {i} | {title} | {lvl} | {len(fs)} |")
    if res.libs:
        out += ["", "### Libraries (requirements.txt)", "", "| package | version | license | verdict |",
                "|---|---|---|---|"]
        out += [f"| {d['name']} | {d['version']} | {d['license']} | {d['verdict']} |" for d in res.libs]
    if res.models:
        out += ["", "### Models (models_manifest.json)", "",
                "| model | role | license | exact params | counted from files | revision | verdict |",
                "|---|---|---|---|---|---|---|"]
        for d in res.models:
            p = f"{d['params']:,}" if isinstance(d["params"], int) and not isinstance(d["params"], bool) else repr(d["params"])
            c = f"{d['params_counted_from_files']:,}" if d["params_counted_from_files"] is not None else "-"
            out.append(f"| {d['id']} | {d['role']} | {d['license']} | {p} | {c} | {d['revision'][:12] or '-'} | "
                       f"{d['verdict']} |")
    if res.artifacts:
        out += ["", "### Data artifacts", "", "| path | origin | how | verdict |", "|---|---|---|---|"]
        out += [f"| {a['path']} | {a['origin']} | {a['how']} | {a['verdict']} |" for a in res.artifacts]
    fs = sorted(res.findings, key=lambda f: ({"FAIL": 0, "WARN": 1}.get(f.level, 2), f.rule, f.where))
    if fs:
        out += ["", "### Findings", ""]
        out += [f"- **{f.level}** `{f.rule}` {f.where} -- {f.message}" for f in fs]
    out += ["", f"RESULT: {overall}"]
    return _scrub("\n".join(out))


# ----------------------------------------------------------------------------------------------------------------
# self-test: plants violations in a fake tree and checks that each one is caught (and a clean tree passes)
# ----------------------------------------------------------------------------------------------------------------
_LGB_TEXT = """tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx=2
objective=binary sigmoid:1
feature_names=f0 f1 f2
feature_infos=[0:1] [0:1] [0:1]
tree_sizes=100 100

Tree=0
num_leaves=3
num_cat=0
split_feature=0 1
split_gain=1 1
threshold=0.5 0.25
decision_type=2 2
left_child=1 -1
right_child=-2 -3
leaf_value=-0.1 0.2 0.05
leaf_weight=1 1 1
leaf_count=1 1 1
internal_value=0 0
internal_weight=1 1
internal_count=3 2
is_linear=0
shrinkage=1


Tree=1
num_leaves=2
num_cat=0
split_feature=2
split_gain=1
threshold=0.75
decision_type=2
left_child=-1
right_child=-2
leaf_value=0.01 -0.02
leaf_weight=1 1
leaf_count=1 1
internal_value=0
internal_weight=1
internal_count=2
is_linear=0
shrinkage=0.1


end of trees

feature_importances:
f0=1

parameters:
[boosting: gbdt]
end of parameters
"""
_LGB_PARAMS = 2 + 3 + 1 + 2  # thresholds + leaf values over both trees


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _write_safetensors(path: Path, shapes: dict):
    header, off = {}, 0
    for k, shp in shapes.items():
        n = 4
        for d in shp:
            n *= d
        header[k] = {"dtype": "F32", "shape": list(shp), "data_offsets": [off, off + n]}
        off += n
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb + b"\0" * off)


def _build_bad_tree(d: Path, fake_key: str):
    _write(d / "src/ber/__init__.py", "")
    _write(d / "src/ber/llm_judge.py", (
        "import os\nimport groq\nfrom openai import OpenAI\nimport requests\n"
        "KEY = os.environ['GROQ_API_KEY']\n"
        f"FALLBACK = '{fake_key}'\n"
        "MODEL = 'llama-3.1-8b-instant'\n"
        "def judge(a, b):\n"
        "    client = groq.Groq(api_key=KEY)\n"
        "    r = requests.post('https://api.groq.com/openai/v1/chat/completions', json={'m': MODEL})\n"
        "    return r.json()\n"))
    _write(d / "src/ber/geo.py", (
        "from geopy.geocoders import Nominatim\nimport postal\nimport boto3\n"
        "URL = 'https://nominatim.openstreetmap.org/search?q=x'\n"
        "br = boto3.client('bedrock-runtime')\nloc = boto3.client('location')\n"))
    _write(d / "src/ber/translit.py", (
        "from unidecode import unidecode\nimport Levenshtein\nfrom bs4 import BeautifulSoup\n"
        "from rapidfuzz.distance import Levenshtein as RL\n"))
    _write(d / "src/ber/emb.py", (
        "from transformers import AutoModel\nfrom sentence_transformers import SentenceTransformer\n"
        "from huggingface_hub import InferenceClient\n"
        "m = AutoModel.from_pretrained('Qwen/Qwen2.5-3B-Instruct')\n"
        "e = SentenceTransformer('Alibaba-NLP/gte-multilingual-base', trust_remote_code=True)\n"
        "c = InferenceClient()\n"))
    _write(d / "requirements.txt", (
        "--extra-index-url https://example.org/simple\npolars\nlightgbm==4.7.0\nUnidecode==1.4.0\n"
        "aksharamukha==2.3\ngroq==0.25.0\nmystery-pkg-xyz==1.0\n"))
    _write(d / "models_manifest.json", json.dumps({
        "final_model": "master-bolt/lightgbm-matcher",
        "models": [
            {"id": "master-bolt/lightgbm-matcher", "kind": "gbdt", "license": "MIT", "params": "8B"},
            {"id": "Qwen/Qwen3-8B", "license": "Apache-2.0", "params": 8190735360,
             "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "license": "llama3.1", "params": 8030261248},
            {"id": "intfloat/multilingual-e5-small", "license": "MIT", "params": 118000000},
        ]}, indent=2))
    _write_safetensors(d / "models/extra.safetensors", {"w": (3, 4), "b": (4,)})
    _write(d / "src/ber/data/geonames_cities500.txt", "123\tParis\n")
    _write(d / ".env", f"GROQ_API_KEY={fake_key}\n")
    _write(d / "README.md", "Run: export OPENAI_API_KEY=... then python src/run_pipeline.py\n")


_REFERENCE_MODULES = (("er-data-loading", "io.py"), ("er-text-normalization", "normalize.py"),
                      ("er-text-normalization", "normalize_frame.py"), ("er-blocking-candidates", "blocking.py"),
                      ("er-pair-features", "features.py"), ("er-matcher-training", "model.py"),
                      ("er-f05-decisions", "metric.py"), ("er-f05-decisions", "decide.py"),
                      ("er-submission-packaging", "outputs.py"), ("er-challenge-playbook", "config.py"))


def _pins_for_tree(d: Path, base_pins: dict) -> list:
    """Exact pins for every third-party import in the tree, versions taken from the running interpreter."""
    local = {p.stem for p in d.rglob("*.py")} | {p.parent.name for p in d.rglob("*.py")}
    import_to_dist = {}
    for dist, mods in DIST_IMPORTS.items():
        for m in mods:
            import_to_dist.setdefault(m.split(".")[0], dist)
    pins = dict(base_pins)
    for p in d.rglob("*.py"):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = [a.name for a in n.names] if isinstance(n, ast.Import) else \
                [n.module] if isinstance(n, ast.ImportFrom) and n.module and n.level == 0 else []
            for m in mods:
                top = m.split(".")[0]
                if top in local or top in sys.stdlib_module_names or top == "__future__":
                    continue
                cand = _canon_dist(top)
                dist = cand if cand in ALLOW_LIBS else import_to_dist.get(top, cand)
                if dist not in pins:
                    try:
                        import importlib.metadata as md
                        pins[dist] = md.version(dist)
                    except Exception:
                        pins[dist] = "0.0.0"
    return [f"{k}=={v}" for k, v in sorted(pins.items())]


def _build_good_tree(d: Path, skills_dir: Path):
    copied = []
    for sub, fn in _REFERENCE_MODULES:
        src = skills_dir / sub / fn
        if src.exists():
            _write(d / "src/ber" / fn, src.read_text(encoding="utf-8"))
            copied.append(fn)
    _write(d / "src/ber/__init__.py", "")
    _write(d / "src/ber/booster_io.py", (
        "import lightgbm as lgb\nimport numpy as np\nimport polars as pl\n"
        "def load(path):\n    return lgb.Booster(model_file=path)\n"))
    _write(d / "src/ber/embed.py", (
        "import os\nos.environ.setdefault('HF_HUB_OFFLINE', '1')\n"
        "from sentence_transformers import SentenceTransformer\n"
        "E5_ID = 'intfloat/multilingual-e5-small'\n"
        "def encoder():\n"
        "    return SentenceTransformer(E5_ID, revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', device='cpu')\n"))
    _write(d / "src/run_pipeline.py", (
        "import argparse\nfrom ber import booster_io\n\ndef main():\n    ap = argparse.ArgumentParser()\n"
        "    ap.parse_args()\n\nif __name__ == '__main__':\n    main()\n"))
    _write(d / "models/lgbm_matcher.txt", _LGB_TEXT)
    _write(d / "src/ber/native_token_dict.tsv", "native\tlatin\n")
    _write(d / "README.md", "Reproduce: set HF_HUB_OFFLINE=1, then python src/run_pipeline.py --stage all\n")
    pins = _pins_for_tree(d, {"polars": "1.44.2", "numpy": "2.2.6", "lightgbm": "4.7.0",
                              "sentence-transformers": "6.1.0"})
    _write(d / "requirements.txt", "\n".join(pins) + "\n")
    _write(d / "models_manifest.json", json.dumps({
        "final_model": "master-bolt/lightgbm-matcher",
        "models": [
            {"id": "master-bolt/lightgbm-matcher", "kind": "gbdt", "role": "final matcher", "license": "MIT",
             "params": _LGB_PARAMS, "files": ["models/lgbm_matcher.txt"]},
            {"id": "intfloat/multilingual-e5-small", "kind": "hf-encoder", "role": "optional feature",
             "license": "MIT", "params": 117654272, "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3"},
        ],
        "artifacts": [{"path": "src/ber/native_token_dict.tsv", "origin": "learned-from-provided-train",
                       "how": "normalize.learn_native_dict on train links"}]}, indent=2))
    return copied


def self_test(base_dir=None) -> bool:
    base = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="audit_selftest_"))
    base.mkdir(parents=True, exist_ok=True)
    fake_key = "gsk" + "_" + "SELFTEST" + "0" * 26  # built at run time so this source holds no key-shaped literal
    bad, good = base / "bad_tree", base / "good_tree"
    _build_bad_tree(bad, fake_key)
    copied = _build_good_tree(good, Path(__file__).resolve().parent.parent)
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok = ok and bool(cond)

    print(f"[self-test] trees under {base}")
    check(count_lightgbm_params(good / "models/lgbm_matcher.txt") == _LGB_PARAMS, "LightGBM text param count exact")
    check(count_safetensors_params(bad / "models/extra.safetensors") == 16, "safetensors param count exact")
    rb = audit_tree(bad, use_installed_metadata=False)
    rep_b = render_report(rb)
    want_fail = {"IMP-API", "API-CALL", "ENV-KEY", "SECRET", "NET-CALL", "URL-FORBIDDEN", "MODEL-HOSTED",
                 "IMP-GEO", "IMP-ADDR", "AWS-SVC", "IMP-GPL", "IMP-SCRAPE", "MODEL-BANNED", "MODEL-UNLISTED",
                 "HF-INFER", "REQ-DENY", "MAN-PARAMS", "MAN-LICENSE", "MAN-KNOWN-MISMATCH", "FILE-UNLISTED",
                 "DATA-GAZ", "CRED-FILE"}
    want_warn = {"REQ-UNPINNED", "REQ-INDEX", "REQ-UNKNOWN", "TRC", "IMP-NET", "MAN-REVISION", "KEY-MENTION",
                 "MENTION", "MODEL-REV", "AWS-SDK"}
    got_fail, got_warn = rb.rules("FAIL"), rb.rules("WARN")
    for r in sorted(want_fail):
        check(r in got_fail, f"bad tree: FAIL {r}")
    for r in sorted(want_warn):
        check(r in got_warn, f"bad tree: WARN {r}")
    check(not any(f.where.endswith("translit.py:4") for f in rb.findings if f.rule == "IMP-GPL"),
          "`from rapidfuzz.distance import Levenshtein` is not flagged as GPL")
    check(fake_key not in rep_b and fake_key not in json.dumps([asdict(f) for f in rb.findings]),
          "secret never printed (redacted)")
    check(rep_b.rstrip().endswith("RESULT: FAIL"), "bad tree overall FAIL")
    rg = audit_tree(good, use_installed_metadata=False)
    rep_g = render_report(rg)
    print(f"  info good tree copied real modules: {', '.join(copied) or 'none found'}")
    check(rg.n_fail == 0, f"good tree: 0 FAIL (got {rg.n_fail}: {sorted(rg.rules('FAIL'))})")
    check(rg.n_warn == 0, f"good tree: 0 WARN (got {rg.n_warn}: {sorted(rg.rules('WARN'))})")
    check(rep_g.rstrip().endswith("RESULT: PASS"), "good tree overall PASS")
    check(any(m["params_counted_from_files"] == _LGB_PARAMS for m in rg.models), "final model params verified")
    if not ok:
        print(rep_b)
        print(rep_g)
    print("SELF-TEST " + ("PASS" if ok else "FAIL"))
    return ok


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def _default_root() -> Path:
    cand = Path.cwd() / "code" / "business_entity_resolution"
    if cand.exists():
        return cand
    return Path(__file__).resolve().parents[3] / "code" / "business_entity_resolution"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fair-play + license audit of the submission code tree.")
    ap.add_argument("root", nargs="?", help="code tree (default code/business_entity_resolution)")
    ap.add_argument("--manifest", help="models_manifest.json (default ROOT/models_manifest.json)")
    ap.add_argument("--requirements", help="requirements.txt (default ROOT/requirements.txt)")
    ap.add_argument("--exclude", nargs="*", default=[], help="extra glob patterns to skip")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN too")
    ap.add_argument("--json", help="write findings as JSON")
    ap.add_argument("--report", help="also write the Markdown report to this file")
    ap.add_argument("--no-installed-metadata", action="store_true",
                    help="do not read licenses of installed packages for unknown pins")
    ap.add_argument("--count-params", nargs="+", metavar="FILE", help="print exact params of model files and exit")
    ap.add_argument("--self-test", nargs="?", const="", metavar="DIR", help="run the planted-violation self-test")
    a = ap.parse_args(argv)
    if a.self_test is not None:
        return 0 if self_test(a.self_test or None) else 1
    if a.count_params:
        total = 0
        for f in a.count_params:
            n = count_params_file(f)
            print(f"{f}\t{n if n is not None else 'not countable (use safetensors or LightGBM text format)'}")
            total += n or 0
        print(f"TOTAL\t{total}\t({'<=' if total <= PARAM_CAP else '>'} {PARAM_CAP:,} cap)")
        return 0
    root = Path(a.root) if a.root else _default_root()
    if not root.is_dir():
        print(f"error: code tree not found: {root} (pass ROOT explicitly)", file=sys.stderr)
        return 2
    res = audit_tree(root, a.manifest, a.requirements, a.exclude, not a.no_installed_metadata)
    rep = render_report(res)
    print(rep)
    if a.report:
        _write(Path(a.report), rep + "\n")
    if a.json:
        _write(Path(a.json), json.dumps({"root": res.root, "when": res.when, "fail": res.n_fail, "warn": res.n_warn,
                                         "findings": [asdict(f) for f in res.findings], "libs": res.libs,
                                         "models": res.models, "artifacts": res.artifacts}, indent=2))
    return 1 if res.n_fail or (a.strict and res.n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
