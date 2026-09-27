---
name: er-compute-and-aws
description: Use when a Master Bolt entity-resolution stage is too slow or runs out of memory on the 16 GB Windows laptop, when deciding between the laptop, AWS EC2 (credits), SageMaker or Kaggle, when setting up the AWS CLI (aws login), budgets, vCPU quotas, launching or terminating an instance, uploading the dataset to S3, or checking AWS spend and credits.
---

# Compute: laptop discipline and AWS

## Overview
The laptop (i5-1235U 10C/12T, 15.7 GB RAM with often only 1–4 GB free, no CUDA) can develop everything on slices and, with sharding, run full scale slowly. A full train+test run is comfortable on an **EC2 r6a.4xlarge in ap-south-1 (16 vCPU / 128 GiB, $0.572/h on-demand, ~$0.21/h spot)**: ~1.5–3 h and ~$3 per run, far inside the $100–200 credits — **if nothing is left running**.

## When to use
- Memory errors, paging, or a stage taking hours; before the first full-scale run; any AWS action.
- Not for what to compute (see the step skills).

## Quick reference
| Need | Where |
|---|---|
| Exact AWS commands (sign-in, budget, quotas, key pair, SG, S3, launch, spot, run, download, teardown, audit, spend, Kaggle) | `aws-runbook.md` (this folder) |
| RAM guard, chunk sizes, worker counts | `memory_guard.py` → `src/ber/memory_guard.py`; `MemoryGuard().stage("features")`, `chunk_rows(bytes_per_row(df), guard=g)`; env `BER_MEM_BUDGET_GB`, `BER_MIN_FREE_GB` (default 0.75) |
| Prices, quotas, throughputs (measured) | `reference-aws-prices-throughput.md` |
| CLI state on this laptop | AWS CLI 2.37.3 installed; sign in with `aws login` (console credentials → temporary creds, auto-refresh up to 12 h) |

**Claude Code rule:** before any step that costs money (launch, spot request, bucket with data), state the expected cost and wait for the user's go-ahead. Never ask for, paste or store keys/passwords.

## Laptop or AWS? (measured throughputs, laptop under load)
| Stage | Laptop | Verdict |
|---|---|---|
| Normalization, 24M rows (distinct strings only) | US 6.2M rows 12 s (regex part), India 4.1M 49 s; full normalize_ref ~0.25 ms/row → ~12–15 min on 8 procs | laptop OK |
| Hash-vectorize 3 fields | US 264 s, India 418 s (10× slowdown once paging starts) | laptop OK if RAM free |
| Blocking, state-blocked + df-pruned | Karnataka 59k S1: 288 s, 1.6 GB peak; full ≈0.5–1.3 h | laptop OK per shard; AWS 20–45 min |
| rapidfuzz features | 0.7–3.6M pairs/s per metric | 40M pairs × 10 metrics ≈ 30–60 min |
| LightGBM | 2.8M row-rounds/s; 2M×40 in 0.8 GB | subsample negatives on laptop |
| Everything together (features 5–9 GB per 40M pairs + CSR 4–5 GB) | needs 15–25 GB | **AWS for the full run** |
| Transformer encoders (MiniLM/e5) on CPU | 35–105 strings/s → 26–84 h per 10M | **never at full scale on CPU**; GPU g4dn only if justified |
| Static embeddings (model2vec) | 2–3.7k strings/s → ~1–1.4 h per 10M | feasible |

## Laptop discipline
1. Close Edge/WhatsApp/Spotify before heavy stages; check free RAM (`python memory_guard.py`).
2. Work per country × state block; normalize distinct strings only; checkpoint every stage to `work/*.parquet`.
3. One heavy Python process at a time; Windows multiprocessing needs `if __name__ == "__main__":` and spawn-safe top-level functions.
4. Polars lazy scans + `collect()` per shard; float32; `del` + `gc.collect()` between shards.

## AWS path (details and exact commands in aws-runbook.md)
1. **Account:** Free-plan accounts can only launch ≤ 2 vCPU / 8 GiB types → upgrade to the **Paid plan** (credits carry over, expire 12 months after sign-up). Never create an AWS Organization / Control Tower / Identity Center org instance (credits expire immediately) — hence no `aws configure sso`.
2. **Identity:** IAM user `mlc-cli` with the runbook's least-privilege policy + `SignInLocalDevelopmentAccess`, MFA; then `aws configure set region ap-south-1` → `aws login` → `aws sts get-caller-identity`.
3. **Guardrails:** budget with e-mail alert that **excludes credits** (else it never fires); quota requests on day 1 (EC2 standard on-demand `L-1216C47A` and spot `L-34B43A08` default 5 vCPU → ask 32; G/VT GPU quota default 0).
4. **Run:** S3 bucket in ap-south-1 → upload dataset once, code each run → launch r6a.4xlarge (Ubuntu 24.04 x86_64, 100 GB gp3, SSH only from your IP) → `run_stages.sh` in tmux with per-stage S3 checkpoints (spot-safe) → download `output/` + logs.
5. **Teardown after every run:** terminate, delete volumes/snapshots, run the audit (every Region), check spend (`ce get-cost-and-usage`, `freetier get-account-plan-state`).
Zero-cost fallback: Kaggle notebook (≈30 GB RAM CPU sessions; upload data as a *private* dataset). SageMaker is ~2.2× the EC2 price with 0 default quotas for big instances — not the main path.

## Common mistakes
| Mistake | Fix |
|---|---|
| Staying on the Free plan and wondering why r6a is refused | upgrade to Paid (credits kept) |
| Creating an Organization / Identity Center to "use SSO" | never; use an IAM user + `aws login` |
| Budget includes credits → alert never fires | `IncludeCredit: false` |
| Graviton (`r7g`) to save money | sparse-dot-topn has no aarch64 wheel → x86 only |
| Forgetting quotas (5 vCPU default) until launch day | request on day 1 |
| Stopping instead of terminating (EBS + IPv4 still bill) | terminate + delete volumes, then audit |
| Uploading competition data to Bedrock/SageMaker Canvas/any hosted model | forbidden (fair-play); only your own EC2/S3 |
| CRLF `.sh` user-data from Windows | write LF |
| Inline JSON args in PowerShell 5.1 | `file://x.json` |
