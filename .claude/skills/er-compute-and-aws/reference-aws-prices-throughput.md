> Research notes captured 2026-09-25 while building these skills (measured on the real data on the team laptop). Paths such as `research/...`, `scratchpad/...` or `blocking_exp/...` point to a temporary scratch folder that is NOT part of the project — rely on the numbers and conclusions, and re-run measurements with the project code when needed. Sections marked PENDING/TODO were not finished.

# R4 — AWS compute plan for Amazon ML Challenge 2026 (Business Entity Resolution), team Master Bolt

Status: IN PROGRESS (written incrementally; sections marked TODO are not finished yet).
Date of research: 2026-09-25. Prices from AWS public bulk price files published 2026-09-24 (cached under research/pricing/).

## 0. TL;DR
TODO

## 1. AWS Free Tier (2025+ model) — what the credits can and cannot do

Verified 2026-09-25 against AWS docs (sources at the end of this section). Applies to accounts created **on or after 15 July 2025**.

| Question | Answer (verified) |
|---|---|
| Credits | **USD 100 on sign-up** (either plan) + **up to USD 100 more**: USD 20 each for 5 console "Explore AWS" activities: (1) launch an EC2 instance, (2) use a model in the Bedrock playground, (3) create a budget (AWS Budgets), (4) create a web app with Lambda, (5) create an RDS database. The activities themselves are billed against the credits. They must be completed within 6 months of opening the account. |
| Credit expiry | **12 months** from account creation (Free Tier terms, updated 9 July 2025). |
| Free plan duration | Ends at **6 months OR when credits run out, whichever comes first**. The account then closes. You have **90 days** to upgrade to Paid before AWS deletes the account and all its resources. |
| Charges on the Free plan | None. You can't exceed the credits. |
| **EC2 on the Free plan** | Only instance types marked *Free tier eligible*: **t3.micro, t3.small, t4g.micro, t4g.small, c7i-flex.large, m7i-flex.large** (the burstable-instances page also lists t8i.micro/small). The biggest is **m7i-flex.large = 2 vCPU / 8 GiB**, which is useless for this workload: less than the laptop. r6i/r7i/c6i/g4dn/g5/g6 need the **Paid plan**. List the eligible types with `aws ec2 describe-instance-types --filters Name=free-tier-eligible,Values=true`. |
| Other Free-plan restrictions | No "services that could deplete your credits" or hardware purchases: e.g. Savings Plans, Reserved Instances, some Marketplace offers. Only *Always Free* offers are active. Short-term trials are listed as Paid-plan features, although the SageMaker 2-month trial is advertised for "Free and Paid plan". Free-plan accounts are not eligible for other promotional credits or discounts. **Joining AWS Organizations or a Control Tower landing zone auto-upgrades the account to Paid, and the remaining Free Tier credits expire immediately** (FAQ Q9). So do NOT create an AWS Organization on this account. IAM Identity Center's *organization instance* requires Organizations, so use a plain IAM user instead (runbook 4.2). |
| **Upgrading to Paid keeps the credits?** | **Yes.** "If you upgrade to a paid plan, any remaining Free Tier credit balance will automatically apply to your AWS bills. All Free Tier credits must be used within 12 months of your account creation date." Upgrading via Organizations/Control Tower is the exception: credits are lost. |
| What credits don't cover | Per the AWS Promotional Credit Terms: Marketplace, Route 53 domain registration/transfer, upfront fees for Savings Plans/RIs, Mechanical Turk, Managed Services, Professional Services, Training/Certification, Enterprise/ineligible Support, crypto mining. EC2 (On-Demand and Spot), EBS, S3, SageMaker and data transfer are not on the exclusion list, so the credits apply to them. |
| Paid-plan billing risk | On the Paid plan, anything above the remaining credit is billed to your card at standard rates. You need a **budget + alarm** (runbook 4.3) and must **terminate** instances and **delete** volumes when done. |
| SageMaker AI free trial | "Free Tier usage per month for the first 2 months", starting from the month you create your first SageMaker resource: **250 h ml.t3.medium** (Studio notebooks OR notebook instances), **50 h m4.xlarge/m5.xlarge training**, 125 h m4/m5.xlarge real-time inference, 25 h ml.m5.4xlarge Data Wrangler, 160 h Canvas. **It does not cover memory-heavy or GPU instances.** ml.t3.medium = 2 vCPU / 4 GiB; ml.m5.xlarge = 4 vCPU / 16 GiB. Both are no bigger than the laptop, so the trial is only useful for light orchestration or a smoke test. |

### 1.1 Default service quotas: the real gate (plan 1-3 days ahead)
New accounts start with low vCPU quotas (defaults from the EC2 instance-type quotas page, verified 2026-09-25):

| Quota (EC2, per Region) | Quota code | Default | Needed for |
|---|---|---|---|
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances | `L-1216C47A` | **5 vCPUs** | r6i.2xlarge needs 8; r6i.4xlarge / r6a.4xlarge / m6i.4xlarge need 16; c6i.8xlarge needs 32 |
| All Standard (A, C, D, H, I, M, R, T, Z) Spot Instance Requests | `L-34B43A08` | **5 vCPUs** | same as above, for Spot |
| Running On-Demand G and VT instances | `L-DB2E81BA` | **0** | g4dn / g5 / g6.xlarge need 4 |
| All G and VT Spot Instance Requests | `L-3819A6DF` | **0** | same, for Spot |

Quota codes were cross-checked against the open-source Zuul nodepool AWS adapter, which maps the a/c/d/h/i/m/r/t/z families to L-1216C47A (on-demand) and L-34B43A08 (spot), and g/vt to L-DB2E81BA / L-3819A6DF.

SageMaker defaults, from the SageMaker quotas page cached in `research/pricing/sm_quotas.html`:
- **0** for `ml.r5.4xlarge`, `ml.r7i.4xlarge`, `ml.m5.4xlarge` and `ml.g4dn.xlarge`, for *training job*, *processing job* and *notebook instance* usage.
- **6** for `ml.t3.medium` notebook instances.

**So SageMaker also needs a quota request** before any memory-heavy job can run (Service Quotas -> Amazon SageMaker, e.g. "ml.r5.4xlarge for processing job usage").

Some accounts show higher applied values, because AWS raises quotas automatically with usage. Check first:
```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --region ap-south-1 --query "Quota.Value"
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-1216C47A --desired-value 32 --region ap-south-1
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-34B43A08 --desired-value 32 --region ap-south-1
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8  --region ap-south-1   # only if you want a GPU
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region ap-south-1 --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" --output table
```
Small increases (e.g. 5 -> 16 or 32 vCPUs) are often approved within minutes to hours. GPU requests on brand-new accounts often need a support case, can take 1-2 days, or can be denied (this is general experience, not an AWS SLA). **Request on day 1, in the Region you will use.** Quotas are per Region.

### 1.2 Bottom line for the team
1. **Upgrade the account to the Paid plan**: console -> "Upgrade plan". Do *not* use Organizations or Control Tower. The credits (100 + up to 100) carry over and expire 12 months after sign-up. Without upgrading, the largest instance is 2 vCPU / 8 GiB.
2. Earn the extra credits cheaply:
   - Create the budget (step 4.3): +$20.
   - Launch and terminate one t3.micro: +$20.
   - Lambda, Bedrock playground and RDS are +$20 each for a few cents of use. **Do not paste any competition data into Bedrock.** Use a generic prompt such as "hello", or skip that activity.
3. File the quota increases on day 1 (above).
4. Set a USD 20-50 budget with email alerts before launching anything.

Sources:
- aws.amazon.com/free/terms (updated 9 Jul 2025)
- docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/free-tier-plans.html
- docs.aws.amazon.com/accounts/latest/reference/bcm-lite-free-tier-credits.html
- docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-free-tier-usage.html
- docs.aws.amazon.com/AWSEC2/latest/UserGuide/burstable-performance-instances.html
- aws.amazon.com/free/free-tier-faqs (Q3, Q5, Q9, Q11)
- repost.aws/knowledge-center/aws-free-tier-account-start-expire
- aws.amazon.com/sagemaker/ai/pricing (Free Tier table)
- docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html
- aws.amazon.com/awscredits (Promotional Credit Terms)
- opendev.org/zuul/nodepool (driver/aws/adapter.py, quota codes)


## 2. Prices (on-demand, spot; us-east-1 and ap-south-1 Mumbai)

### 2.0 Where the numbers come from
- **Price files.**
  - EC2 on-demand: AWS public price JSON (`b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/...`), publication date **2026-09-24T16:51Z**, cached as `research/pricing/od_use1.json` and `od_aps1.json`.
  - SageMaker: public bulk price CSV (offer `AmazonSageMaker`), published **2026-09-24T23:59Z**, cached as `sm_us-east-1.csv` and `sm_ap-south-1.csv`.
  - EBS: `ebs.json` (2026-09-24).
  - S3: bulk CSV (2026-09-18).
- **Spot.** From the public feed behind the EC2 Spot pricing page (`spot.js`). It carries **no timestamp**, and a few rows look stale or odd: g6.xlarge us-east-1 at $0.105 is -87% vs OD, well below the advisor's quoted saving. **Treat spot numbers as indicative.** The authoritative value comes from `aws ec2 describe-spot-price-history --instance-types r6a.4xlarge --product-descriptions "Linux/UNIX" --start-time <now> --region ap-south-1` once the CLI is configured.
- **Interruption frequency.** From the Spot Instance Advisor feed (`spot_advisor.json`), shown as "save X% / interruption band".
- **Parsers** (re-runnable, stdlib + json/csv): `research/pricing/parse_prices.py` (EC2 OD + spot + advisor) and `research/pricing/parse_sm2.py` (SageMaker per component). Raw outputs are in `research/pricing/prices_table.txt` and `sm_table.txt`.
- **Pitfall when parsing the SageMaker CSV.** Most rows have an empty `Instance Type` column. The instance lives in `usageType`, e.g. `USE1-Notebk:ml.t3.medium`, `APS3-Train:ml.r5.4xlarge`, `USE1-Studio:JupyterLab-ml.r6i.4xlarge`, so match on `usageType` (see `parse_sm2.py`).
- **The pricing MCP server** (`awspricing get_pricing`) needs AWS credentials and failed with "Unable to locate credentials". The public files above need no credentials.

### 2.1 EC2 Linux on-demand and spot, USD per hour

| type | vCPU | RAM | us-east-1 OD | us-east-1 spot | ap-south-1 OD | ap-south-1 spot | advisor us-east-1 | advisor ap-south-1 |
|---|---|---|---|---|---|---|---|---|
| **Memory-optimized (x86)** | | | | | | | | |
| r6i.2xlarge | 8 | 64 GiB | 0.5040 | 0.2428 | 0.5200 | 0.2197 | save 56%, intr >20% | save 62%, intr >20% |
| r6i.4xlarge | 16 | 128 GiB | 1.0080 | 0.4426 | 1.0400 | 0.3620 | 56%, >20% | 71%, >20% |
| **r6a.2xlarge** (AMD) | 8 | 64 GiB | 0.4536 | 0.2182 | **0.2860** | **0.1187** | 57%, >20% | 47%, 5-10% |
| **r6a.4xlarge** (AMD) | 16 | 128 GiB | 0.9072 | 0.3793 | **0.5720** | **0.2129** | 60%, >20% | 51%, **<5%** |
| r7i.2xlarge | 8 | 64 GiB | 0.5292 | 0.2435 | 0.5460 | 0.2015 | 57%, >20% | 53%, 15-20% |
| r7i.4xlarge | 16 | 128 GiB | 1.0584 | 0.4588 | 1.0920 | 0.3983 | 64%, >20% | 62%, 5-10% |
| r7a.2xlarge (AMD) | 8 | 64 GiB | 0.6086 | 0.2968 | n/a | n/a | 58%, >20% | n/a |
| r7a.4xlarge (AMD) | 16 | 128 GiB | 1.2172 | 0.5987 | n/a | n/a | 63%, 15-20% | n/a |
| r8i.4xlarge | 16 | 128 GiB | 1.1114 | - | 1.1466 | - | 56%, >20% | 68%, 5-10% |
| r6i.8xlarge | 32 | 256 GiB | 2.0160 | 0.8785 | 2.0800 | 0.9040 | 62%, >20% | 74%, >20% |
| **General purpose / compute (x86)** | | | | | | | | |
| m6i.2xlarge | 8 | 32 GiB | 0.3840 | 0.1645 | 0.4040 | 0.1446 | 54%, 15-20% | 62%, >20% |
| m6i.4xlarge | 16 | 64 GiB | 0.7680 | 0.3205 | 0.8080 | 0.3364 | 66%, >20% | 58%, >20% |
| m7i.4xlarge | 16 | 64 GiB | 0.8064 | 0.3939 | 0.8484 | 0.3330 | 55%, >20% | 58%, 5-10% |
| m7i-flex.4xlarge | 16 | 64 GiB | 0.7661 | 0.3519 | 0.8060 | 0.2781 | 56%, >20% | 59%, <5% |
| c6i.8xlarge | 32 | 64 GiB | 1.3600 | 0.6765 | 1.3600 | 0.4575 | 64%, >20% | 63%, >20% |
| c6a.8xlarge (AMD) | 32 | 64 GiB | 1.2240 | 0.6021 | 0.7480 | 0.3128 | 51%, 15-20% | 53%, <5% |
| c7i.8xlarge | 32 | 64 GiB | 1.4280 | 0.6582 | 1.4280 | 0.5527 | 66%, >20% | 62%, <5% |
| **Graviton (arm64): see pitfall below** | | | | | | | | |
| r7g.4xlarge | 16 | 128 GiB | 0.8568 | 0.3802 | 0.6006 | 0.2306 | 61%, >20% | 59%, 15-20% |
| **GPU** | | | | | | | | |
| g4dn.xlarge (1x T4 16 GB) | 4 | 16 GiB | 0.5260 | 0.2204 | 0.5790 | 0.1898 | 60%, 10-15% | 54%, 15-20% |
| g4dn.2xlarge (1x T4) | 8 | 32 GiB | 0.7520 | 0.3063 | 0.8280 | 0.3004 | 61%, >20% | 58%, >20% |
| g5.xlarge (1x A10G 24 GB) | 4 | 16 GiB | 1.0060 | 0.5006 | 1.2080 | 0.5065 | 56%, >20% | 54%, >20% |
| g6.xlarge (1x L4 24 GB) | 4 | 16 GiB | 0.8048 | 0.1050 (suspect) | 0.9664 | - | 49%, >20% | 53%, >20% |
| g6f.xlarge (fractional L4) | 4 | 16 GiB | 0.2375 | - | 0.2852 | - | 62%, 5-10% | 22%, <5% |
| **Free-plan eligible (for reference)** | | | | | | | | |
| m7i-flex.large | 2 | 8 GiB | 0.0958 | 0.0446 | 0.1008 | 0.0315 | | |
| t3.small | 2 | 2 GiB | 0.0208 | 0.0080 | 0.0224 | 0.0073 | | |

Observations:
- **Mumbai r6a is the price outlier**: r6a.4xlarge (16 vCPU, 128 GiB) costs **$0.572/h on-demand**, 37% cheaper than in us-east-1, and 45% cheaper than r6i.4xlarge in Mumbai. Its Mumbai spot price is **$0.213/h**, with advisor interruption **<5%**, the best band. It is the best RAM-per-dollar x86 box in either Region. Verified in the raw JSON: `r6a.4xlarge` = `0.5720000000` for "Asia Pacific (Mumbai)", Linux, 128 GiB.
- Mumbai also gives the lowest network latency for a laptop in India, which speeds up the ~3 GB upload.
- In us-east-1 the cheapest 128 GiB x86 box is r6a.4xlarge at $0.907/h, or $0.379 spot, with advisor interruption >20%.
- **Graviton (r7g) is cheaper but not recommended.** `sparse-dot-topn` 1.2.0 (Apache-2.0) ships **no manylinux aarch64 wheels** on PyPI (checked 2026-09-25: 4 x86_64 wheels, 0 aarch64), so it would need a C++ source build (cmake + nanobind). LightGBM 4.7.0, faiss-cpu 1.15.1 and duckdb 1.5.5 do have aarch64 wheels. Stick to x86_64 (r6a/r6i/c6a).
- For GPUs, g4dn.xlarge (T4) is the cheapest general-purpose NVIDIA box, at $0.526-0.579/h OD. It needs the G/VT quota (default 0).

### 2.2 SageMaker AI on-demand, USD per hour, per component

Parsed from the bulk CSV, `usageType` components: `Notebk` = notebook instance, `Studio:JupyterLab`, `Train`, `Processing`, `Tsform` = batch transform, `Host`. "nan" means that component is not offered for that instance type in that Region.

| instance | us-east-1 Notebook | us-east-1 JupyterLab | us-east-1 Train | us-east-1 Processing | ap-south-1 Notebook | ap-south-1 Train | ap-south-1 Processing |
|---|---|---|---|---|---|---|---|
| ml.t3.medium (2 vCPU / 4 GiB) | 0.0500 | 0.0500 | n/a | 0.0500 | 0.0540 | n/a | 0.0540 |
| ml.m5.xlarge (4 / 16) | 0.2300 | 0.2300 | 0.2300 | 0.2300 | 0.2420 | 0.2420 | 0.2420 |
| ml.m5.4xlarge (16 / 64) | 0.9220 | 0.9220 | 0.9220 | 0.9220 | 0.9700 | 0.9700 | 0.9700 |
| ml.m6i.4xlarge (16 / 64) | 0.9220 | 0.9220 | 0.9220 | 0.9220 | 0.9700 | 0.9700 | 0.9700 |
| ml.r5.4xlarge (16 / 128) | 1.2100 | 1.2100 | 1.2100 | 1.2100 | 1.2480 | 1.2480 | 1.2480 |
| ml.r6i.4xlarge (16 / 128) | 1.2100 | 1.2100 | **n/a** | **n/a** | 1.2480 | **n/a** | **n/a** |
| ml.r7i.2xlarge (8 / 64) | 0.6350 | 0.6350 | 0.6350 | 0.6350 | 0.6550 | 0.6552 | 0.6552 |
| ml.r7i.4xlarge (16 / 128) | n/a | 1.2700 | 1.2701 | 1.2701 | 1.3100 | 1.3104 | 1.3104 |
| ml.r6i.8xlarge (32 / 256) | 2.4190 | 2.4190 | n/a | n/a | 2.4960 | n/a | n/a |
| ml.c6i.8xlarge (32 / 64) | 1.6320 | 1.6320 | 1.6320 | 1.6320 | 1.6320 | 1.6320 | 1.6320 |
| ml.g4dn.xlarge (T4) | 0.7360 | 0.7364 | 0.7360 | 0.7360 | 0.8110 | 0.8110 | 0.8110 |
| ml.g5.xlarge (A10G) | 1.4100 | 1.4100 | 1.4080 | 1.4080 | 1.6900 | 1.6900 | 1.6910 |
| ml.g6.xlarge (L4) | 1.1270 | 1.1270 | 1.1270 | 1.1267 | 1.3530 | 1.3530 | 1.3530 |

- **The SageMaker premium over raw EC2 is 20-120%.**
  - r6i.4xlarge: $1.008 on EC2 vs ml.r6i.4xlarge $1.21 (+20%) in us-east-1.
  - Against the cheapest comparable EC2 box in Mumbai (r6a.4xlarge, $0.572), SageMaker Processing on ml.r5.4xlarge ($1.248) is **2.2x** more expensive.
- **ml.r6i.\* has no Training or Processing SKU.** For 128 GiB jobs use ml.r5.4xlarge or ml.r7i.4xlarge.
- SageMaker Managed Spot Training (`TrSpt` usage type) exists and bills at spot rates. It is only available for Training jobs, not Processing.

### 2.3 Storage, network and other line items

| item | us-east-1 | ap-south-1 | our usage and cost |
|---|---|---|---|
| EBS gp3 storage | $0.08 / GB-month | $0.0912 / GB-month | 100 GB for 1 day = 100 x 0.08 / 30 = **$0.27** (Mumbai $0.30). Billed per second while the volume exists, **even when the instance is stopped**. |
| EBS gp3 baseline | 3,000 IOPS + 125 MB/s included | same | enough. Extra throughput costs $40.96 (use1) / $46.69 (aps1) per GiB/s-month; not needed |
| EBS snapshot | $0.05 / GB-month | $0.05 / GB-month | only if you keep a snapshot; delete it afterwards |
| S3 Standard | $0.023 / GB-month | $0.025 / GB-month | 5 GB (parquet + outputs) for 1 month = **$0.12** |
| S3 requests | PUT $0.005 / 1k, GET $0.0004 / 1k | same | negligible (<$0.01) |
| Data transfer IN | free | free | upload of ~3 GB = $0 |
| Data transfer OUT to internet | first **100 GB/month free** across all services and Regions, then about $0.09/GB (use1) | about $0.11-0.13/GB after the free 100 GB | downloading outputs (~0.2-1 GB) = $0 |
| S3 <-> EC2, same Region | free | free | put the bucket in the same Region as the instance |
| Public IPv4 address | $0.005 / h per in-use IP | same | $0.12 per 24 h. The legacy 750 h/month IPv4 allowance belongs to the old 12-month EC2 free tier; assume you pay it, or it comes out of credits. |
| CloudWatch basic metrics; AWS Budgets cost budget with email alerts | free | free | $0 (budget *actions* beyond the first two cost about $0.10/day each; we don't need actions) |

Sources for 2.3: `research/pricing/ebs.json` and the S3 CSVs; aws.amazon.com/ec2/pricing/on-demand (Data Transfer: "100 GB of free data transfer out to the internet free each month, aggregated across all AWS Services and Regions"); AWS News Blog "New – AWS Public IPv4 Address Charge" ($0.005/IP/h from 1 Feb 2024).


## 3. Workload estimate and recommended configuration

### 3.1 Measured stage throughputs (laptop; all real runs on this project's data)
Hardware: i5-1235U (2 P-cores + 8 E-cores, 12 threads), Windows 11, 15.7 GB RAM. Other agents and apps were using **25-100% of the CPU** during every run, so these are pessimistic numbers.

| stage | what was run | measured | source |
|---|---|---|---|
| Normalization | polars regex lowercase / punctuation / NULL-token strip + `anyascii` transliteration on the non-ASCII unique subset, name + address | US train pool **6.19M rows in 12.3 s**; India train pool (752k native-script rows) **4.13M rows in 49.4 s**; peak RSS 3.4 GB | `blocking_exp/build.log`, `blocking_exp/common.py` |
| Char-3gram vectorization | `HashingVectorizer(analyzer="char_wb", ngram_range=(3,3), n_features=2**20, alternate_sign=False)` on 3 fields (name, address, key), 500k-row chunks, joblib | US 6.19M rows: first 4M in 13 s, then slowed about 10x under memory pressure, **264 s total**; India 4.13M rows: **418 s**. nnz: US name 112M (18/row), address 170M (27.5/row); India name 64M, address 187M (45/row) | `blocking_exp/vec.log` |
| Sparse top-k (`sparse_dot_topn` 1.2.0 `sp_matmul_topn`, 10 threads) | 10k US S1 names x 400k S3 names, char_wb 3-gram TF-IDF, top_n=20 | `max_df=1.0`: 29.5 s = **0.14 G pairs/s**; `max_df=0.05`: 20.7 s = 0.19 G/s; **`max_df=0.01`: 3.1 s = 1.27 G/s** (nnz/row drops 20.3 -> 7.7). Pruning very frequent n-grams gives a 9.5x speedup. | `r4_bench/bench_topn_prune.out` |
| Sparse top-k, one real India state block | Karnataka: 59k S1 x 291k pool, name char_wb 3-gram | S1->pool 188.6 s (0.09 G pairs/s); pool->S1 71.2 s | `bench/IN_KA_tfidf.log` |
| TF-IDF memory | `TfidfVectorizer(char_wb, (2,4))` fit on 350k strings | **MemoryError** on the laptop. The sklearn vocabulary dict does not scale: use `HashingVectorizer` + your own IDF. | `bench/IN_KA_tfidf.log` |
| rapidfuzz pairwise (`process.cpdist`, 1M random pairs, `workers=-1`) | JaroWinkler(name) **3.56 M pairs/s**; Indel ratio(address) **4.03 M/s**; partial_ratio(name) **1.27 M/s**; token_set_ratio(name) **1.04 M/s**; token_sort_ratio(address) **0.73 M/s**. Single-thread is 3.5-6x slower. | | `r4_bench/bench_feat_lgb.out` |
| LightGBM 4.7.0 | 2M x 40 float32, binary, 127 leaves, 200 rounds, 12 threads | Dataset construct 2.4 s; train **142 s = 2.8 M row-rounds/s**; predict 0.22 M rows/s; RSS 0.77 GB | `r4_bench/bench_feat_lgb.out` |
| Transformer encoder, CPU (torch 2.6 CPU) | all-MiniLM-L6-v2 / paraphrase-multilingual-MiniLM-L12-v2 / multilingual-e5-small, names | **35-105 strings/s** (10 threads). 10M strings would take 26-84 h: infeasible on the laptop. | `bench_encoders_results.json`, `bench_threads_results.txt` |
| Cross-encoder, CPU | ms-marco-MiniLM-L6-v2, mmarco-mMiniLMv2-L12 | **14-16 pairs/s**. 10M pairs would take 175-205 h: infeasible. | `bench_encoders_results.json` |
| Static embeddings, CPU (model2vec 0.9.0, MIT) | potion-base-8M / potion-multilingual-128M | **2.0-3.7k strings/s** single process. 10M full strings take 1.0-1.4 h. | `bench_model2vec_results.txt` |

### 3.2 Size of the full job
- **Records**: train S1 2.21M + S2 5.03M + S3 5.29M = 12.5M; test S1 1.73M + S2 4.89M + S3 5.08M = 11.7M.
- **Blocking pair-work.** Sum over blocks of |S1_block| x |pool_block|, computed by `r4_bench/pairwork.py` (DuckDB on the parquet files; output in `pairwork.json`). "State" = last comma component of the S1 address. Pool per block is approximated as the country ratio x |S1_block|.

| split / country | S1 | pool | whole-country pairs | state-blocked pairs | reduction |
|---|---|---|---|---|---|
| train US | 1.32M | 6.19M | 8.2e12 | 2.8e11 | 29x |
| train India | 0.88M | 4.13M | 3.7e12 | 2.7e11 | 13.5x |
| test US | 0.66M | 3.82M | 2.5e12 | 8.6e10 | 29x |
| test India | 0.81M | 4.72M | 3.8e12 | 2.8e11 | 13.5x |
| test France | 0.26M | 1.43M | 3.7e11 | 9.5e10 | 3.9x (only 3 regions; block by city/street inside region) |
| **total** | | | **1.9e13** | **1.0e12** | |

- **Top-k cost per field and direction**:
  - At 0.14 G pairs/s (no df pruning, measured on the laptop), 1.0e12 pairs take about **2.0 h**.
  - At 1.27 G/s (max_df=0.01) they take about **13 min**.
  - With 3 fields (name, address, key) x 2 directions that becomes 4-12 h (unpruned) or 0.5-1.3 h (pruned) **on the laptop**.
  - Whole-country blocking without pruning (1.9e13 pairs) would take about 38 h per field on the laptop. So state or city blocking plus df-pruning is required on any hardware.
- **Candidate pairs**: 20-40M (train) + 15-30M (test). At about 30 float32 features + 2 int32 ids, that is **5-9 GB per 40M pairs** in memory. Never materialize Python `str` lists for 40M pairs, which costs about 10 GB: gather strings per chunk of 2-5M pairs.
- **Memory peak, one split in one process**:
  - normalized frames: about 2.5 GB
  - hashed CSR matrices for 3 fields: about 4-5 GB, plus transposes
  - candidate features: 5-9 GB
  - LightGBM on 20-40M x 40: raw 3.2-6.4 GB + bins 1-1.6 GB
  - total **about 15-25 GB** with careful chunking, 40+ GB if sloppy.

  **64 GB is the floor and 128 GB is comfortable.** The laptop (1-4 GB free) cannot do a full run without heavy sharding.

### 3.3 Projected wall-clock on the recommended box (r6a.4xlarge: 16 vCPU AMD EPYC Milan, 128 GiB)
Assumption: a clean 16-vCPU cloud box is about **2x the loaded laptop** on multi-threaded stages. This is an estimate, not measured on AWS. Verify with the smoke test in runbook step 4.8, which prints stage timings.

| stage | laptop-measured basis | est. on r6a.4xlarge |
|---|---|---|
| boot + `pip install` + `aws s3 sync` of 1-2.5 GB | - | 10-15 min |
| normalize 24.2M rows (train + test) | 6.2M US in 12 s; 4.1M India in 49 s | 2-4 min |
| hash-vectorize 3 fields, 24.2M rows | 10.3M in 264 + 418 s (with memory pressure) | 5-10 min |
| sparse top-k, state/city-blocked, 3 fields x 2 directions, max_df pruning | 0.5-1.3 h pruned | **20-45 min** (up to 3 h unpruned) |
| pair features for about 60M pairs (about 10 rapidfuzz metrics + numeric features, chunked) | 0.7-4 M pairs/s per metric | 15-30 min |
| LightGBM train (subsample to about 10M rows x 40 features, 500 rounds) | 2.8 M row-rounds/s gives 30 min | 10-15 min |
| LightGBM predict about 30M test pairs + exclusivity assignment + TSV write | 0.22 M rows/s at 200 rounds | 5-10 min |
| **total, one full train + test run** | | **about 1.5-3 h** |
| optional: model2vec embeddings of 24M strings (16 processes) | 2-3.7k strings/s per process | 15-30 min |
| optional: MiniLM-class transformer, 24M strings, **CPU** | 35-105/s on laptop | 2x the laptop gives 30-95 h; even an optimistic 300-500/s gives 13-22 h: **don't** |
| optional: MiniLM-class transformer, 24M short strings, **GPU g4dn.xlarge (T4)** | sbert.net lists 18,000 / 750 queries/s (GPU / CPU) for MiniLM-L6 models, GPU type unstated; a T4 is roughly 1/3-1/2 of that, i.e. about 6-9k/s | **about 45-70 min** |

### 3.4 Recommendation and cost (USD)

**Primary: EC2 `r6a.4xlarge` in `ap-south-1` (Mumbai), Ubuntu 24.04 LTS x86_64, 100 GB gp3 root, one S3 bucket in the same Region.**
- $0.572/h on-demand, or about $0.21/h spot (advisor: <5% interruption, save 51%).
- 128 GiB covers every stage without sharding, and the 16 vCPUs halve the wall-clock vs 8.

| scenario | compute | EBS 100 GB | IPv4 | S3 | **total** |
|---|---|---|---|---|---|
| 1 full run on-demand, 3 h + 1 h slack | 4 h x 0.572 = $2.29 | 1 day = $0.30 | $0.02 | $0.12/month | **about $2.7** |
| 1 full run on spot | 4 h x about 0.21 = $0.85 | $0.30 | $0.02 | $0.12 | **about $1.3** |
| realistic competition use: 10 full runs + 20 h of interactive debugging on-demand | 60 h x 0.572 = $34 | 100 GB x 10 days = $3.0 | $0.30 | $0.12 | **about $38** |
| same on us-east-1 r6a.4xlarge ($0.907/h) | 60 h = $54 | $2.7 | $0.30 | $0.12 | about $57 |
| add a GPU encoding pass: g4dn.xlarge Mumbai, 1.5 h on-demand | 1.5 x 0.579 = $0.87 | 50 GB x 1 day = $0.15 | | | about $1 per pass |
| alternative: SageMaker Processing ml.r5.4xlarge Mumbai, 10 runs x 3 h | 30 h x 1.248 = $37 | incl. (volume per job) | | | about $37 for 10 runs vs about $17 on EC2 on-demand |

- **Cheaper fallback**: `r6a.2xlarge` Mumbai (8 vCPU, 64 GiB) at $0.286/h OD / $0.119 spot. The same run takes about 1.5-2x longer and needs tighter chunking, at about $1.5-2 per run.
- **Compute-heavy variant**: `c6a.8xlarge` Mumbai (32 vCPU, 64 GiB) at $0.748/h. Only worth it if top-k and rapidfuzz dominate and memory stays under 50 GB.
- **Why not us-east-1**: the same instance costs 59% more (r6a.4xlarge $0.907 vs $0.572), the spot interruption band is worse (>20% vs <5%), and latency from India is higher. Use us-east-1 only if the Mumbai quota or capacity is unavailable.
- **Why not SageMaker as the main path**: 2.2x the price for the same RAM, default quotas of 0, a container and IAM-role setup, and no interactive debugging. See section 5 for when it *is* useful.
- **Budget verdict**: the 100-200 USD of credits covers far more than the whole competition. A realistic 30-60 USD of use leaves margin, **provided nothing is left running**. An r6a.4xlarge forgotten for a week costs about $96 and would exhaust the sign-up credit.


## 4. Runbook (Windows, copy-paste)
TODO

## 5. SageMaker variant
TODO

## 6. Zero-cost alternatives (Kaggle / Colab)
TODO

## 7. Pitfalls
TODO
