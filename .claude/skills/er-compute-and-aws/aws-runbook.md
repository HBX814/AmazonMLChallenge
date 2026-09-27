# AWS runbook: one full pipeline run on EC2 r6a.4xlarge (ap-south-1 Mumbai)

Copy-paste commands for **Windows PowerShell 5.1 / pwsh 7 on the laptop** and **bash on the Ubuntu instance**.
Prices, quotas and throughput numbers are in `reference-aws-prices-throughput.md` (same folder).

**How these commands were checked (2026-09-25, aws-cli 2.37.3 on this laptop).** No AWS credentials existed yet, so nothing
was run against AWS:
- Every API command passed the CLI's client-side parameter validation with the real flags and JSON files. The check is
  `--generate-cli-skeleton output`, which sends no request.
- Every flag of the high-level commands (`login`, `configure`, `s3 *`, `* wait *`) was matched against `aws <cmd> help`.
- Every PowerShell block parsed with the PowerShell parser, and every bash script passed `bash -n`.

First real use can still hit account-specific errors. See the troubleshooting table in section 19.

## 0. Rules before you start
| Rule | Why |
|---|---|
| **Claude Code:** before any step marked **[COSTS MONEY]**, tell the user the expected cost and wait for a go-ahead | the account is on a card once it is on the Paid plan |
| Never paste access keys or passwords into chat, the repo, skill files or scripts. Use `aws login` (temporary credentials cached in `%USERPROFILE%\.aws\login\cache`) | audit plus account safety |
| Never create an AWS Organization, a Control Tower landing zone or an IAM Identity Center *organization instance* | the account auto-upgrades and **all Free Tier credits expire immediately** |
| Never use `aws configure sso` on this account | it needs Identity Center permission sets, which need an organization instance (see the row above) |
| x86_64 only (r6a/r6i/c6a); never Graviton (`*g` types) | `sparse-dot-topn` 1.2.0 has no aarch64 wheel |
| Region **ap-south-1** for everything (instance, bucket, quotas) | cheapest r6a, and cross-Region transfer costs money |
| Structured arguments go in JSON **files** passed as `file://x.json` | PowerShell 5.1 strips the double quotes of inline JSON passed to native exes |
| `.sh` files sent to Linux must have **LF** line endings | a CRLF shebang breaks user data silently |
| Competition data goes only to your own EC2/S3 (or a private Kaggle dataset); never to Bedrock, SageMaker JumpStart/Canvas or any hosted model | README fair-play: external services used to resolve entities means disqualification |
| Terminate the instance after every run, then run the audit (section 14) | a forgotten r6a.4xlarge costs ~$96/week |

In Claude Code every PowerShell tool call is a **fresh shell**. Start each step's call with the variables block from section 3.
The block rediscovers the security group, instance and bucket by name, so no state is lost between calls.

## 1. One-time account setup (browser, as the root user)
1. **Protect root.** Enable MFA on the root user.
2. **Upgrade to the Paid plan.** In the console, open Billing and Cost Management -> Free Tier -> "Upgrade plan".
   - The Free plan allows only instances of at most 2 vCPU / 8 GiB.
   - The remaining credits carry over and still expire 12 months after sign-up.
   - Do not upgrade by joining Organizations or Control Tower.
   - CLI alternative, as root: `aws freetier upgrade-account-plan --account-plan-type PAID --region us-east-1`.
3. **Optional:** Account -> "IAM user and role access to Billing information" -> Activate. The IAM user then sees the
   Billing pages in the console. The CLI cost calls work without this setting.
4. **Enable Cost Explorer:** open Billing -> Cost Explorer once. Data appears within 24 h.
5. **Create the working identity** (least privilege, no access keys):
   1. IAM -> Policies -> Create policy -> JSON: paste the policy below. Name it `MasterBoltMlc`.
   2. IAM -> Users -> Create user `mlc-cli`. Tick "Provide user access to the AWS Management Console" and pick a password.
   3. Attach `MasterBoltMlc` and the AWS managed policy **`SignInLocalDevelopmentAccess`**. That policy is what `aws login` needs.
   4. Enable MFA for `mlc-cli`.
   5. Sign out of root. Sign in as `mlc-cli` from now on.
6. **Credit activities.** Up to +$100 of credits: +$20 each for creating a budget, launching an EC2 instance, a Lambda
   web app, an RDS database and the Bedrock playground.
   - For Bedrock, type only a generic prompt such as "hello". **Never competition data.**
   - Check progress: `aws freetier list-account-activities --region us-east-1 --query "activities[].[title,status]" --output table`.
   - If a budget created with the CLI does not mark its activity COMPLETED, use the console "Explore AWS" widget.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "Ec2Read", "Effect": "Allow", "Action": ["ec2:Describe*", "ec2:GetConsoleOutput"], "Resource": "*"},
    {"Sid": "Ec2ManageOnlyInMumbai", "Effect": "Allow",
     "Action": ["ec2:RunInstances", "ec2:StartInstances", "ec2:StopInstances", "ec2:TerminateInstances",
                "ec2:RebootInstances", "ec2:CreateTags", "ec2:ImportKeyPair", "ec2:DeleteKeyPair",
                "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup", "ec2:AuthorizeSecurityGroupIngress",
                "ec2:RevokeSecurityGroupIngress", "ec2:DeleteVolume", "ec2:DeleteSnapshot",
                "ec2:CancelSpotInstanceRequests", "ec2:ReleaseAddress", "ec2:ModifyInstanceAttribute"],
     "Resource": "*", "Condition": {"StringEquals": {"aws:RequestedRegion": "ap-south-1"}}},
    {"Sid": "DenyUnapprovedInstanceTypes", "Effect": "Deny", "Action": ["ec2:RunInstances", "ec2:StartInstances"],
     "Resource": "arn:aws:ec2:*:*:instance/*",
     "Condition": {"StringNotEquals": {"ec2:InstanceType": ["t3.micro", "t3.small", "m7i-flex.large", "r6a.2xlarge",
                                                          "r6a.4xlarge", "r6i.4xlarge", "c6a.8xlarge", "g4dn.xlarge"]}}},
    {"Sid": "ServiceLinkedRolesForSpotAndAlarms", "Effect": "Allow", "Action": "iam:CreateServiceLinkedRole",
     "Resource": "*", "Condition": {"StringEquals": {"iam:AWSServiceName": ["spot.amazonaws.com", "events.amazonaws.com"]}}},
    {"Sid": "OnlyTheMlcInstanceRole", "Effect": "Allow",
     "Action": ["iam:CreateRole", "iam:GetRole", "iam:DeleteRole", "iam:PutRolePolicy", "iam:GetRolePolicy",
                "iam:DeleteRolePolicy", "iam:CreateInstanceProfile", "iam:GetInstanceProfile",
                "iam:DeleteInstanceProfile", "iam:AddRoleToInstanceProfile", "iam:RemoveRoleFromInstanceProfile",
                "iam:PassRole"],
     "Resource": ["arn:aws:iam::*:role/mlc-ec2-s3", "arn:aws:iam::*:instance-profile/mlc-ec2-s3"]},
    {"Sid": "PublicAmiParameters", "Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters"],
     "Resource": "arn:aws:ssm:*::parameter/aws/service/*"},
    {"Sid": "ProjectBucket", "Effect": "Allow",
     "Action": ["s3:CreateBucket", "s3:DeleteBucket", "s3:ListBucket", "s3:GetBucketLocation",
                "s3:PutBucketPublicAccessBlock", "s3:GetBucketPublicAccessBlock"],
     "Resource": "arn:aws:s3:::master-bolt-mlc-*"},
    {"Sid": "ProjectBucketObjects", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
     "Resource": "arn:aws:s3:::master-bolt-mlc-*/*"},
    {"Sid": "MonitoringQuotasCost", "Effect": "Allow",
     "Action": ["s3:ListAllMyBuckets", "cloudwatch:PutMetricAlarm", "cloudwatch:DeleteAlarms", "cloudwatch:DescribeAlarms",
                "cloudwatch:GetMetricData", "cloudwatch:GetMetricStatistics", "budgets:ViewBudget", "budgets:ModifyBudget",
                "ce:GetCostAndUsage", "freetier:GetAccountPlanState", "freetier:ListAccountActivities",
                "freetier:GetFreeTierUsage", "servicequotas:GetServiceQuota", "servicequotas:ListServiceQuotas",
                "servicequotas:RequestServiceQuotaIncrease", "servicequotas:ListRequestedServiceQuotaChangeHistory",
                "servicequotas:GetRequestedServiceQuotaChange"],
     "Resource": "*"}
  ]
}
```
The instance-type Deny also blocks Graviton by construction.

Caveat: whoever may create `mlc-ec2-s3` could attach any policy to that role. This is acceptable for a one-person account. If you
want a tighter setup, create the role once as root in the console (section 7) and drop the `iam:Create*/Put*` actions.

## 2. CLI sign-in on the laptop (once, then every 12 h)
```powershell
aws --version                                   # needs >= 2.32.0 for `aws login` (installed: 2.37.3)
aws configure set region ap-south-1             # BEFORE login, or login offers [us-east-1] as the default
aws configure set output json
aws login                                       # browser opens -> sign in as mlc-cli -> return to the terminal
aws sts get-caller-identity                     # Arn must end in :user/mlc-cli (not :root)
aws freetier get-account-plan-state --region us-east-1   # accountPlanType must be PAID; shows remaining credits
```
- The session refreshes itself for up to 12 h. After that, `ExpiredToken` means: run `aws login` again. `aws logout` ends it early.
- Use the **default** profile. It works in every new shell without `$env:AWS_PROFILE`, which does not persist between Claude Code tool calls.
- To sign in on a machine with no browser (e.g. over SSH), use `aws login --remote`. It prints a URL; paste the code back.
- **Last resort only**, if `aws login` is impossible: create an access key for `mlc-cli` (never for root) and run `aws configure`.
  The key lands in `%USERPROFILE%\.aws\credentials`, outside the repo. Delete it when done:
  `aws iam delete-access-key --user-name mlc-cli --access-key-id <id>`. The policy above does not allow this call, so
  delete the key in the console.

## 3. Variables block (paste at the start of every PowerShell session or tool call)
```powershell
$REGION  = "ap-south-1"
$PROJ    = "C:\Users\harsh\Downloads\AmazonMLChallenge"
$W       = "$HOME\mlc-aws"                  # JSON + scripts, deliberately OUTSIDE the project tree
$KEY     = "$HOME\.ssh\mlc-key"             # private key, generated locally, never uploaded
$ITYPE   = "r6a.4xlarge"                    # 16 vCPU / 128 GiB; fallback r6a.2xlarge (8 / 64)
$EMAIL   = "you@example.com"                # EDIT: address that receives budget alerts
New-Item -ItemType Directory -Force $W, "$HOME\.ssh" | Out-Null
Set-Location $W
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$BUCKET  = "master-bolt-mlc-$ACCOUNT"
$SG      = aws ec2 describe-security-groups --filters "Name=group-name,Values=mlc-ssh" --query "SecurityGroups[].GroupId" --output text
$IID     = aws ec2 describe-instances --filters "Name=tag:Name,Values=mlc-run" "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[].Instances[].InstanceId" --output text
$IP      = if ($IID) { aws ec2 describe-instances --instance-ids $IID --query "Reservations[0].Instances[0].PublicIpAddress" --output text } else { "" }
$SSH     = @("-i", $KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=60")
"account=$ACCOUNT bucket=$BUCKET sg=$SG instance=$IID ip=$IP"
```

## 4. Budget with e-mail alerts (once, free)
The budget must **exclude credits** (`IncludeCredit: false`). Otherwise credits make the net cost $0 and the alert never fires.
```powershell
@'
{
  "BudgetName": "mlc-gross-usage-25usd",
  "BudgetLimit": {"Amount": "25", "Unit": "USD"},
  "BudgetType": "COST",
  "TimeUnit": "MONTHLY",
  "CostTypes": {"IncludeCredit": false, "IncludeRefund": false, "IncludeTax": true, "IncludeSupport": true,
                "IncludeDiscount": true, "IncludeOtherSubscription": true, "IncludeRecurring": true,
                "IncludeSubscription": true, "IncludeUpfront": true, "UseBlended": false}
}
'@ | Set-Content -Encoding ascii budget.json
@"
[
  {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 20, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]},
  {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN", "Threshold": 80, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]},
  {"Notification": {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN", "Threshold": 100, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]}
]
"@ | Set-Content -Encoding ascii budget-notifications.json
aws budgets create-budget --account-id $ACCOUNT --budget file://budget.json --notifications-with-subscribers file://budget-notifications.json --region us-east-1
aws budgets describe-budgets --account-id $ACCOUNT --query "Budgets[].[BudgetName,BudgetLimit.Amount]" --output table --region us-east-1
```
The alerts fire at $5 and $20 actual spend, and at a forecast above $25 for the month. Budgets data lags actual usage by
up to about 8-24 h, so the budget backs up the section 14 audit; it does not replace it.

## 5. Quotas, capacity and spot price (day 1; approval can take hours)
The default is **5 vCPU** for on-demand standard instances (`L-1216C47A`) and 5 for Spot (`L-34B43A08`). r6a.4xlarge needs 16.
Ask for 32 so that c6a.8xlarge also fits.
```powershell
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query "Quota.Value"
aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08 --query "Quota.Value"
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-1216C47A --desired-value 32
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-34B43A08 --desired-value 32
# GPU only if really needed (g4dn.xlarge = 4 vCPU; default 0; often slow to approve on new accounts):
#   --quota-code L-DB2E81BA (on-demand G/VT) / L-3819A6DF (spot G/VT)  --desired-value 8
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --query "RequestedQuotas[].[QuotaName,DesiredValue,Status]" --output table
aws ec2 describe-instance-type-offerings --location-type availability-zone --filters "Name=instance-type,Values=r6a.4xlarge,r6a.2xlarge" --query "InstanceTypeOfferings[].[InstanceType,Location]" --output table
aws ec2 describe-spot-price-history --instance-types r6a.4xlarge --product-descriptions "Linux/UNIX" --start-time (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") --query "SpotPriceHistory[].[AvailabilityZone,SpotPrice]" --output table
```
Quotas are per Region. If the Mumbai request is stuck, the fallback is us-east-1: file the same request there, but the
instance costs $0.907/h instead of $0.572 and spot interruptions run above 20%.

## 6. SSH key and security group (once)
```powershell
ssh-keygen -t ed25519 -f $KEY -C "mlc" -N '""'           # empty passphrase; drop -N to set one interactively
aws ec2 import-key-pair --key-name mlc-key --public-key-material "fileb://$KEY.pub"
$MYIP = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
$VPC  = aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query "Vpcs[0].VpcId" --output text
$SG   = aws ec2 create-security-group --group-name mlc-ssh --description "SSH from my IP only" --vpc-id $VPC --query GroupId --output text
aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr "$MYIP/32"
```
When your home IP changes and SSH times out:
- Allow the new IP: `aws ec2 authorize-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr "<new-ip>/32"`.
- Remove the old one: `aws ec2 revoke-security-group-ingress --group-id $SG --protocol tcp --port 22 --cidr "<old-ip>/32"`.

Never open port 22 to `0.0.0.0/0`.

## 7. S3 bucket and instance role (once)
```powershell
aws s3 mb "s3://$BUCKET" --region $REGION
aws s3api put-public-access-block --bucket $BUCKET --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
@'
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
'@ | Set-Content -Encoding ascii trust.json
@"
{"Version": "2012-10-17", "Statement": [
  {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": "arn:aws:s3:::$BUCKET"},
  {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"], "Resource": "arn:aws:s3:::$BUCKET/*"}
]}
"@ | Set-Content -Encoding ascii role-s3.json
aws iam create-role --role-name mlc-ec2-s3 --assume-role-policy-document file://trust.json
aws iam put-role-policy --role-name mlc-ec2-s3 --policy-name mlc-s3-bucket --policy-document file://role-s3.json
aws iam create-instance-profile --instance-profile-name mlc-ec2-s3
aws iam add-role-to-instance-profile --instance-profile-name mlc-ec2-s3 --role-name mlc-ec2-s3
aws iam wait instance-profile-exists --instance-profile-name mlc-ec2-s3
```
The instance gets S3 access through this role. No keys ever sit on the box. EC2 may take a further 10-20 s to see a
new instance profile.

## 8. Upload data and code (dataset once, code before every run)
```powershell
aws s3 sync "$PROJ\student_resource\dataset" "s3://$BUCKET/dataset" --only-show-errors      # ~2.4 GB, read-only source
aws s3 sync "$PROJ\code" "s3://$BUCKET/code" --delete --exclude "*__pycache__/*" --exclude "*.pyc" --only-show-errors
aws s3 ls "s3://$BUCKET/" --recursive --summarize --human-readable | Select-Object -Last 3
```
- Only the dataset and `code/` go up. Never upload `.venv`, `work/` caches or anything from the scratchpad.
- The data has to cross the laptop's own uplink: 2.4 GB at 10-50 Mbit/s takes about 7-30 min. Transfer into AWS is free.
- `aws s3 sync` is resumable: if the upload breaks, run the same command again.

## 9. Launch the instance (on-demand) [COSTS MONEY: ~$0.57/h + $0.30/day disk + $0.005/h IPv4]
**9a. User data** (runs once as root at first boot). It installs git/zip/tmux, AWS CLI v2 and a Python 3.10 venv through uv
(Ubuntu 24.04 ships 3.12). It also writes the three helper scripts.
```powershell
$ud = @'
#!/bin/bash
# Master Bolt EC2 bootstrap: runs ONCE as root. Log: /var/log/cloud-init-output.log  Done-marker: /home/ubuntu/BOOTSTRAP_DONE
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git zip unzip tmux htop pigz time curl build-essential python3-venv python3-pip
if ! command -v aws >/dev/null; then
  curl -sSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install && rm -rf /tmp/aws /tmp/awscliv2.zip
fi
U=/home/ubuntu
mkdir -p $U/mlc/student_resource/dataset $U/mlc/code $U/mlc/work/logs $U/mlc/output
sudo -u ubuntu -H bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u ubuntu -H bash -c '$HOME/.local/bin/uv python install 3.10 && $HOME/.local/bin/uv venv --seed --python 3.10 $HOME/venv' \
  || sudo -u ubuntu -H python3 -m venv $U/venv

cat > $U/mlc/sync_and_setup.sh <<'EOF'
#!/usr/bin/env bash
# Pull dataset, code and checkpoints from S3; install pinned requirements.   usage: bash ~/mlc/sync_and_setup.sh
set -euo pipefail
M="$HOME/mlc"; BUCKET="master-bolt-mlc-$(aws sts get-caller-identity --query Account --output text)"
aws s3 sync "s3://$BUCKET/dataset" "$M/student_resource/dataset" --only-show-errors
aws s3 sync "s3://$BUCKET/code" "$M/code" --delete --only-show-errors
aws s3 sync "s3://$BUCKET/work" "$M/work" --only-show-errors   # resume: checkpoints + _done_* markers (empty on a fresh run)
source "$HOME/venv/bin/activate"
REQ="$M/code/business_entity_resolution/requirements.txt"
if [ -x "$HOME/.local/bin/uv" ]; then "$HOME/.local/bin/uv" pip install -r "$REQ"; else python -m pip install -r "$REQ"; fi
python -c "import platform, sys; print('python', sys.version.split()[0], platform.machine())"
for m in polars pyarrow numpy rapidfuzz lightgbm sklearn psutil duckdb sparse_dot_topn anyascii; do
  python -c "import importlib; x = importlib.import_module('$m'); print('  $m', getattr(x, '__version__', '?'))" 2>/dev/null || echo "  $m not installed"
done
wc -l "$M"/student_resource/dataset/*/*.tsv
echo "vCPU $(nproc)"; free -g | head -2; df -h / | tail -1
EOF

cat > $U/mlc/run_stages.sh <<'EOF'
#!/usr/bin/env bash
# Run the pipeline stage by stage; after each stage push work/ + output/ to S3 and drop work/_done_<stage>,
# so a re-run (new Spot box after sync_and_setup.sh) skips finished stages.   usage: bash ~/mlc/run_stages.sh [max_hours]
set -uo pipefail
MAXH="${1:-6}"
M="$HOME/mlc"; CODE="$M/code/business_entity_resolution"
BUCKET="master-bolt-mlc-$(aws sts get-caller-identity --query Account --output text)"
mkdir -p "$M/work/logs" "$M/output"; LOG="$M/work/logs/run_$(date +%Y%m%d_%H%M%S).log"
sudo shutdown -h "+$((MAXH * 60))" "mlc safety power-off after ${MAXH}h" || true    # cancel: sudo shutdown -c
source "$HOME/venv/bin/activate"
export BER_MEM_BUDGET_GB="${BER_MEM_BUDGET_GB:-110}" BER_MIN_FREE_GB="${BER_MIN_FREE_GB:-4}" BER_MEM_LOG="$M/work/logs/memory.log"
push() { aws s3 sync "$M/work" "s3://$BUCKET/work" --only-show-errors; aws s3 sync "$M/output" "s3://$BUCKET/output" --only-show-errors; }
cd "$CODE"
for st in prepare block features train predict write; do
  if [ -f "$M/work/_done_$st" ]; then echo "=== skip $st (done)" | tee -a "$LOG"; continue; fi
  echo "=== stage $st start $(date -Is)" | tee -a "$LOG"
  /usr/bin/time -v python -u src/run_pipeline.py --data-dir ../../student_resource/dataset \
      --work-dir ../../work --out-dir ../../output --stage "$st" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "=== stage $st FAILED rc=$rc $(date -Is)" | tee -a "$LOG"; push; exit "$rc"; fi
  touch "$M/work/_done_$st"; push
done
echo "PIPELINE_DONE $(date -Is)" | tee -a "$LOG"; push
sudo shutdown -h +20 "pipeline done - powering off in 20 min (cancel: sudo shutdown -c)"
EOF

cat > $U/mlc/spot_watch.sh <<'EOF'
#!/usr/bin/env bash
# Poll the Spot interruption notice (IMDSv2) every 5 s; on notice push checkpoints + logs to S3 (2-minute warning).
BUCKET="master-bolt-mlc-$(aws sts get-caller-identity --query Account --output text)"
while sleep 5; do
  T=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
  if curl -sf -H "X-aws-ec2-metadata-token: $T" "http://169.254.169.254/latest/meta-data/spot/instance-action" >/dev/null; then
    echo "SPOT INTERRUPTION NOTICE $(date -Is)" | tee -a "$HOME/mlc/work/logs/spot.log"
    aws s3 sync "$HOME/mlc/work" "s3://$BUCKET/work" --only-show-errors
    break
  fi
done
EOF
chmod +x $U/mlc/*.sh
chown -R ubuntu:ubuntu $U/mlc
touch $U/BOOTSTRAP_DONE
'@
[IO.File]::WriteAllText("$W\user-data.sh", $ud.Replace("`r`n", "`n"))     # LF + UTF-8 without BOM
```

**9b. Root volume, AMI and launch.**
```powershell
@'
[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": true, "Encrypted": true}}]
'@ | Set-Content -Encoding ascii bdm.json
$AMI = aws ssm get-parameters --names "/aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp3/ami-id" --query "Parameters[0].Value" --output text
aws ec2 describe-images --image-ids $AMI --query "Images[0].[Name,Architecture]" --output text    # must say ...noble...amd64 / x86_64
$IID = aws ec2 run-instances --image-id $AMI --instance-type $ITYPE --key-name mlc-key --security-group-ids $SG `
  --iam-instance-profile Name=mlc-ec2-s3 --block-device-mappings file://bdm.json --user-data file://user-data.sh `
  --instance-initiated-shutdown-behavior terminate --metadata-options "HttpTokens=required,HttpEndpoint=enabled" `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mlc-run}]" "ResourceType=volume,Tags=[{Key=Name,Value=mlc-run}]" `
  --query "Instances[0].InstanceId" --output text
aws ec2 wait instance-running --instance-ids $IID
$IP = aws ec2 describe-instances --instance-ids $IID --query "Reservations[0].Instances[0].PublicIpAddress" --output text
"instance $IID at $IP"
```
- `--instance-initiated-shutdown-behavior terminate`: the in-box safety timers (6 h cap, 20 min after `PIPELINE_DONE`)
  **delete** the box. `run_stages.sh` has already pushed `work/` and `output/` to S3 after every stage.
- To keep a box between runs, stop it yourself with `aws ec2 stop-instances --instance-ids $IID`. Its 100 GB disk stays
  billed at about $0.30/day.

**9c. Idle safety net.** The alarm stops the instance after 60 min below 3% CPU, which catches a forgotten
or crashed box. It is free.
```powershell
aws cloudwatch put-metric-alarm --alarm-name "mlc-idle-$IID" --namespace AWS/EC2 --metric-name CPUUtilization `
  --dimensions "Name=InstanceId,Value=$IID" --statistic Average --period 300 --evaluation-periods 12 `
  --threshold 3 --comparison-operator LessThanThreshold --treat-missing-data notBreaching `
  --alarm-actions "arn:aws:automate:${REGION}:ec2:stop"
```

## 10. Spot variant (about 60% cheaper: r6a.4xlarge Mumbai ~$0.21/h, interruption band <5%) [COSTS MONEY]
- Use the same steps as section 9, with the market options below.
- A one-time Spot instance cannot be *stopped*, so interruption and shutdown both **terminate** it.
- The idle alarm must therefore use `...:ec2:terminate`.
```powershell
@'
{"MarketType": "spot", "SpotOptions": {"SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate"}}
'@ | Set-Content -Encoding ascii spot.json
$IID = aws ec2 run-instances --image-id $AMI --instance-type $ITYPE --key-name mlc-key --security-group-ids $SG `
  --iam-instance-profile Name=mlc-ec2-s3 --block-device-mappings file://bdm.json --user-data file://user-data.sh `
  --instance-market-options file://spot.json --instance-initiated-shutdown-behavior terminate `
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" `
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=mlc-run}]" "ResourceType=volume,Tags=[{Key=Name,Value=mlc-run}]" `
  --query "Instances[0].InstanceId" --output text
aws ec2 wait instance-running --instance-ids $IID
aws cloudwatch put-metric-alarm --alarm-name "mlc-idle-$IID" --namespace AWS/EC2 --metric-name CPUUtilization `
  --dimensions "Name=InstanceId,Value=$IID" --statistic Average --period 300 --evaluation-periods 12 `
  --threshold 3 --comparison-operator LessThanThreshold --treat-missing-data notBreaching `
  --alarm-actions "arn:aws:automate:${REGION}:ec2:terminate"
```
**Checkpoint and resume.**
- `run_stages.sh` pushes `work/` (the parquet checkpoints) and `output/` to S3 after each stage and writes `work/_done_<stage>`.
- `spot_watch.sh` pushes again on the 2-minute interruption notice.
- After an interruption:
  1. Launch a new Spot box. The same user data works.
  2. Run section 11. `sync_and_setup.sh` pulls `s3://$BUCKET/work` back, markers included.
  3. Run `run_stages.sh` again. It skips the finished stages.
- Resume granularity is **one stage**. A stage that dies halfway restarts from its beginning. So keep every stage under
  ~1 h, or make long stages write per-shard part files (country x state blocks, feature chunks) and skip parts that already exist.
- **A new code version needs a fresh run.** Delete the markers first, or finished stages are skipped with stale outputs:
  `aws s3 rm "s3://$BUCKET/work/" --recursive --exclude "*" --include "_done_*"`. To wipe the checkpoints too, drop
  the `--exclude`/`--include`.

## 11. On the instance: bootstrap check, sync, environment
From the laptop, non-interactively. This form works for Claude Code:
```powershell
ssh @SSH "ubuntu@$IP" "cloud-init status --wait; ls ~/BOOTSTRAP_DONE && tail -n 3 /var/log/cloud-init-output.log"
ssh @SSH "ubuntu@$IP" "bash ~/mlc/sync_and_setup.sh"
```
`sync_and_setup.sh` prints the Python version and `x86_64`, the library versions and `wc -l` of the seven TSVs. Compare
those counts with the laptop (`reference-data-facts.md`: e.g. train S1 2,206,821 rows + 1 header line).

**Cheap rehearsal (recommended).** Before the 2-3 h run, run one short stage, or the full pipeline on a small dev
slice, to catch environment errors for cents. Use the dev slice from `make_dev_slice.py` (skill er-data-loading),
uploaded under `s3://$BUCKET/dataset_dev/`.

Interactive alternative: run `ssh @SSH "ubuntu@$IP"`, then `source ~/venv/bin/activate && cd ~/mlc/code/business_entity_resolution`.

## 12. Start the full run in tmux and monitor it
```powershell
ssh @SSH "ubuntu@$IP" "tmux new-session -d -s mlc 'bash ~/mlc/run_stages.sh 6'"          # 6 h hard cap
ssh @SSH "ubuntu@$IP" "tmux new-window -d -t mlc 'bash ~/mlc/spot_watch.sh'"              # Spot only
ssh @SSH "ubuntu@$IP" "tail -n 25 ~/mlc/work/logs/run_*.log; tail -n 3 ~/mlc/work/logs/memory.log; free -g | head -2; uptime"
```
- To watch live: `ssh -t @SSH "ubuntu@$IP" "tmux attach -t mlc"`. Detach with Ctrl-b then d.
- To keep the box past the cap, cancel the timer: `sudo shutdown -c`.
- The run survives laptop sleep and SSH drops, because it lives inside tmux.

## 13. Download results
```powershell
New-Item -ItemType Directory -Force "$PROJ\output_aws", "$PROJ\work\logs_aws" | Out-Null
aws s3 sync "s3://$BUCKET/output" "$PROJ\output_aws" --only-show-errors
aws s3 sync "s3://$BUCKET/work/logs" "$PROJ\work\logs_aws" --only-show-errors
aws s3 sync "s3://$BUCKET/work/models" "$PROJ\work\models_aws" --only-show-errors         # if the model is needed locally
```
- Validate `output_aws\*.tsv` on the laptop before copying them to `output\` (**See also:** er-submission-packaging).
- Log the run in `work/experiments.md`. Record the stage timings (`=== stage` lines plus `/usr/bin/time` output in the
  run log) and the peak memory (`memory.log`).

## 14. Teardown after EVERY run, then audit
```powershell
aws ec2 terminate-instances --instance-ids $IID
aws ec2 wait instance-terminated --instance-ids $IID
aws cloudwatch delete-alarms --alarm-names "mlc-idle-$IID"
# audit: every command below must print nothing (or only 'terminated' rows)
aws ec2 describe-instances --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "Reservations[].Instances[].[InstanceId,InstanceType,State.Name]" --output table
aws ec2 describe-volumes --query "Volumes[].[VolumeId,Size,State]" --output table          # leftovers: aws ec2 delete-volume --volume-id vol-...
aws ec2 describe-snapshots --owner-ids self --query "Snapshots[].[SnapshotId,VolumeSize]" --output table   # aws ec2 delete-snapshot --snapshot-id snap-...
aws ec2 describe-addresses --query "Addresses[].[PublicIp,AllocationId]" --output table    # aws ec2 release-address --allocation-id eipalloc-...
aws ec2 describe-spot-instance-requests --filters "Name=state,Values=open,active" --query "SpotInstanceRequests[].SpotInstanceRequestId" --output text
aws cloudwatch describe-alarms --alarm-name-prefix mlc- --query "MetricAlarms[].AlarmName" --output text
# every Region (catches a launch into the wrong Region):
foreach ($r in (aws ec2 describe-regions --query "Regions[].RegionName" --output text).Split()) {
  $n = aws ec2 describe-instances --region $r --filters "Name=instance-state-name,Values=pending,running,stopping,stopped" --query "length(Reservations[].Instances[])" --output text
  if ($n -ne "0") { "$r : $n instance(s) still exist" }
}
```
What each teardown choice keeps billing:

| Action | Compute billing | EBS disk billing | Public IPv4 billing |
|---|---|---|---|
| stop | ends | continues (~$0.30/day for 100 GB) | ends |
| terminate | ends | ends: the root volume is deleted because `DeleteOnTermination: true` | ends |
| Elastic IP (we allocate none) | n/a | n/a | billed while allocated, even with no instance |

The S3 bucket costs about $0.025 per GB-month. Keep it during the competition.

## 15. Check spend
```powershell
aws freetier get-account-plan-state --region us-east-1 --query "[accountPlanType,accountPlanRemainingCredits.amount,accountPlanExpirationDate]" --output text   # free call
@'
{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}
'@ | Set-Content -Encoding ascii ce-usage.json
$START = (Get-Date -Day 1).ToString("yyyy-MM-dd"); $END = (Get-Date).AddDays(1).ToString("yyyy-MM-dd")
aws ce get-cost-and-usage --time-period "Start=$START,End=$END" --granularity MONTHLY --metrics UnblendedCost --group-by "Type=DIMENSION,Key=SERVICE" --filter file://ce-usage.json --region us-east-1 --output table
aws ce get-cost-and-usage --time-period "Start=$START,End=$END" --granularity DAILY --metrics UnblendedCost --region us-east-1 --query "ResultsByTime[].[TimePeriod.Start,Total.UnblendedCost.Amount]" --output text
```
- The first `ce` call shows gross usage per service, before credits. The second shows the daily **net** cost, after credits.
- Each Cost Explorer request costs **$0.01**, so do not poll it in a loop.
- Billing data lags by up to 24 h.
- Expected spend: one full run on-demand is about $2.7, on Spot about $1.3. Ten runs plus 20 h of debugging come to
  about $38 (reference section 3.4).

## 16. End-of-competition cleanup
```powershell
aws s3 rb "s3://$BUCKET" --force
aws iam remove-role-from-instance-profile --instance-profile-name mlc-ec2-s3 --role-name mlc-ec2-s3
aws iam delete-instance-profile --instance-profile-name mlc-ec2-s3
aws iam delete-role-policy --role-name mlc-ec2-s3 --policy-name mlc-s3-bucket
aws iam delete-role --role-name mlc-ec2-s3
aws ec2 delete-security-group --group-id $SG
aws ec2 delete-key-pair --key-name mlc-key
aws budgets delete-budget --account-id $ACCOUNT --budget-name mlc-gross-usage-25usd --region us-east-1   # optional; keeping it is free
aws logout
```
Then run the section 14 audit one last time. In the console, delete or deactivate `mlc-cli` and any access key it ever had.

## 17. Zero-cost alternative: Kaggle notebook (when AWS is blocked by quota or payment)
The limits below are Kaggle's published figures, not measured by us, and they change. Check them in the notebook with
`!nproc; !free -g; !df -h`.
- A CPU session has **4 cores and about 30 GB RAM**, with a **12 h** limit per session and about **20 GB** of persistent `/kaggle/working`.
- GPU sessions (P100 / 2x T4) have about 29-32 GB RAM and a weekly GPU quota of about 30 h.
- 30 GB is below the 64 GB floor for a single-process full run (reference section 3.2), so the notebook still needs the
  laptop's discipline: shard by country, and checkpoint parquet to `/kaggle/working`.
- Split the stages across sessions. Chain them by adding a finished notebook's output as the input of the next one.
- 4 cores make blocking and features about 2-4x slower than r6a.4xlarge.
- Upload the TSVs as a **private** Kaggle dataset. A public dataset would redistribute competition data.
- Upload `code/` as a second private dataset, or with `%%writefile`. `pip install` needs Internet switched on (phone-verified account).
- Use the notebook as plain compute. Do not use Kaggle-hosted models or AI assistants on the data (fair-play rule).
- Colab's free tier has about 12 GB RAM, which is no better than the laptop.

## 18. SageMaker (not the main path)
- SageMaker costs 2.2x as much for the same RAM. ml.r5.4xlarge Processing is $1.248/h against $0.572 for EC2 r6a.4xlarge in Mumbai.
- **Default quotas are 0** for ml.r5/r7i.4xlarge training, processing and notebook instances. They need their own
  Service Quotas request.
- ml.r6i.* has no Training or Processing SKU.
- The 2-month free trial covers only ml.t3.medium and ml.m5.xlarge (<= 16 GiB), which is no bigger than the laptop.
- Use it only if the team already knows SageMaker Processing. A Studio notebook running our own code is fine for
  compliance. JumpStart/Canvas/Bedrock models applied to the data are not.
- GPUs (g4dn.xlarge on EC2, needs the G/VT quota) only pay off for transformer embeddings, and even then the model must be
  MIT/Apache and <= 8B (**See also:** er-compliance-licensing).

## 19. Troubleshooting
| Symptom | Fix |
|---|---|
| `VcpuLimitExceeded` / "You have requested more vCPU capacity than your current vCPU limit" | Quota still 5: section 5, wait for APPROVED; or use r6a.2xlarge (8 vCPU) only if quota >= 8 |
| Launch refused as not free-tier eligible / only small types allowed | Account still on the Free plan: section 1 step 2 |
| `InsufficientInstanceCapacity` | Retry in another AZ: add `--subnet-id` from `aws ec2 describe-subnets --filters "Name=default-for-az,Values=true" --query "Subnets[].[SubnetId,AvailabilityZone]" --output table`; or r6i.4xlarge ($1.04/h) |
| `Invalid IAM Instance Profile name` right after section 7 | Propagation delay: wait 20 s and rerun |
| `UnauthorizedOperation` / `AccessDenied` | Signed in as the wrong identity (`aws sts get-caller-identity`) or the action is missing from `MasterBoltMlc`; decode with the console's IAM policy simulator |
| `ExpiredToken` / "Your session has expired" | `aws login` again (12 h max) |
| `Error parsing parameter ... Expected: '=', received: '"'` | Inline JSON in PowerShell: put it in a file and pass `file://x.json` |
| JSON file rejected with odd characters | It was written UTF-16 (`Out-File`/`>` in PS 5.1): use `Set-Content -Encoding ascii` |
| SSH `Connection timed out` | Home IP changed: section 6; or the instance was stopped/terminated by a timer or alarm |
| SSH `UNPROTECTED PRIVATE KEY FILE` | `icacls $KEY /inheritance:r /grant:r "$($env:USERNAME):(R)"` |
| `cloud-init status` = error | `ssh ... "sudo tail -n 80 /var/log/cloud-init-output.log"`; most often CRLF in user-data (regenerate with section 9a) |
| `pip`/`uv` builds `sparse-dot-topn` from source | Wrong architecture (arm64) or Python without a wheel: check `platform.machine()` == x86_64 and Python 3.10 |
| A stage is killed with exit code 137 | Linux OOM killer: lower chunk sizes or `BER_MEM_BUDGET_GB`; r6a.4xlarge has 128 GiB, so something is materialising Python lists/strings |
