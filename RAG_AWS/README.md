# RAG Ingestion Pipeline — ECS Fargate + SQS Edition

> **Architecture:** S3 → SQS → ECS Fargate Worker → Bedrock Embeddings → OpenSearch

Upgraded from Lambda. Handles large files, no 15-minute timeout, unlimited memory.

---

## What Changed from Lambda Version

| | Lambda (old) | ECS Fargate (new) |
|---|---|---|
| Trigger | S3 → Lambda directly | S3 → SQS → ECS polls queue |
| Max runtime | 15 minutes | Unlimited |
| Max file size | ~50 MB practical | Any size (streams to /tmp) |
| Memory | 10 GB max | Up to 120 GB |
| Files changed | `lambda_handler.py` removed | `sqs_worker.py` added |
| New files | — | `Dockerfile`, `sqs_setup.py`, `ecs_deploy.sh` |
| Unchanged files | — | All pipeline logic files |

---

## Architecture Flow

```
┌─────────┐  PUT    ┌──────────────────────┐  Event  ┌─────────────────┐
│   S3    │────────▶│  S3 Event Notification│────────▶│   SQS Queue     │
│ Bucket  │         │  (configured on bucket)│         │ rag-ingestion-q │
└─────────┘         └──────────────────────┘         └────────┬────────┘
                                                               │ polls
                                                     ┌─────────▼────────┐
                                                     │  ECS Fargate     │
                                                     │  sqs_worker.py   │
                                                     │  (long-running)  │
                                                     └─────────┬────────┘
                                              ┌────────────────┼────────────────┐
                                     ┌────────▼───────┐        │       ┌────────▼───────┐
                                     │Amazon Bedrock  │        │       │  OpenSearch    │
                                     │Titan Embed V2  │        │       │  Vector Index  │
                                     └────────────────┘        │       └────────────────┘
                                                      ┌────────▼───────┐
                                                      │   SQS DLQ      │
                                                      │(failed messages)│
                                                      └────────────────┘
```

---

## Prerequisites

Same as Lambda version, plus:
- Docker Desktop installed locally
- AWS CLI v2 installed

---

## Phase 1 — AWS Account + IAM (same as Lambda version)

If you completed the Lambda setup, your IAM user already has S3, Bedrock, and OpenSearch access.

Add these additional permissions to your `rag-developer` IAM user:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["sqs:*"],
      "Resource": "arn:aws:sqs:us-east-1:ACCOUNT_ID:rag-ingestion-*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:CreateRepository",
        "ecr:DescribeRepositories"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:*",
        "iam:PassRole",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## Phase 2 — Create SQS Queues

```bash
python sqs_setup.py
```

Expected output:
```
✅ DLQ created: https://sqs.us-east-1.amazonaws.com/123456789/rag-ingestion-dlq
✅ Main queue created: https://sqs.us-east-1.amazonaws.com/123456789/rag-ingestion-queue
✅ S3 → SQS publish permission set

Copy these into your .env file:
  SQS_QUEUE_URL="https://sqs.us-east-1.amazonaws.com/123456789/rag-ingestion-queue"
  SQS_DLQ_URL="https://sqs.us-east-1.amazonaws.com/123456789/rag-ingestion-dlq"
```

---

## Phase 3 — Create OpenSearch Index

Same as Lambda version — run once:

```bash
python opensearch_setup.py
```

---

## Phase 4 — Link S3 Bucket → SQS Queue

This replaces the Lambda trigger. S3 will now send PUT events to SQS instead of invoking Lambda.

```bash
# Get your SQS queue ARN
aws sqs get-queue-attributes \
  --queue-url "YOUR_SQS_QUEUE_URL" \
  --attribute-names QueueArn

# Configure S3 event notification to send to SQS
aws s3api put-bucket-notification-configuration \
  --bucket your-rag-documents-bucket \
  --notification-configuration '{
    "QueueConfigurations": [
      {
        "QueueArn": "arn:aws:sqs:us-east-1:ACCOUNT_ID:rag-ingestion-queue",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "prefix", "Value": "documents/"}
            ]
          }
        }
      }
    ]
  }'
```

**Verify:** Upload a file to S3, then check the SQS queue:
```bash
aws s3 cp test.pdf s3://your-bucket/documents/
aws sqs get-queue-attributes \
  --queue-url "YOUR_SQS_QUEUE_URL" \
  --attribute-names ApproximateNumberOfMessages
# Should show: "ApproximateNumberOfMessages": "1"
```

---

## Phase 5 — Local Testing (Before Docker)

Test the worker locally with real SQS before containerising:

```bash
# Create .env file
cat > .env << 'EOF'
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-rag-documents-bucket
SQS_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/rag-ingestion-queue
SQS_DLQ_URL=https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/rag-ingestion-dlq
OPENSEARCH_HOST=https://your-domain.us-east-1.es.amazonaws.com
OPENSEARCH_INDEX_NAME=rag-vector-index
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=your-password
USE_AWS_OPENSEARCH=false
BEDROCK_REGION=us-east-1
WORKER_CONCURRENCY=1
EOF

# Install dependencies
pip install -r requirements.txt

# Start the worker — it will poll SQS immediately
python sqs_worker.py
```

In another terminal, upload a file:
```bash
aws s3 cp sample_loan_policy.pdf s3://your-bucket/documents/
```

Watch the worker terminal — you should see ingestion logs appear within seconds.

---

## Phase 6 — Local Docker Testing

Test the container locally before pushing to ECR:

```bash
# Create .env.local (same as .env but Docker-compatible)
cp .env .env.local

# Build and run
docker-compose up --build

# Send a test message by uploading to S3
aws s3 cp sample.pdf s3://your-bucket/documents/

# Watch logs
docker-compose logs -f worker
```

---

## Phase 7 — Push to ECR (Container Registry)

```bash
# Create ECR repository (one time only)
aws ecr create-repository \
  --repository-name rag-ingestion-worker \
  --region us-east-1

# Note the repository URI from the output:
# ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ingestion-worker

# Login, build, and push
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build --platform linux/amd64 \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ingestion-worker:latest .

docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ingestion-worker:latest
```

---

## Phase 8 — Create ECS Cluster + Task Definition

### 8A. Create ECS cluster

```bash
aws ecs create-cluster --cluster-name rag-pipeline-cluster
```

### 8B. Create IAM role for ECS task

```bash
# Create task execution role (allows ECS to pull from ECR + write logs)
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Principal":{"Service":"ecs-tasks.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create task role (what the running container is allowed to do)
aws iam create-role \
  --role-name rag-ingestion-task-role \
  --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{
      "Effect":"Allow",
      "Principal":{"Service":"ecs-tasks.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }'

# Attach S3, SQS, Bedrock, OpenSearch permissions
aws iam put-role-policy \
  --role-name rag-ingestion-task-role \
  --policy-name rag-ingestion-policy \
  --policy-document '{
    "Version":"2012-10-17",
    "Statement":[
      {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::your-rag-documents-bucket/*"},
      {"Effect":"Allow","Action":["sqs:ReceiveMessage","sqs:DeleteMessage","sqs:GetQueueAttributes"],"Resource":"arn:aws:sqs:us-east-1:ACCOUNT_ID:rag-ingestion-*"},
      {"Effect":"Allow","Action":["bedrock:InvokeModel"],"Resource":"*"},
      {"Effect":"Allow","Action":["es:ESHttpGet","es:ESHttpPost","es:ESHttpPut","es:ESHttpDelete"],"Resource":"arn:aws:es:us-east-1:ACCOUNT_ID:domain/your-domain/*"},
      {"Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents"],"Resource":"*"}
    ]
  }'
```

### 8C. Register ECS task definition

Save this as `task-definition.json` (replace placeholders):

```json
{
  "family": "rag-ingestion-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/rag-ingestion-task-role",
  "ephemeralStorage": {"sizeInGiB": 50},
  "containerDefinitions": [
    {
      "name": "rag-ingestion-worker",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/rag-ingestion-worker:latest",
      "essential": true,
      "environment": [
        {"name": "AWS_REGION",             "value": "us-east-1"},
        {"name": "S3_BUCKET_NAME",         "value": "your-rag-documents-bucket"},
        {"name": "SQS_QUEUE_URL",          "value": "https://sqs.us-east-1.amazonaws.com/ACCOUNT_ID/rag-ingestion-queue"},
        {"name": "OPENSEARCH_HOST",        "value": "https://your-domain.us-east-1.es.amazonaws.com"},
        {"name": "OPENSEARCH_INDEX_NAME",  "value": "rag-vector-index"},
        {"name": "USE_AWS_OPENSEARCH",     "value": "true"},
        {"name": "BEDROCK_REGION",         "value": "us-east-1"},
        {"name": "WORKER_CONCURRENCY",     "value": "2"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group":         "/ecs/rag-ingestion-worker",
          "awslogs-region":        "us-east-1",
          "awslogs-stream-prefix": "ecs",
          "awslogs-create-group":  "true"
        }
      }
    }
  ]
}
```

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### 8D. Create ECS service

```bash
# First create a VPC subnet (use the default VPC subnet ID from your account)
VPC_SUBNET=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query "Subnets[0].SubnetId" --output text)

# Create security group for the ECS task
SG_ID=$(aws ec2 create-security-group \
  --group-name rag-worker-sg \
  --description "RAG ingestion worker" \
  --query GroupId --output text)

# Allow outbound HTTPS (for Bedrock, SQS, OpenSearch, S3 calls)
aws ec2 authorize-security-group-egress \
  --group-id $SG_ID \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Create the ECS service (1 task running at all times)
aws ecs create-service \
  --cluster rag-pipeline-cluster \
  --service-name rag-ingestion-service \
  --task-definition rag-ingestion-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[$VPC_SUBNET],
    securityGroups=[$SG_ID],
    assignPublicIp=ENABLED
  }"
```

---

## Phase 9 — Verify It's Running

```bash
# Check service status
aws ecs describe-services \
  --cluster rag-pipeline-cluster \
  --services rag-ingestion-service \
  --query "services[0].{status:status,running:runningCount,desired:desiredCount}"

# View live logs
aws logs tail /ecs/rag-ingestion-worker --follow

# Upload a file and watch it get ingested
aws s3 cp your_document.pdf s3://your-bucket/documents/
# Logs should show ingestion within 20 seconds (SQS long-poll delay)
```

---

## Phase 10 — Future Deployments

After changing any Python file:

```bash
# Set your account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

chmod +x ecs_deploy.sh
./ecs_deploy.sh
```

This script: builds → pushes to ECR → registers new task def → triggers rolling ECS deployment with zero downtime.

---

## Scaling

```bash
# Scale up to 3 workers (processes 3 documents in parallel)
aws ecs update-service \
  --cluster rag-pipeline-cluster \
  --service rag-ingestion-service \
  --desired-count 3

# Scale back down
aws ecs update-service \
  --cluster rag-pipeline-cluster \
  --service rag-ingestion-service \
  --desired-count 1
```

Or set up **Application Auto Scaling** to scale automatically based on SQS queue depth — when queue depth > 10 messages, add more tasks; when queue is empty, scale to 1.

---

## Cost Estimate (us-east-1)

| Resource | Config | Monthly cost |
|----------|--------|--------------|
| ECS Fargate | 1 task × 1 vCPU × 2 GB, 24×7 | ~$35 |
| SQS | 1M requests/month | ~$0.40 |
| ECR | 1 GB image storage | ~$0.10 |
| CloudWatch Logs | 5 GB/month | ~$2.50 |
| **Total overhead** | | **~$38/month** |

Bedrock and OpenSearch costs are the same as Lambda version.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Task keeps stopping | Container crashes on startup | Check CloudWatch logs for Python errors |
| Messages stuck in queue | Worker not polling | Verify `SQS_QUEUE_URL` env var in task definition |
| `AccessDenied` on SQS | Task role missing SQS permission | Re-attach `rag-ingestion-policy` to task role |
| `AccessDenied` on S3 | Task role missing S3 permission | Check task role policy for S3 GetObject |
| Files not appearing in SQS | S3 → SQS notification wrong | Re-run `put-bucket-notification-configuration` |
| DLQ growing | Document processing fails 3 times | Check DLQ message body for error details |
| `/tmp` full | Large files + no cleanup | Confirm `finally` block in `ingestion_pipeline.py` |
