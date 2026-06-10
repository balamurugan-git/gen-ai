#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# ecs_deploy.sh
# Builds the Docker image, pushes to ECR, and updates the ECS service.
#
# Run:
#   chmod +x ecs_deploy.sh
#   ./ecs_deploy.sh
#
# Prerequisites (set these as shell env vars or edit the defaults below):
#   AWS_ACCOUNT_ID   — your 12-digit AWS account ID
#   AWS_REGION       — e.g. us-east-1
#   ECR_REPO_NAME    — ECR repository name (created in Step 2 of README)
#   ECS_CLUSTER      — ECS cluster name
#   ECS_SERVICE      — ECS service name
#   ECS_TASK_FAMILY  — ECS task definition family name
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail   # Exit on error, undefined var, or pipe failure

# ── Configuration (edit these or export as env vars) ─────────────────────────
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-YOUR_ACCOUNT_ID}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-rag-ingestion-worker}"
ECS_CLUSTER="${ECS_CLUSTER:-rag-pipeline-cluster}"
ECS_SERVICE="${ECS_SERVICE:-rag-ingestion-service}"
ECS_TASK_FAMILY="${ECS_TASK_FAMILY:-rag-ingestion-task}"

# Derived values
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"    # Timestamp tag for traceability
IMAGE_LATEST="${ECR_URI}:latest"
IMAGE_TAGGED="${ECR_URI}:${IMAGE_TAG}"

echo ""
echo "=========================================="
echo "  RAG Pipeline — ECS Fargate Deploy"
echo "=========================================="
echo "  Account  : ${AWS_ACCOUNT_ID}"
echo "  Region   : ${AWS_REGION}"
echo "  ECR Repo : ${ECR_REPO_NAME}"
echo "  Cluster  : ${ECS_CLUSTER}"
echo "  Service  : ${ECS_SERVICE}"
echo "  Tag      : ${IMAGE_TAG}"
echo "=========================================="
echo ""

# ── Step 1: Login to ECR ──────────────────────────────────────────────────────
echo "[1/5] Logging into ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"
echo "  ✅ ECR login successful"

# ── Step 2: Build Docker image ────────────────────────────────────────────────
echo ""
echo "[2/5] Building Docker image..."
docker build \
  --platform linux/amd64 \   # Important: build for Linux x86_64 even on Mac M1/M2
  --tag "${IMAGE_LATEST}" \
  --tag "${IMAGE_TAGGED}" \
  .
echo "  ✅ Image built: ${IMAGE_TAG}"

# ── Step 3: Push to ECR ───────────────────────────────────────────────────────
echo ""
echo "[3/5] Pushing image to ECR..."
docker push "${IMAGE_LATEST}"
docker push "${IMAGE_TAGGED}"
echo "  ✅ Pushed: ${IMAGE_TAGGED}"

# ── Step 4: Register new task definition revision ────────────────────────────
echo ""
echo "[4/5] Registering new ECS task definition..."

# Fetch current task definition and update the image URI
CURRENT_TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition "${ECS_TASK_FAMILY}" \
  --query "taskDefinition" \
  --output json)

NEW_TASK_DEF=$(echo "${CURRENT_TASK_DEF}" \
  | python3 -c "
import json, sys
td = json.load(sys.stdin)
# Update image in first container
td['containerDefinitions'][0]['image'] = '${IMAGE_LATEST}'
# Remove fields that can't be in RegisterTaskDefinition
for key in ['taskDefinitionArn','revision','status','requiresAttributes',
            'placementConstraints','compatibilities','registeredAt','registeredBy']:
    td.pop(key, None)
print(json.dumps(td))
")

NEW_TASK_ARN=$(aws ecs register-task-definition \
  --cli-input-json "${NEW_TASK_DEF}" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)

echo "  ✅ New task definition: ${NEW_TASK_ARN}"

# ── Step 5: Update ECS service ────────────────────────────────────────────────
echo ""
echo "[5/5] Updating ECS service to use new task definition..."
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --task-definition "${NEW_TASK_ARN}" \
  --force-new-deployment \
  --output text \
  --query "service.serviceName"

echo ""
echo "=========================================="
echo "  ✅ Deployment complete!"
echo "  Image tag : ${IMAGE_TAG}"
echo "  Task ARN  : ${NEW_TASK_ARN}"
echo ""
echo "  Watch rollout:"
echo "  aws ecs describe-services \\"
echo "    --cluster ${ECS_CLUSTER} \\"
echo "    --services ${ECS_SERVICE} \\"
echo "    --query 'services[0].deployments'"
echo "=========================================="
