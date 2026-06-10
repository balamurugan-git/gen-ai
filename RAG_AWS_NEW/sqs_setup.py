"""
sqs_setup.py
------------
One-time script — creates the SQS queue and dead-letter queue (DLQ).
Run once before deploying ECS.

What it creates:
  1. DLQ  (rag-ingestion-dlq)   — receives messages that failed 3 times
  2. Main queue (rag-ingestion-queue) — receives S3 event notifications

Run:
    python sqs_setup.py

Output:
    Prints queue URLs to copy into your .env file.
"""

import json
import logging
import sys

import boto3
from botocore.exceptions import ClientError

from config import AWS_REGION, SQS_VISIBILITY_TIMEOUT, SQS_MAX_RECEIVE_COUNT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAIN_QUEUE_NAME = "rag-ingestion-queue"
DLQ_NAME        = "rag-ingestion-dlq"


def create_queues():
    sqs = boto3.client("sqs", region_name=AWS_REGION)

    print("\n" + "=" * 60)
    print("  RAG Pipeline — SQS Queue Setup")
    print("=" * 60)

    # ── Step 1: Create DLQ first ──────────────────────────────────────────────
    print(f"\n[1/4] Creating dead-letter queue: {DLQ_NAME} ...")
    try:
        dlq_response = sqs.create_queue(
            QueueName=DLQ_NAME,
            Attributes={
                # Messages in DLQ stay for 14 days (max) for inspection
                "MessageRetentionPeriod": str(14 * 24 * 3600),
            },
        )
        dlq_url = dlq_response["QueueUrl"]
        print(f"  ✅ DLQ created: {dlq_url}")
    except ClientError as e:
        if "QueueAlreadyExists" in str(e):
            dlq_url = sqs.get_queue_url(QueueName=DLQ_NAME)["QueueUrl"]
            print(f"  ℹ️  DLQ already exists: {dlq_url}")
        else:
            raise

    # Get DLQ ARN (needed to link it to the main queue)
    dlq_attrs = sqs.get_queue_attributes(
        QueueUrl=dlq_url, AttributeNames=["QueueArn"]
    )
    dlq_arn = dlq_attrs["Attributes"]["QueueArn"]
    print(f"  DLQ ARN: {dlq_arn}")

    # ── Step 2: Create main queue ─────────────────────────────────────────────
    print(f"\n[2/4] Creating main queue: {MAIN_QUEUE_NAME} ...")

    redrive_policy = json.dumps({
        "maxReceiveCount": str(SQS_MAX_RECEIVE_COUNT),
        "deadLetterTargetArn": dlq_arn,
    })

    try:
        main_response = sqs.create_queue(
            QueueName=MAIN_QUEUE_NAME,
            Attributes={
                # How long a message is hidden after being picked up by a worker
                "VisibilityTimeout": str(SQS_VISIBILITY_TIMEOUT),
                # Retain messages for 4 days if not processed
                "MessageRetentionPeriod": str(4 * 24 * 3600),
                # After SQS_MAX_RECEIVE_COUNT failures, move to DLQ
                "RedrivePolicy": redrive_policy,
            },
        )
        main_url = main_response["QueueUrl"]
        print(f"  ✅ Main queue created: {main_url}")
    except ClientError as e:
        if "QueueAlreadyExists" in str(e):
            main_url = sqs.get_queue_url(QueueName=MAIN_QUEUE_NAME)["QueueUrl"]
            print(f"  ℹ️  Main queue already exists: {main_url}")
        else:
            raise

    # Get main queue ARN (needed for S3 notification policy)
    main_attrs = sqs.get_queue_attributes(
        QueueUrl=main_url, AttributeNames=["QueueArn"]
    )
    main_arn = main_attrs["Attributes"]["QueueArn"]

    # ── Step 3: Set queue policy to allow S3 to send messages ─────────────────
    print(f"\n[3/4] Setting queue policy to allow S3 to publish ...")

    # This policy allows any S3 bucket in your account to send to this queue.
    # In production, restrict the source bucket ARN for tighter security.
    queue_policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "s3.amazonaws.com"},
                "Action": "sqs:SendMessage",
                "Resource": main_arn,
                "Condition": {
                    "StringLike": {
                        "aws:SourceArn": "arn:aws:s3:::*"
                    }
                },
            }
        ],
    })

    sqs.set_queue_attributes(
        QueueUrl=main_url,
        Attributes={"Policy": queue_policy},
    )
    print("  ✅ S3 → SQS publish permission set")

    # ── Step 4: Print summary ─────────────────────────────────────────────────
    print(f"\n[4/4] Summary")
    print("=" * 60)
    print(f"\n  Main Queue URL : {main_url}")
    print(f"  DLQ URL        : {dlq_url}")
    print(f"  Visibility     : {SQS_VISIBILITY_TIMEOUT}s")
    print(f"  Max retries    : {SQS_MAX_RECEIVE_COUNT} (then → DLQ)")

    print("\n" + "─" * 60)
    print("  Copy these into your .env file:")
    print("─" * 60)
    print(f'\n  SQS_QUEUE_URL="{main_url}"')
    print(f'  SQS_DLQ_URL="{dlq_url}"')

    print("\n  Next step:")
    print("  Configure S3 event notification to send to this queue.")
    print("  (See README — Step 4: Link S3 → SQS)")
    print()

    return main_url, dlq_url


if __name__ == "__main__":
    create_queues()
