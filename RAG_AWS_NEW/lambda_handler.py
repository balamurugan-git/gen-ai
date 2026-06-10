"""
lambda_handler.py
-----------------
AWS Lambda function triggered by S3 PUT/COPY events.

Trigger setup (S3 Event Notification):
  - Bucket: your-rag-documents-bucket
  - Event type: s3:ObjectCreated:*  (covers Put, Copy, CompleteMultipartUpload)
  - Prefix filter: documents/        (only trigger for files under this prefix)

Each S3 event may contain multiple records (batch upload).
Each record is processed independently.

Lambda IAM Role must have:
  - s3:GetObject on source bucket
  - bedrock:InvokeModel on Titan embedding model
  - es:ESHttpGet, es:ESHttpPost, es:ESHttpPut, es:ESHttpDelete on OpenSearch domain
"""

import json
import logging
import os
import urllib.parse

from ingestion_pipeline import IngestionPipeline

# ─── Logging setup ────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Pipeline (initialized at module level for Lambda warm-start reuse) ───────
# On Lambda cold start: instantiates pipeline (connects to OpenSearch, checks index)
# On warm invocations: reuses the same instance — saves 1-2 seconds per call
pipeline = IngestionPipeline()


def handler(event: dict, context) -> dict:
    """
    Lambda entry point.

    Args:
        event:   S3 event payload (see AWS docs: S3 Event Notifications)
        context: Lambda runtime context (used for request ID logging)

    Returns:
        dict with statusCode and results for each record processed.
    """
    request_id = getattr(context, "aws_request_id", "local-test")
    logger.info(f"Lambda invoked — RequestId={request_id}")
    logger.debug(f"Full event: {json.dumps(event, indent=2)}")

    records = event.get("Records", [])
    if not records:
        logger.warning("No records in S3 event — nothing to process")
        return {"statusCode": 200, "body": "No records to process"}

    results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    for record in records:
        result = _process_record(record)
        results.append(result)

        status = result.get("status", "error")
        if status == "success":
            success_count += 1
        elif status == "skipped":
            skip_count += 1
        else:
            error_count += 1

    summary = {
        "total_records":  len(records),
        "success":        success_count,
        "skipped":        skip_count,
        "errors":         error_count,
        "results":        results,
    }

    logger.info(f"Lambda complete — {summary}")

    # Lambda returns 200 even with partial errors — avoids infinite S3 retry loops
    # Individual errors are logged with full context for CloudWatch alerting
    return {
        "statusCode": 200,
        "body": json.dumps(summary),
    }


def _process_record(record: dict) -> dict:
    """
    Process a single S3 event record.

    Extracts bucket + key from the event, then calls the ingestion pipeline.
    Returns a result dict (never raises — errors are caught and logged).
    """
    try:
        # Extract bucket and key from S3 event structure
        s3_info = record.get("s3", {})
        bucket = s3_info.get("bucket", {}).get("name", "")
        # S3 keys in events are URL-encoded (spaces become +, etc.)
        raw_key = s3_info.get("object", {}).get("key", "")
        key = urllib.parse.unquote_plus(raw_key)

        if not bucket or not key:
            return {
                "status": "error",
                "reason": "Missing bucket or key in S3 event record",
                "record": record,
            }

        logger.info(f"Processing S3 event: s3://{bucket}/{key}")

        # Ignore folder creation events (S3 sends events for "directory" placeholders)
        if key.endswith("/"):
            return {
                "status": "skipped",
                "reason": "Folder creation event — not a file",
                "key": key,
            }

        # Run the ingestion pipeline
        result = pipeline.ingest(bucket=bucket, key=key)
        return result

    except Exception as e:
        logger.exception(f"Unhandled error processing record: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "record_keys": list(record.keys()),
        }


# ─── Local Testing ────────────────────────────────────────────────────────────
# Run: python lambda_handler.py
if __name__ == "__main__":
    import sys

    # Simulate an S3 event for local testing
    test_bucket = os.getenv("S3_BUCKET_NAME", "rag-documents-bucket")
    test_key = sys.argv[1] if len(sys.argv) > 1 else "documents/sample_loan_policy.pdf"

    mock_event = {
        "Records": [
            {
                "s3": {
                    "bucket": {"name": test_bucket},
                    "object": {"key": test_key},
                }
            }
        ]
    }

    class MockContext:
        aws_request_id = "local-test-001"

    print(f"\n{'='*60}")
    print(f"LOCAL TEST: s3://{test_bucket}/{test_key}")
    print(f"{'='*60}\n")

    result = handler(mock_event, MockContext())
    print(json.dumps(result, indent=2))
