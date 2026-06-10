"""
sqs_worker.py
-------------
Long-running SQS consumer — the ECS Fargate entry point.

Replaces lambda_handler.py entirely.

Architecture:
    S3 PUT event
        → S3 Event Notification → SQS Queue
            → This worker (polling loop)
                → IngestionPipeline.ingest()
                    → Bedrock + OpenSearch

Why SQS instead of direct S3 trigger:
    - Lambda: S3 calls Lambda directly. 15-min hard limit. One call per file.
    - ECS+SQS: S3 sends event to a queue. Worker polls at its own pace.
      No timeout. Can process a 500-page PDF without any clock pressure.
      Queue acts as a buffer — 1000 files uploaded at once? They wait in
      the queue; worker processes them one by one (or in parallel threads).

Visibility timeout:
    When a worker picks up a message, SQS hides it from other workers for
    SQS_VISIBILITY_TIMEOUT seconds. If the worker crashes mid-processing,
    the message reappears after the timeout and another worker retries it.
    After SQS_MAX_RECEIVE_COUNT failures, it moves to the DLQ.

Graceful shutdown:
    ECS sends SIGTERM before stopping a task (during deployments, scale-in).
    The worker catches SIGTERM, finishes the current document, then exits.
    No document is left half-processed.

Run locally:
    python sqs_worker.py

Run in ECS:
    CMD ["python", "sqs_worker.py"]   ← in Dockerfile
"""

import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.parse

import boto3

from config import (
    SQS_QUEUE_URL,
    SQS_BATCH_SIZE,
    SQS_WAIT_SECONDS,
    SQS_VISIBILITY_TIMEOUT,
    WORKER_CONCURRENCY,
    AWS_REGION,
)
from ingestion_pipeline import IngestionPipeline

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,   # ECS captures stdout → CloudWatch Logs
)
logger = logging.getLogger(__name__)


# ─── Graceful Shutdown ────────────────────────────────────────────────────────
# ECS sends SIGTERM 30 seconds before force-killing the container.
# We set this flag and let the polling loop finish its current document.
_shutdown_requested = False

def _handle_sigterm(signum, frame):
    global _shutdown_requested
    logger.info("SIGTERM received — finishing current document then shutting down...")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT,  _handle_sigterm)   # Ctrl+C during local testing


# ─── Worker ───────────────────────────────────────────────────────────────────

class SQSWorker:
    """
    Polls an SQS queue and calls IngestionPipeline for each message.

    One IngestionPipeline instance is shared across all messages processed
    by this worker — avoids reconnecting to OpenSearch on every document.
    """

    def __init__(self):
        if not SQS_QUEUE_URL:
            raise ValueError(
                "SQS_QUEUE_URL is not set. "
                "Add it to your .env file or ECS task environment variables."
            )
        self.sqs      = boto3.client("sqs", region_name=AWS_REGION)
        self.pipeline = IngestionPipeline()
        self._stats   = {"processed": 0, "skipped": 0, "failed": 0}

        logger.info(f"SQSWorker ready — queue={SQS_QUEUE_URL}")
        logger.info(f"  concurrency={WORKER_CONCURRENCY}, batch={SQS_BATCH_SIZE}, "
                    f"visibility_timeout={SQS_VISIBILITY_TIMEOUT}s")

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def run(self):
        """
        Poll SQS queue in a loop until SIGTERM.
        Each message = one S3 file to ingest.
        """
        logger.info("Worker loop started — waiting for messages...")

        while not _shutdown_requested:
            try:
                messages = self._poll()

                if not messages:
                    # Queue is empty — long-poll already waited SQS_WAIT_SECONDS
                    logger.debug("Queue empty — polling again...")
                    continue

                logger.info(f"Received {len(messages)} message(s)")

                if WORKER_CONCURRENCY == 1:
                    # Single-threaded — simple sequential processing
                    for msg in messages:
                        self._handle_message(msg)
                else:
                    # Multi-threaded — process batch in parallel
                    threads = [
                        threading.Thread(target=self._handle_message, args=(msg,))
                        for msg in messages
                    ]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()

            except Exception as e:
                logger.exception(f"Unexpected error in worker loop: {e}")
                time.sleep(5)  # Brief pause before retrying to avoid tight error loops

        # ── Shutdown ──────────────────────────────────────────────────────────
        logger.info(
            f"Worker shutting down — stats: "
            f"processed={self._stats['processed']}, "
            f"skipped={self._stats['skipped']}, "
            f"failed={self._stats['failed']}"
        )

    # ─── Message Handling ─────────────────────────────────────────────────────

    def _poll(self) -> list:
        """Pull up to SQS_BATCH_SIZE messages from the queue."""
        response = self.sqs.receive_message(
            QueueUrl            = SQS_QUEUE_URL,
            MaxNumberOfMessages = SQS_BATCH_SIZE,
            WaitTimeSeconds     = SQS_WAIT_SECONDS,   # Long-polling
            VisibilityTimeout   = SQS_VISIBILITY_TIMEOUT,
            AttributeNames      = ["ApproximateReceiveCount"],
        )
        return response.get("Messages", [])

    def _handle_message(self, message: dict):
        """
        Process one SQS message end-to-end.

        On success  → delete message from queue (prevents re-processing)
        On failure  → do NOT delete → SQS will retry up to max_receive_count
                    → after that, SQS moves it to the DLQ automatically
        """
        receipt_handle   = message["ReceiptHandle"]
        receive_count    = int(message.get("Attributes", {}).get("ApproximateReceiveCount", 1))
        message_id       = message.get("MessageId", "unknown")

        logger.info(f"[MSG {message_id[:8]}] Processing (attempt #{receive_count})")

        try:
            bucket, key = self._parse_s3_event(message)
        except Exception as e:
            logger.error(f"[MSG {message_id[:8]}] Could not parse S3 event: {e}")
            logger.error(f"  Raw body: {message.get('Body', '')[:500]}")
            # Delete malformed messages — they will never succeed, no point retrying
            self._delete_message(receipt_handle)
            self._stats["failed"] += 1
            return

        # Skip folder creation events
        if key.endswith("/"):
            logger.info(f"[MSG {message_id[:8]}] Folder event — skipping")
            self._delete_message(receipt_handle)
            self._stats["skipped"] += 1
            return

        try:
            result = self.pipeline.ingest(bucket=bucket, key=key)
            status = result.get("status", "unknown")

            if status in ("success", "skipped"):
                # Only delete from queue on clean completion
                self._delete_message(receipt_handle)
                if status == "success":
                    self._stats["processed"] += 1
                    logger.info(
                        f"[MSG {message_id[:8]}] SUCCESS — "
                        f"{result.get('chunks_stored')} chunks stored "
                        f"({result.get('file_size_mb')} MB)"
                    )
                else:
                    self._stats["skipped"] += 1
                    logger.info(
                        f"[MSG {message_id[:8]}] SKIPPED — {result.get('reason')}"
                    )
            else:
                # "failed" status — leave in queue for SQS retry
                logger.warning(
                    f"[MSG {message_id[:8]}] FAILED (attempt {receive_count}) — "
                    f"{result.get('reason')}. Will retry."
                )
                self._stats["failed"] += 1

        except Exception as e:
            # Unexpected exception — leave in queue for SQS retry
            logger.exception(
                f"[MSG {message_id[:8]}] Exception on attempt {receive_count}: {e}"
            )
            self._stats["failed"] += 1

    def _parse_s3_event(self, message: dict) -> tuple[str, str]:
        """
        Extract S3 bucket and key from an SQS message wrapping an S3 event.

        SQS message structure:
          message["Body"] = JSON string of S3 Event Notification
            → Records[0]["s3"]["bucket"]["name"]
            → Records[0]["s3"]["object"]["key"]

        S3 keys in events are URL-encoded (spaces → +, special chars → %xx).
        """
        body = json.loads(message["Body"])

        # S3 sends a test event when you first configure the notification
        if "Event" in body and body["Event"] == "s3:TestEvent":
            raise ValueError("S3 test event — ignoring")

        records = body.get("Records", [])
        if not records:
            raise ValueError(f"No Records in S3 event body: {body}")

        s3_info = records[0].get("s3", {})
        bucket  = s3_info.get("bucket", {}).get("name", "")
        raw_key = s3_info.get("object", {}).get("key", "")
        key     = urllib.parse.unquote_plus(raw_key)

        if not bucket or not key:
            raise ValueError(f"Missing bucket or key. bucket='{bucket}', key='{key}'")

        return bucket, key

    def _delete_message(self, receipt_handle: str):
        """Remove a successfully processed message from the queue."""
        try:
            self.sqs.delete_message(
                QueueUrl      = SQS_QUEUE_URL,
                ReceiptHandle = receipt_handle,
            )
        except Exception as e:
            logger.warning(f"Failed to delete SQS message: {e}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  RAG Ingestion Worker — ECS Fargate Edition")
    logger.info("=" * 55)
    logger.info(f"  SQS Queue  : {SQS_QUEUE_URL or 'NOT SET'}")
    logger.info(f"  Concurrency: {WORKER_CONCURRENCY}")
    logger.info(f"  Region     : {AWS_REGION}")
    logger.info("=" * 55)

    worker = SQSWorker()
    worker.run()
