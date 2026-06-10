"""
config.py
---------
Central configuration for the AWS RAG Ingestion Pipeline (ECS Fargate + SQS edition).

Changes from Lambda version:
  - Added SQS queue settings (replaces S3 direct trigger)
  - Added ECS worker settings (concurrency, visibility timeout)
  - Removed Lambda-specific comment about cold starts
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── AWS Settings ────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ─── S3 Settings ─────────────────────────────────────────────────────────────
S3_BUCKET_NAME    = os.getenv("S3_BUCKET_NAME", "rag-documents-bucket")
S3_PREFIX_FILTER  = os.getenv("S3_PREFIX_FILTER", "documents/")

# ─── SQS Settings ────────────────────────────────────────────────────────────
# Main queue: S3 events land here, worker picks them up
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL", "")

# Dead-letter queue: messages that fail 3 times are moved here for inspection
SQS_DLQ_URL   = os.getenv("SQS_DLQ_URL", "")

# How long (seconds) a message is invisible after a worker picks it up.
# Must be > max time to process one document.
# Large PPTX with 500 slides can take ~8 min → set to 600s (10 min) to be safe.
SQS_VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "600"))

# How many SQS messages to pull per batch (1 keeps it simple; max is 10)
SQS_BATCH_SIZE = int(os.getenv("SQS_BATCH_SIZE", "1"))

# How long to wait for a message if the queue is empty (long-polling, reduces API calls)
SQS_WAIT_SECONDS = int(os.getenv("SQS_WAIT_SECONDS", "20"))

# How many times to retry a failed message before sending to DLQ
SQS_MAX_RECEIVE_COUNT = int(os.getenv("SQS_MAX_RECEIVE_COUNT", "3"))

# ─── ECS Worker Settings ─────────────────────────────────────────────────────
# Number of parallel worker threads inside one ECS task.
# Each thread picks one SQS message and processes it independently.
# Keep at 1 for dev; set 2-4 for production (match your Fargate vCPU allocation).
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))

# ─── Bedrock Settings ────────────────────────────────────────────────────────
BEDROCK_REGION             = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSIONS       = 1024  # Titan V2 supports 256 | 512 | 1024
BEDROCK_LLM_MODEL_ID       = "anthropic.claude-3-sonnet-20240229-v1:0"

# ─── OpenSearch Settings ─────────────────────────────────────────────────────
OPENSEARCH_HOST       = os.getenv("OPENSEARCH_HOST", "https://your-domain.us-east-1.es.amazonaws.com")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "rag-vector-index")
OPENSEARCH_USERNAME   = os.getenv("OPENSEARCH_USERNAME", "admin")
OPENSEARCH_PASSWORD   = os.getenv("OPENSEARCH_PASSWORD", "admin")
USE_AWS_OPENSEARCH    = os.getenv("USE_AWS_OPENSEARCH", "false").lower() == "true"

# ─── Chunking Settings ───────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ─── Supported File Types ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".pptx", ".ppt"}

# ─── Metadata Fields Stored in OpenSearch ────────────────────────────────────
METADATA_FIELDS = [
    "source_bucket",
    "source_key",
    "file_name",
    "file_type",
    "chunk_index",
    "total_chunks",
    "document_hash",
    "ingested_at",
    "last_modified",
]
