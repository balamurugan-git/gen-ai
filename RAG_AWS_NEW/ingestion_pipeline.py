"""
ingestion_pipeline.py
---------------------
Core RAG data ingestion pipeline — ECS Fargate edition.

Changes from Lambda version:
  - _download_from_s3() now streams large files to /tmp instead of loading
    entirely into memory. Lambda had 10 GB RAM max; ECS Fargate can have 120 GB,
    but streaming is still best practice for files >100 MB.
  - Added explicit /tmp cleanup after processing (Lambda recycles containers,
    ECS tasks are long-running so /tmp fills up over time).
  - Comment updated: "cold start" reference removed (ECS has no cold starts).
  - Everything else (parse → embed → upsert) is identical to Lambda version.

Flow:
  S3 key (from SQS message)
      ↓
  Stream download → /tmp/
      ↓
  SHA-256 hash (change detection)
      ↓
  Check OpenSearch: new / unchanged / updated
      ↓
  Parse text (PDF / DOCX / HTML / PPTX)
      ↓
  Chunk → Bedrock embeddings → OpenSearch upsert
      ↓
  Cleanup /tmp
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import boto3

from config import SUPPORTED_EXTENSIONS
from document_parser import DocumentParser
from embeddings import BedrockEmbeddings
from opensearch_client import OpenSearchVectorClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the full RAG ingestion flow.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.ingest(bucket="my-bucket", key="documents/rbi_circular.pdf")

    The pipeline instance is created ONCE when the ECS worker starts, then
    reused for every message — no reconnect overhead per document.
    """

    def __init__(self):
        self.s3_client = boto3.client("s3")
        self.parser    = DocumentParser()
        self.embedder  = BedrockEmbeddings()
        self.os_client = OpenSearchVectorClient()

        # Ensure index exists — safe to call on worker startup (idempotent)
        self.os_client.ensure_index_exists()
        logger.info("IngestionPipeline initialised — OpenSearch index ready")

    # ─── Public Entry Point ───────────────────────────────────────────────────

    def ingest(self, bucket: str, key: str) -> dict:
        """
        Ingest a single document from S3 into OpenSearch.

        Args:
            bucket: S3 bucket name
            key:    S3 object key (e.g. "documents/rbi_master_circular.pdf")

        Returns:
            Result dict with status and ingestion stats.
        """
        file_name = Path(key).name
        file_ext  = Path(key).suffix.lower()
        tmp_path  = None

        logger.info(f"[Pipeline] Starting ingestion — bucket={bucket}, key={key}")

        try:
            # ── Step 0: Validate file type ──────────────────────────────────
            if file_ext not in SUPPORTED_EXTENSIONS:
                msg = f"Unsupported file type: '{file_ext}'"
                logger.warning(msg)
                return {"status": "skipped", "reason": msg, "key": key}

            # ── Step 1: Stream download from S3 → /tmp ──────────────────────
            logger.info("[Step 1] Streaming download from S3 → /tmp ...")
            tmp_path, file_size, last_modified = self._stream_download(bucket, key)
            logger.info(f"  Saved to {tmp_path} ({file_size:,} bytes), last_modified={last_modified}")

            # ── Step 2: Compute SHA-256 from the temp file ──────────────────
            doc_hash = self._hash_file(tmp_path)
            logger.info(f"[Step 2] SHA-256: {doc_hash[:12]}...")

            # ── Step 3: Deduplication / update check ────────────────────────
            logger.info("[Step 3] Checking OpenSearch for existing document...")
            check = self.os_client.document_exists(source_key=key)

            if check["exists"]:
                stored_hash = check.get("document_hash", "")
                if stored_hash == doc_hash:
                    msg = "Document unchanged (same hash) — skipping"
                    logger.info(f"  {msg}")
                    return {"status": "skipped", "reason": msg, "key": key, "hash": doc_hash}
                else:
                    logger.info(
                        f"  Document UPDATED ({stored_hash[:8]}... → {doc_hash[:8]}...)"
                    )
                    deleted = self.os_client.delete_document_chunks(source_key=key)
                    logger.info(f"  Deleted {deleted} old chunks")
            else:
                logger.info("  Document is NEW — first-time ingestion")

            # ── Step 4: Parse document ───────────────────────────────────────
            logger.info(f"[Step 4] Parsing document (type={file_ext}) ...")
            file_bytes = Path(tmp_path).read_bytes()
            chunks = self.parser.parse(file_bytes=file_bytes, file_extension=file_ext)

            if not chunks:
                msg = "No text extracted from document"
                logger.warning(msg)
                return {"status": "failed", "reason": msg, "key": key}

            logger.info(f"  Extracted {len(chunks)} chunks")

            # ── Step 5: Bedrock embeddings ───────────────────────────────────
            logger.info(f"[Step 5] Generating embeddings for {len(chunks)} chunks ...")
            embeddings = self.embedder.embed_batch(chunks)
            logger.info(f"  Generated {len(embeddings)} vectors (dim=1536)")

            # ── Step 6: Upsert into OpenSearch ──────────────────────────────
            metadata = {
                "source_bucket": bucket,
                "source_key":    key,
                "file_name":     file_name,
                "file_type":     file_ext.lstrip("."),
                "document_hash": doc_hash,
                "last_modified": last_modified,
            }

            logger.info(f"[Step 6] Upserting {len(chunks)} chunks into OpenSearch ...")
            inserted = self.os_client.upsert_chunks(
                chunks=chunks,
                embeddings=embeddings,
                metadata=metadata,
            )

            result = {
                "status":        "success",
                "key":           key,
                "file_name":     file_name,
                "file_type":     file_ext,
                "file_size_mb":  round(file_size / 1_048_576, 2),
                "chunks_total":  len(chunks),
                "chunks_stored": inserted,
                "document_hash": doc_hash,
                "last_modified": last_modified,
            }
            logger.info(f"[Pipeline] Complete — {result}")
            return result

        finally:
            # ── Cleanup: remove temp file regardless of success/failure ─────
            # Critical for long-running ECS tasks — /tmp does NOT self-clear
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                logger.debug(f"  Cleaned up temp file: {tmp_path}")

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _stream_download(self, bucket: str, key: str) -> tuple[str, int, str]:
        """
        Stream S3 object to a temp file in /tmp.

        Returns:
            (tmp_path, file_size_bytes, last_modified_str)

        Why streaming instead of .read():
            - Lambda version used response["Body"].read() — loads entire file into RAM
            - For a 200 MB annual report PDF, that's 200 MB of RAM just for the raw bytes
            - ECS tasks run for hours; streaming keeps peak memory low
            - Also allows processing files larger than available RAM (e.g. 2 GB PPTX)
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            last_modified = str(response.get("LastModified", "unknown"))

            # Create a named temp file with the correct extension (parsers use it)
            suffix = Path(key).suffix
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=tempfile.gettempdir())

            file_size = 0
            chunk_size = 8 * 1024 * 1024  # 8 MB chunks

            with os.fdopen(tmp_fd, "wb") as f:
                for chunk in response["Body"].iter_chunks(chunk_size=chunk_size):
                    f.write(chunk)
                    file_size += len(chunk)

            return tmp_path, file_size, last_modified

        except Exception as e:
            logger.error(f"S3 stream download failed for s3://{bucket}/{key}: {e}")
            raise

    def _hash_file(self, file_path: str) -> str:
        """
        Compute SHA-256 hash of a file without loading it fully into memory.
        Reads in 8 MB blocks — safe for very large files.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                sha256.update(block)
        return sha256.hexdigest()
