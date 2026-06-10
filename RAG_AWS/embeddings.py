"""
embeddings.py
-------------
Wraps Amazon Bedrock Titan Embeddings V2 for generating vector embeddings.

Model: amazon.titan-embed-text-v2:0  (native AWS — no Marketplace subscription)
Output: 1024-dimensional float vector per text input.

Titan V2 request/response format:
  - Request:  {"inputText": "...", "dimensions": 1024, "normalize": true}
  - Response: {"embedding": [...], "inputTextTokenCount": N}
  - Titan V2 supports dimensions of 256, 512, or 1024 (NOT 1536 — that was V1).
  - Titan has no notion of query vs document input types (unlike Cohere), so the
    `input_type` argument below is accepted for API compatibility but ignored.

Handles:
  - Single text embedding
  - Batch embedding with retry logic
  - Token limit safety (Titan V2 max: 8192 tokens ≈ ~30000 chars)
"""

import json
import logging
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import BEDROCK_REGION, BEDROCK_EMBEDDING_MODEL_ID, EMBEDDING_DIMENSIONS

logger = logging.getLogger(__name__)

# Titan V2 safe character limit per call (~8192 tokens ≈ 30000 chars)
MAX_CHARS_PER_CALL = 30_000

# Retry settings for throttling
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 10  # exponential: 10s, 20s, 30s, 40s, 50s


class BedrockEmbeddings:
    """
    Generates text embeddings using Amazon Bedrock Titan Embeddings V2.

    Usage:
        embedder = BedrockEmbeddings()
        vector = embedder.embed_text("What is CIBIL score?")
        # Returns list of 1024 floats

        vectors = embedder.embed_batch(["chunk1", "chunk2", ...])
        # Returns list of vectors

    The `input_type` argument on embed_text/embed_batch is accepted but ignored
    (Titan, unlike Cohere, uses the same embedding for queries and documents).
    """

    def __init__(self):
        # Disable botocore's own retries — our _call_bedrock handles backoff
        _no_retry = Config(retries={"max_attempts": 1, "mode": "standard"})
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=_no_retry,
        )
        self.model_id = BEDROCK_EMBEDDING_MODEL_ID
        logger.info(f"BedrockEmbeddings initialized — model={self.model_id}, region={BEDROCK_REGION}")

    # ─── Public Methods ───────────────────────────────────────────────────────

    def embed_text(self, text: str, input_type: str = "search_document") -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text:       Input text (will be truncated if > MAX_CHARS_PER_CALL)
            input_type: Ignored for Titan (kept for API compatibility with callers
                        that distinguish query vs document embeddings).

        Returns:
            List of floats (1024 dimensions for Titan V2)
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to embed_text — returning zero vector")
            return [0.0] * EMBEDDING_DIMENSIONS

        # Truncate if too long
        if len(text) > MAX_CHARS_PER_CALL:
            logger.warning(
                f"Text truncated: {len(text)} → {MAX_CHARS_PER_CALL} chars"
            )
            text = text[:MAX_CHARS_PER_CALL]

        return self._call_bedrock(text)

    def embed_batch(
        self, texts: list[str], input_type: str = "search_document"
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Bedrock Titan does NOT have a native batch API — each text is embedded
        via a separate call, with a small delay between calls to stay under
        Bedrock rate limits.

        Args:
            texts:      List of text chunks
            input_type: Ignored for Titan (kept for API compatibility).

        Returns:
            List of embedding vectors (one per chunk)
        """
        embeddings = []
        total = len(texts)

        logger.info(f"Embedding {total} chunks via Bedrock Titan...")

        for idx, text in enumerate(texts):
            vector = self.embed_text(text)
            embeddings.append(vector)

            # Progress log every 10 chunks
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                logger.info(f"  Embedded {idx + 1}/{total} chunks")

            # Throttle buffer between calls
            if idx < total - 1:
                time.sleep(1.0)

        logger.info(f"Batch embedding complete — {total} vectors generated")
        return embeddings

    # ─── Private Methods ──────────────────────────────────────────────────────

    def _call_bedrock(self, text: str, attempt: int = 1) -> list[float]:
        """
        Call Bedrock InvokeModel for a single text with retry on throttle.
        """
        payload = {
            "inputText": text,
            "dimensions": EMBEDDING_DIMENSIONS,  # Titan V2: 256 | 512 | 1024
            "normalize": True,                   # unit-length vectors → cosine similarity
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            result = json.loads(response["body"].read())
            return result["embedding"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            if error_code == "ThrottlingException" and attempt <= MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    f"Bedrock throttled — retry {attempt}/{MAX_RETRIES} in {wait}s"
                )
                time.sleep(wait)
                return self._call_bedrock(text, attempt + 1)

            elif error_code == "ValidationException":
                logger.error(f"Bedrock validation error: {e}")
                raise ValueError(f"Invalid input to Bedrock: {e}") from e

            else:
                logger.error(f"Bedrock call failed [{error_code}]: {e}")
                raise
