"""
opensearch_client.py
--------------------
Manages all interactions with Amazon OpenSearch:
  - Index creation with kNN vector mapping
  - Upsert (insert + update) of document chunks
  - Delete all chunks for a given document (for re-ingestion)
  - Semantic similarity search

Auth modes:
  - Basic auth (local / OpenSearch Serverless dev)
  - AWS SigV4 (production OpenSearch Service)

OpenSearch kNN plugin is used for approximate nearest neighbour search.
"""

import logging
from datetime import datetime, timezone

from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3

from config import (
    OPENSEARCH_HOST,
    OPENSEARCH_INDEX_NAME,
    OPENSEARCH_USERNAME,
    OPENSEARCH_PASSWORD,
    USE_AWS_OPENSEARCH,
    EMBEDDING_DIMENSIONS,
    AWS_REGION,
)

logger = logging.getLogger(__name__)


class OpenSearchVectorClient:
    """
    Client for OpenSearch vector operations.

    Usage:
        client = OpenSearchVectorClient()
        client.ensure_index_exists()
        client.upsert_chunks(chunks_with_embeddings, metadata)
        results = client.search(query_vector, top_k=5)
    """

    def __init__(self):
        self.index_name = OPENSEARCH_INDEX_NAME
        self.client = self._build_client()
        logger.info(
            f"OpenSearchVectorClient connected — host={OPENSEARCH_HOST}, "
            f"index={self.index_name}, aws_auth={USE_AWS_OPENSEARCH}"
        )

    # ─── Client Setup ─────────────────────────────────────────────────────────

    def _build_client(self) -> OpenSearch:
        """Build OpenSearch client with appropriate auth."""
        if USE_AWS_OPENSEARCH:
            # Production: AWS SigV4 auth (IAM role-based)
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, AWS_REGION, "es")
        else:
            # Dev/local: basic auth
            auth = (OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD)

        return OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST.replace("https://", "").replace("http://", ""), "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )

    # ─── Index Management ─────────────────────────────────────────────────────

    def ensure_index_exists(self) -> None:
        """
        Create the vector index if it doesn't exist.
        Index mapping includes:
          - 'embedding': knn_vector field (1536 dims, cosine similarity)
          - 'content': text field (BM25 for hybrid search)
          - metadata fields: keyword for exact filtering
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists — skipping creation")
            return

        index_body = {
            "settings": {
                "index": {
                    "knn": True,               # Enable kNN plugin
                    "knn.algo_param.ef_search": 100,
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                }
            },
            "mappings": {
                "properties": {
                    # ── Vector field ──
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIMENSIONS,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24,
                            },
                        },
                    },
                    # ── Content field (for BM25 / keyword fallback) ──
                    "content": {"type": "text", "analyzer": "english"},

                    # ── Metadata fields ──
                    "source_bucket":   {"type": "keyword"},
                    "source_key":      {"type": "keyword"},
                    "file_name":       {"type": "keyword"},
                    "file_type":       {"type": "keyword"},
                    "chunk_index":     {"type": "integer"},
                    "total_chunks":    {"type": "integer"},
                    "document_hash":   {"type": "keyword"},  # SHA-256 of original file
                    "ingested_at":     {"type": "date"},
                    "last_modified":   {"type": "keyword"},
                }
            },
        }

        self.client.indices.create(index=self.index_name, body=index_body)
        logger.info(f"Created index '{self.index_name}' with kNN vector mapping")

    # ─── Document Operations ──────────────────────────────────────────────────

    def document_exists(self, source_key: str) -> dict | None:
        """
        Check if a document (by S3 key) is already in the index.
        Returns the stored document_hash if found, else None.

        Used to detect:
          - First ingestion (not exists → ingest)
          - Re-upload of same version (hash match → skip)
          - Re-upload of updated version (hash mismatch → delete old + ingest new)
        """
        query = {
            "size": 1,
            "query": {"term": {"source_key": source_key}},
            "_source": ["document_hash"],
        }
        try:
            response = self.client.search(index=self.index_name, body=query)
            hits = response["hits"]["hits"]
            if hits:
                return {
                    "exists": True,
                    "document_hash": hits[0]["_source"]["document_hash"],
                    "id_sample": hits[0]["_id"],
                }
            return {"exists": False}
        except Exception as e:
            logger.error(f"Error checking document existence: {e}")
            return {"exists": False}

    def delete_document_chunks(self, source_key: str) -> int:
        """
        Delete ALL chunks for a given S3 source_key.
        Called before re-ingesting an updated document version.

        Returns: number of deleted documents
        """
        query = {"query": {"term": {"source_key": source_key}}}
        response = self.client.delete_by_query(
            index=self.index_name,
            body=query,
            refresh=True,  # Make deletion visible immediately
        )
        deleted = response.get("deleted", 0)
        logger.info(f"Deleted {deleted} old chunks for key='{source_key}'")
        return deleted

    def upsert_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict,
    ) -> int:
        """
        Bulk-insert chunks with their embeddings and metadata into OpenSearch.

        Each document stored in OpenSearch = one chunk with:
          - embedding: the vector
          - content: the raw text
          - metadata fields

        Returns: number of chunks inserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        total_chunks = len(chunks)
        now_iso = datetime.now(timezone.utc).isoformat()

        bulk_body = []
        for idx, (chunk_text, vector) in enumerate(zip(chunks, embeddings)):
            # Document ID = source_key + chunk index (deterministic for updates)
            doc_id = f"{metadata['source_key']}::chunk_{idx}"

            doc = {
                "embedding":     vector,
                "content":       chunk_text,
                "source_bucket": metadata.get("source_bucket", ""),
                "source_key":    metadata.get("source_key", ""),
                "file_name":     metadata.get("file_name", ""),
                "file_type":     metadata.get("file_type", ""),
                "chunk_index":   idx,
                "total_chunks":  total_chunks,
                "document_hash": metadata.get("document_hash", ""),
                "ingested_at":   now_iso,
                "last_modified": metadata.get("last_modified", ""),
            }

            # Use index (not create) so it's idempotent
            bulk_body.append({"index": {"_index": self.index_name, "_id": doc_id}})
            bulk_body.append(doc)

        if not bulk_body:
            return 0

        response = self.client.bulk(body=bulk_body, refresh=True)

        errors = [item for item in response["items"] if "error" in item.get("index", {})]
        if errors:
            logger.error(f"Bulk insert had {len(errors)} errors: {errors[:2]}")

        inserted = total_chunks - len(errors)
        logger.info(f"Upserted {inserted}/{total_chunks} chunks into OpenSearch")
        return inserted

    # ─── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_file_type: str | None = None,
        filter_source_key: str | None = None,
    ) -> list[dict]:
        """
        Semantic (kNN) search over stored embeddings.

        Args:
            query_vector:      Embedding of the user's query
            top_k:             Number of results to return
            filter_file_type:  Optional — restrict to a file type (e.g. "pdf")
            filter_source_key: Optional — restrict to one specific document

        Returns:
            List of result dicts with: content, score, metadata
        """
        knn_query = {
            "vector": query_vector,
            "k": top_k,
        }

        # Optional metadata filters
        filters = []
        if filter_file_type:
            filters.append({"term": {"file_type": filter_file_type}})
        if filter_source_key:
            filters.append({"term": {"source_key": filter_source_key}})

        if filters:
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"knn": {"embedding": knn_query}}],
                        "filter": filters,
                    }
                },
                "_source": {
                    "excludes": ["embedding"]  # Don't return raw vectors in results
                },
            }
        else:
            query_body = {
                "size": top_k,
                "query": {"knn": {"embedding": knn_query}},
                "_source": {"excludes": ["embedding"]},
            }

        response = self.client.search(index=self.index_name, body=query_body)
        hits = response["hits"]["hits"]

        results = []
        for hit in hits:
            src = hit["_source"]
            results.append({
                "content":       src.get("content", ""),
                "score":         hit["_score"],
                "file_name":     src.get("file_name", ""),
                "source_key":    src.get("source_key", ""),
                "file_type":     src.get("file_type", ""),
                "chunk_index":   src.get("chunk_index", 0),
                "total_chunks":  src.get("total_chunks", 0),
                "document_hash": src.get("document_hash", ""),
                "ingested_at":   src.get("ingested_at", ""),
            })

        return results
