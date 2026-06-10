"""
opensearch_setup.py
-------------------
One-time setup script — run BEFORE deploying Lambda or the search API.

What it does:
  1. Connects to OpenSearch using settings from config.py
  2. Creates the kNN vector index with correct mappings
  3. Verifies the index health
  4. Prints a summary

Run:
  python opensearch_setup.py
  python opensearch_setup.py --delete-existing   # ⚠️ Drops and recreates the index
"""

import argparse
import json
import logging
import sys

from opensearch_client import OpenSearchVectorClient
from config import OPENSEARCH_HOST, OPENSEARCH_INDEX_NAME, EMBEDDING_DIMENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def setup(delete_existing: bool = False):
    print("\n" + "=" * 60)
    print("  AWS RAG Pipeline — OpenSearch Index Setup")
    print("=" * 60)
    print(f"  Host  : {OPENSEARCH_HOST}")
    print(f"  Index : {OPENSEARCH_INDEX_NAME}")
    print(f"  Dims  : {EMBEDDING_DIMENSIONS}")
    print("=" * 60 + "\n")

    client = OpenSearchVectorClient()
    os = client.client  # Raw opensearch-py client

    # ── Step 1: Optional delete ────────────────────────────────────────────
    if delete_existing:
        if os.indices.exists(index=OPENSEARCH_INDEX_NAME):
            confirm = input(
                f"⚠️  Delete existing index '{OPENSEARCH_INDEX_NAME}'? "
                f"This will erase ALL vectors. Type 'yes' to confirm: "
            ).strip()
            if confirm.lower() != "yes":
                print("Aborted.")
                sys.exit(0)
            os.indices.delete(index=OPENSEARCH_INDEX_NAME)
            print(f"✅ Deleted index: {OPENSEARCH_INDEX_NAME}")
        else:
            print(f"ℹ️  Index '{OPENSEARCH_INDEX_NAME}' does not exist — nothing to delete")

    # ── Step 2: Create index ───────────────────────────────────────────────
    print("Creating index...")
    client.ensure_index_exists()
    print(f"✅ Index ready: {OPENSEARCH_INDEX_NAME}")

    # ── Step 3: Verify ────────────────────────────────────────────────────
    print("\nVerifying index health...")
    health = os.cluster.health(index=OPENSEARCH_INDEX_NAME)
    mapping = os.indices.get_mapping(index=OPENSEARCH_INDEX_NAME)
    stats = os.indices.stats(index=OPENSEARCH_INDEX_NAME)

    doc_count = stats["indices"][OPENSEARCH_INDEX_NAME]["total"]["docs"]["count"]
    status = health.get("status", "unknown")

    print(f"\n{'─'*40}")
    print(f"  Index name : {OPENSEARCH_INDEX_NAME}")
    print(f"  Status     : {status.upper()}")
    print(f"  Documents  : {doc_count}")
    print(f"  Shards     : {health.get('active_shards', '?')}")
    print(f"{'─'*40}")

    # Show vector field mapping
    props = mapping[OPENSEARCH_INDEX_NAME]["mappings"].get("properties", {})
    embedding_field = props.get("embedding", {})
    print(f"\n  Vector field mapping:")
    print(f"    type      : {embedding_field.get('type')}")
    print(f"    dimension : {embedding_field.get('dimension')}")
    method = embedding_field.get("method", {})
    print(f"    algorithm : {method.get('name')} / {method.get('space_type')}")

    print(f"\n✅ Setup complete. Index is {'healthy' if status == 'green' else 'available'}.")
    print("\nNext steps:")
    print("  1. Upload a document to S3 → Lambda will auto-trigger ingestion")
    print("  2. Or test locally: python lambda_handler.py documents/your_file.pdf")
    print("  3. Start search API: uvicorn search_api:app --reload")
    print()


def list_documents():
    """List all unique documents currently stored in the index."""
    client = OpenSearchVectorClient()
    os = client.client

    query = {
        "size": 0,
        "aggs": {
            "unique_docs": {
                "terms": {
                    "field": "source_key",
                    "size": 100,
                }
            }
        },
    }

    try:
        response = os.search(index=OPENSEARCH_INDEX_NAME, body=query)
        buckets = response["aggregations"]["unique_docs"]["buckets"]

        if not buckets:
            print("\nNo documents indexed yet.")
            return

        print(f"\n{'─'*60}")
        print(f"  Documents in index: {OPENSEARCH_INDEX_NAME}")
        print(f"{'─'*60}")
        for b in buckets:
            print(f"  [{b['doc_count']:>4} chunks]  {b['key']}")
        print(f"{'─'*60}")
        print(f"  Total unique documents: {len(buckets)}")

    except Exception as e:
        print(f"Error listing documents: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenSearch index setup for RAG pipeline")
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete and recreate the index (DESTRUCTIVE — erases all vectors)",
    )
    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List all documents currently stored in the index",
    )
    args = parser.parse_args()

    if args.list_docs:
        list_documents()
    else:
        setup(delete_existing=args.delete_existing)
