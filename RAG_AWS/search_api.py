"""
search_api.py
-------------
FastAPI application exposing two endpoints:

  POST /search      → Returns top-k relevant chunks (semantic search only)
  POST /ask         → Returns an LLM-generated answer using retrieved context (RAG)

Run locally:
  uvicorn search_api:app --host 0.0.0.0 --port 8000 --reload

Deploy on AWS:
  - Package as Lambda + API Gateway (using Mangum adapter)
  - Or run on ECS/EC2 behind ALB
"""

import json
import logging

import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import BEDROCK_REGION, BEDROCK_LLM_MODEL_ID
from embeddings import BedrockEmbeddings
from opensearch_client import OpenSearchVectorClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RAG Search API",
    description="Semantic search + LLM answer generation over ingested documents",
    version="1.0.0",
)

# ─── Singleton clients (initialized once at startup) ─────────────────────────
embedder = BedrockEmbeddings()
os_client = OpenSearchVectorClient()
bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# ─── Request / Response Schemas ───────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="User's search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    filter_file_type: str | None = Field(None, description="Filter by file type: pdf, docx, html, pptx")
    filter_source_key: str | None = Field(None, description="Filter by specific S3 object key")


class ChunkResult(BaseModel):
    content:      str
    score:        float
    file_name:    str
    source_key:   str
    file_type:    str
    chunk_index:  int
    total_chunks: int
    ingested_at:  str


class SearchResponse(BaseModel):
    query:   str
    results: list[ChunkResult]
    total:   int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    top_k:    int = Field(5, ge=1, le=10)
    filter_file_type: str | None = None


class AskResponse(BaseModel):
    question: str
    answer:   str
    sources:  list[ChunkResult]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check — use this for ALB/ECS target group checks."""
    return {"status": "healthy", "service": "rag-search-api"}


@app.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest):
    """
    Semantic similarity search over the OpenSearch vector index.

    Steps:
      1. Embed the query using Bedrock Titan
      2. kNN search in OpenSearch
      3. Return top-k matching chunks with metadata
    """
    logger.info(f"/search — query='{request.query}', top_k={request.top_k}")

    try:
        query_vector = embedder.embed_text(request.query, input_type="search_query")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding service error: {e}")

    try:
        raw_results = os_client.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filter_file_type=request.filter_file_type,
            filter_source_key=request.filter_source_key,
        )
    except Exception as e:
        logger.error(f"OpenSearch search failed: {e}")
        raise HTTPException(status_code=503, detail=f"Search error: {e}")

    results = [ChunkResult(**r) for r in raw_results]

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    RAG endpoint: retrieve relevant chunks → send to Bedrock Claude → return answer.

    Steps:
      1. Embed the question
      2. Retrieve top-k chunks from OpenSearch
      3. Build a prompt with retrieved context
      4. Call Bedrock Claude 3 Sonnet for answer generation
      5. Return answer + source citations
    """
    logger.info(f"/ask — question='{request.question}', top_k={request.top_k}")

    # Step 1: Embed query
    try:
        query_vector = embedder.embed_text(request.question, input_type="search_query")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Embedding error: {e}")

    # Step 2: Retrieve context
    try:
        raw_results = os_client.search(
            query_vector=query_vector,
            top_k=request.top_k,
            filter_file_type=request.filter_file_type,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retrieval error: {e}")

    if not raw_results:
        return AskResponse(
            question=request.question,
            answer="No relevant documents found for your question.",
            sources=[],
        )

    # Step 3: Build RAG prompt
    context_parts = []
    for i, chunk in enumerate(raw_results, start=1):
        context_parts.append(
            f"[Source {i} — {chunk['file_name']}, chunk {chunk['chunk_index'] + 1}]\n"
            f"{chunk['content']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough information in the documents to answer this."
Do not make up facts. Cite source numbers when referencing specific information.

CONTEXT:
{context_block}

QUESTION: {request.question}

ANSWER:"""

    # Step 4: Call Bedrock Claude
    try:
        bedrock_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_LLM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(bedrock_payload),
        )

        response_body = json.loads(response["body"].read())
        answer = response_body["content"][0]["text"].strip()

    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {e}")
        raise HTTPException(status_code=503, detail=f"LLM generation error: {e}")

    sources = [ChunkResult(**r) for r in raw_results]

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
    )


# ─── Run locally ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
