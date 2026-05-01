"""
FastAPI app for Hushdoc.

P1 scope:
  - GET    /api/health             — cheap liveness + lazy-load status
  - GET    /api/documents          — list indexed filenames + summaries
  - DELETE /api/documents          — wipe the vector store + summaries
  - POST   /api/chat/clear         — reset a chat session's memory

Streaming chat, file uploads, and voice routes ship in P2.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import doc_summaries
from server import deps
from server.schemas import (
    ChatClearRequest,
    ChatClearResponse,
    DeleteDocumentsResponse,
    DocumentsResponse,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("server.main")


app = FastAPI(
    title="Hushdoc API",
    description=(
        "Local-only HTTP API for the Hushdoc PDF assistant. Wraps the "
        "ingest / vector-store / RAG-chain modules behind a small set of "
        "JSON + SSE endpoints so the React frontend can drive them without "
        "running Streamlit."
    ),
    version="0.5.0",
)

# CORS — Vite dev server on :5173 needs to call us on :8000. Production
# build will be served from the same origin so this becomes a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + lazy-load status. Never triggers heavy initialization."""
    store = deps.get_store_if_loaded()
    chain = deps.get_chain_if_loaded()
    if store is not None:
        try:
            count = store.count()
            files = store.list_filenames()
        except Exception:
            count, files = 0, []
    else:
        count, files = 0, []
    return HealthResponse(
        ok=True,
        chain_loaded=chain is not None,
        store_loaded=store is not None,
        vector_count=count,
        indexed_files=files,
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
@app.get("/api/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    store = deps.get_store()
    return DocumentsResponse(
        filenames=store.list_filenames(),
        chunk_count=store.count(),
        summaries=doc_summaries.all_summaries(),
    )


@app.delete("/api/documents", response_model=DeleteDocumentsResponse)
def delete_documents() -> DeleteDocumentsResponse:
    """Wipe every chunk + every cached summary. Idempotent."""
    store = deps.get_store()
    was = store.count()
    store.reset()
    doc_summaries.clear_all()
    logger.info("Vector store wiped (was %d chunks).", was)
    return DeleteDocumentsResponse(ok=True, was_count=was)


# ---------------------------------------------------------------------------
# Chat (P1: clear-only; streaming chat lands in P2)
# ---------------------------------------------------------------------------
@app.post("/api/chat/clear", response_model=ChatClearResponse)
def clear_chat(req: ChatClearRequest) -> ChatClearResponse:
    chain = deps.get_chain()
    chain.reset_session(req.session_id)
    logger.info("Reset chat session %r.", req.session_id)
    return ChatClearResponse(ok=True)
