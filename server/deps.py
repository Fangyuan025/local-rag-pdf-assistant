"""
Lazy singletons for the heavy backend objects.

The vector store, ingestor, and RAG chain are each created on first
access. ``/api/health`` peeks via the ``*_if_loaded`` accessors so it
stays cheap and never triggers a 30-second cold-start of llama-server.
Endpoints that actually need the chain (chat, summarize) call the
plain ``get_*`` accessors and pay the load cost on first hit.
"""
from __future__ import annotations

import logging
from typing import Optional

from ingest import PDFIngestor
from llm_chain import RAGChain
from vector_store import LocalVectorStore, build_default_store

logger = logging.getLogger("server.deps")

# Module-level singletons. None until first access.
_store: Optional[LocalVectorStore] = None
_ingestor: Optional[PDFIngestor] = None
_chain: Optional[RAGChain] = None


# ---------------------------------------------------------------------------
# Eager accessors (load on first call)
# ---------------------------------------------------------------------------
def get_store() -> LocalVectorStore:
    global _store
    if _store is None:
        logger.info("Loading vector store + embedding model...")
        _store = build_default_store()
    return _store


def get_ingestor() -> PDFIngestor:
    global _ingestor
    if _ingestor is None:
        logger.info("Initialising Docling ingestor...")
        _ingestor = PDFIngestor()
    return _ingestor


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        logger.info("Spinning up RAGChain (will start llama-server on first ask)...")
        _chain = RAGChain(vector_store=get_store(), k=6)
    return _chain


# ---------------------------------------------------------------------------
# Peek accessors (return current singleton without triggering load)
# ---------------------------------------------------------------------------
def get_store_if_loaded() -> Optional[LocalVectorStore]:
    return _store


def get_ingestor_if_loaded() -> Optional[PDFIngestor]:
    return _ingestor


def get_chain_if_loaded() -> Optional[RAGChain]:
    return _chain
