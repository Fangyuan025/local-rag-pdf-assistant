"""
Step 2: Local Vector Database Module.

Wraps a local HuggingFace embedding model (all-MiniLM-L6-v2) and a persistent
ChromaDB collection. Provides upsert + retrieval helpers for the RAG chain.

100% offline: the embedding model is cached locally on first download and the
vector store lives on disk under ./chroma_db.
"""
from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("vector_store")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_PERSIST_DIR = Path("./chroma_db")
DEFAULT_COLLECTION = "pdf_rag"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class VectorStoreConfig:
    persist_directory: Path = DEFAULT_PERSIST_DIR
    collection_name: str = DEFAULT_COLLECTION
    embedding_model_name: str = DEFAULT_EMBED_MODEL
    # Run on CPU by default to stay portable; switch to "cuda" if available.
    device: str = "cpu"
    normalize_embeddings: bool = True
    model_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vector store wrapper
# ---------------------------------------------------------------------------
class LocalVectorStore:
    """Thin wrapper over LangChain's Chroma + HuggingFace embeddings."""

    def __init__(self, config: Optional[VectorStoreConfig] = None) -> None:
        self.config = config or VectorStoreConfig()
        self.config.persist_directory = Path(self.config.persist_directory)
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        self._embeddings = self._init_embeddings()
        self._store = self._init_store()

    # --------------------------------------------------------------- builders
    @staticmethod
    def _warmup_torch_for_hf() -> None:
        """
        Workaround for a Windows-only segfault in transformers 5.6.x +
        huggingface_hub 1.12.x where cold-loading a sentence-transformers
        model crashes during the first network/cache resolution.
        Instantiating Docling's DocumentConverter first runs the same
        torch/transformers init path safely and "warms" the process so
        the subsequent embedding-model load succeeds. Effect persists for
        the lifetime of the process. Cheap (a few hundred ms) and harmless
        if Docling is not installed.
        """
        try:
            from docling.document_converter import DocumentConverter
            DocumentConverter()
        except Exception as exc:  # pragma: no cover
            logger.debug("Docling warm-up skipped: %s", exc)

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        try:
            self._warmup_torch_for_hf()
            logger.info(
                "Loading local embedding model: %s (device=%s)",
                self.config.embedding_model_name,
                self.config.device,
            )
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs={"device": self.config.device, **self.config.model_kwargs},
                encode_kwargs={"normalize_embeddings": self.config.normalize_embeddings},
            )
        except Exception as exc:
            logger.exception("Failed to load embedding model.")
            raise RuntimeError(
                f"Could not initialize embeddings '{self.config.embedding_model_name}'."
            ) from exc

    def _init_store(self) -> Chroma:
        try:
            logger.info(
                "Opening Chroma collection '%s' at %s",
                self.config.collection_name,
                self.config.persist_directory,
            )
            return Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self._embeddings,
                persist_directory=str(self.config.persist_directory),
            )
        except Exception as exc:
            logger.exception("Failed to open ChromaDB.")
            raise RuntimeError(
                f"Could not open Chroma at {self.config.persist_directory}."
            ) from exc

    # ----------------------------------------------------------------- upsert
    @staticmethod
    def _doc_id(doc: Document) -> str:
        """
        Deterministic ID = hash(source + chunk_index + content). Keeps re-ingest
        idempotent: re-uploading the same PDF overwrites instead of duplicating.
        """
        source = str(doc.metadata.get("source", ""))
        idx = str(doc.metadata.get("chunk_index", ""))
        h = hashlib.sha256()
        h.update(source.encode("utf-8"))
        h.update(idx.encode("utf-8"))
        h.update(doc.page_content.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _sanitize_metadata(doc: Document) -> Document:
        """
        Chroma only accepts str/int/float/bool/None metadata values. Lists
        (like our `pages` field) get joined into comma-separated strings.
        """
        clean = {}
        for k, v in (doc.metadata or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = ",".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return Document(page_content=doc.page_content, metadata=clean)

    def add_documents(self, documents: Sequence[Document]) -> List[str]:
        if not documents:
            logger.warning("add_documents called with empty input.")
            return []

        clean_docs = [self._sanitize_metadata(d) for d in documents]
        ids = [self._doc_id(d) for d in clean_docs]

        try:
            self._store.add_documents(documents=clean_docs, ids=ids)
        except Exception as exc:
            logger.exception("Chroma upsert failed.")
            raise RuntimeError("Failed to upsert documents into Chroma.") from exc

        logger.info("Upserted %d chunks into Chroma.", len(clean_docs))
        return ids

    # ---------------------------------------------------------- introspection
    def count(self) -> int:
        try:
            return self._store._collection.count()  # noqa: SLF001
        except Exception:
            return -1

    def reset(self) -> None:
        """Delete the entire collection. Useful for tests / re-ingest flows."""
        try:
            self._store.delete_collection()
            logger.warning("Chroma collection '%s' deleted.", self.config.collection_name)
        finally:
            self._store = self._init_store()

    # -------------------------------------------------------------- retrieval
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        try:
            return self._store.similarity_search(query, k=k, filter=filter)
        except Exception as exc:
            logger.exception("Similarity search failed for query: %s", query)
            raise RuntimeError("Vector search failed.") from exc

    def similarity_search_with_scores(
        self, query: str, k: int = 4, filter: Optional[dict] = None
    ) -> List[tuple[Document, float]]:
        try:
            return self._store.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as exc:
            logger.exception("Scored similarity search failed.")
            raise RuntimeError("Vector search failed.") from exc

    def as_retriever(self, k: int = 4, **kwargs) -> BaseRetriever:
        """Return a LangChain BaseRetriever for use in chains."""
        search_kwargs = {"k": k}
        search_kwargs.update(kwargs.pop("search_kwargs", {}))
        return self._store.as_retriever(search_kwargs=search_kwargs, **kwargs)


# ---------------------------------------------------------------------------
# Convenience top-level helpers
# ---------------------------------------------------------------------------
def build_default_store() -> LocalVectorStore:
    return LocalVectorStore(VectorStoreConfig())


def index_documents(
    documents: Iterable[Document],
    store: Optional[LocalVectorStore] = None,
) -> LocalVectorStore:
    """Embed a stream of LangChain Documents into the local vector store."""
    store = store or build_default_store()
    docs = list(documents)
    store.add_documents(docs)
    logger.info("Vector store now holds approx %d chunks.", store.count())
    return store


# ---------------------------------------------------------------------------
# CLI: ingest a directory and index it in one shot.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    from ingest import PDFIngestor, DEFAULT_DATA_DIR

    parser = argparse.ArgumentParser(
        description="Index PDFs into the local Chroma vector store."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="PDF files to index. Defaults to ./data/*.pdf.",
    )
    parser.add_argument("--reset", action="store_true", help="Wipe collection first.")
    args = parser.parse_args(argv)

    ingestor = PDFIngestor()
    if args.paths:
        results = ingestor.ingest_many(args.paths)
    else:
        results = ingestor.ingest_directory(DEFAULT_DATA_DIR)

    store = build_default_store()
    if args.reset:
        store.reset()

    all_docs: List[Document] = []
    for r in results:
        all_docs.extend(r.documents)
    index_documents(all_docs, store=store)

    print(f"Indexed {len(all_docs)} chunks. Collection size: {store.count()}.")


if __name__ == "__main__":
    main()
