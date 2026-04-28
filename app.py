"""
Step 5: Streamlit Frontend.

A clean chat UI for the local RAG assistant:
  - Sidebar: upload PDFs and trigger ingest -> embed -> index pipeline.
  - Main: stateful chat with the RAG chain (history is kept across turns).

Everything runs locally. No cloud calls.
"""
from __future__ import annotations

import os
# Avoid the OpenMP duplicate-library segfault on Windows when llama-cpp-python
# and PyTorch coexist in the same process. MUST run before any heavy imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import streamlit as st

from ingest import PDFIngestor
from vector_store import LocalVectorStore, build_default_store
from llm_chain import RAGChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("app")


# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Local PDF RAG Assistant",
    page_icon="\U0001F4DA",
    layout="wide",
)

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cached resources (loaded once per process)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading local embedding model & vector store...")
def get_vector_store() -> LocalVectorStore:
    return build_default_store()


@st.cache_resource(show_spinner="Loading local LLM (this can take a moment)...")
def get_chain() -> RAGChain:
    store = get_vector_store()
    return RAGChain(vector_store=store)


@st.cache_resource(show_spinner=False)
def get_ingestor() -> PDFIngestor:
    return PDFIngestor()


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_uploaded_files(uploaded_files) -> List[Path]:
    paths: List[Path] = []
    for f in uploaded_files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as out:
            shutil.copyfileobj(f, out)
        paths.append(dest)
    return paths


def _list_indexed_filenames() -> List[str]:
    """Pull the unique `filename` values out of the Chroma collection so the
    UI can show what's currently searchable - independent of session state."""
    try:
        store = get_vector_store()
        # Chroma's collection.get() with no ids returns metadatas for all docs.
        data = store._store._collection.get(include=["metadatas"])  # noqa: SLF001
        names = sorted({(m or {}).get("filename", "?") for m in data.get("metadatas", [])})
        return [n for n in names if n and n != "?"]
    except Exception:
        return []


def _ingest_and_index(paths: List[Path]) -> tuple[int, int]:
    ingestor = get_ingestor()
    store = get_vector_store()

    total_chunks = 0
    succeeded = 0
    for p in paths:
        try:
            result = ingestor.ingest(p)
            store.add_documents(result.documents)
            total_chunks += result.chunk_count
            succeeded += 1
            st.session_state.indexed_files.append(p.name)
        except Exception as exc:
            logger.exception("Failed to index %s", p)
            st.error(f"Failed to index {p.name}: {exc}")
    return succeeded, total_chunks


# ---------------------------------------------------------------------------
# Sidebar — ingestion controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("\U0001F4C2 Documents")

    uploaded = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    replace_index = st.checkbox(
        "Replace existing index on upload",
        value=True,
        help="When checked, ingestion wipes the vector store first so "
             "queries only see the newly uploaded PDFs. Uncheck to add "
             "new PDFs alongside what's already indexed.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        ingest_clicked = st.button(
            "Ingest & Index",
            type="primary",
            disabled=not uploaded,
            use_container_width=True,
        )
    with col_b:
        clear_chat_clicked = st.button(
            "Clear Chat",
            use_container_width=True,
        )

    if ingest_clicked and uploaded:
        with st.status("Parsing PDFs with Docling and indexing into Chroma...", expanded=True) as status:
            try:
                if replace_index:
                    st.write("Wiping existing vector store...")
                    get_vector_store().reset()
                    st.session_state.indexed_files = []
                paths = _save_uploaded_files(uploaded)
                st.write(f"Saved {len(paths)} file(s) to {UPLOAD_DIR}.")
                ok, chunks = _ingest_and_index(paths)
                status.update(
                    label=f"Indexed {ok}/{len(paths)} file(s) ({chunks} chunks).",
                    state="complete",
                )
            except Exception as exc:
                status.update(label=f"Ingestion failed: {exc}", state="error")
                logger.exception("Ingestion error")

    if clear_chat_clicked:
        chain = get_chain()
        chain.reset_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Chat cleared.")

    st.divider()

    try:
        store = get_vector_store()
        store_count = store.count()
    except Exception:
        store = None
        store_count = -1
    st.caption(f"Vector store size: **{store_count}** chunks")

    # Wipe-everything button — separate from "Clear Chat" to avoid accidents.
    if store is not None and store_count > 0:
        if st.button("\U0001F5D1 Clear all documents", use_container_width=True,
                     help="Permanently delete every chunk from the vector store."):
            store.reset()
            st.session_state.indexed_files = []
            st.success("Vector store wiped.")
            st.rerun()

    # Show what's actually indexed (across all sessions, read from Chroma).
    indexed_files = _list_indexed_filenames()
    if indexed_files:
        with st.expander(f"Indexed documents ({len(indexed_files)})", expanded=False):
            for name in indexed_files:
                st.write(f"- {name}")


# ---------------------------------------------------------------------------
# Main pane — chat
# ---------------------------------------------------------------------------
st.title("\U0001F4DA Local PDF RAG Assistant")
st.caption("Offline RAG over your PDFs — Docling + ChromaDB + llama.cpp + LangChain.")

# Render past messages.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    meta = s.get("metadata", {})
                    label = f"**{meta.get('filename', 'unknown')}** — p.{meta.get('page', '?')}"
                    if meta.get("headings"):
                        label += f" — {meta['headings']}"
                    st.markdown(label)
                    st.markdown(f"> {s.get('snippet', '')}")

# Input.
prompt = st.chat_input("Ask a question about your PDFs...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            chain = get_chain()
        except Exception as exc:
            err = f"Failed to load local LLM: {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

        with st.spinner("Thinking locally..."):
            try:
                result = chain.ask(prompt, session_id=st.session_state.session_id)
            except Exception as exc:
                err = f"Error during RAG query: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                st.stop()

        answer = result["answer"] or "_(empty response)_"
        st.markdown(answer)

        sources_payload = []
        if result["source_documents"]:
            with st.expander("Sources"):
                for d in result["source_documents"]:
                    meta = d.metadata or {}
                    label = f"**{meta.get('filename', 'unknown')}** — p.{meta.get('page', '?')}"
                    if meta.get("headings"):
                        label += f" — {meta['headings']}"
                    snippet = d.page_content[:400] + ("..." if len(d.page_content) > 400 else "")
                    st.markdown(label)
                    st.markdown(f"> {snippet}")
                    sources_payload.append({"metadata": meta, "snippet": snippet})

        # Hide the standalone-query debug expander on chitchat turns - the
        # rewriter never ran and the field is just an echo of the user's
        # message, which would be confusing to display.
        if not result.get("chitchat"):
            with st.expander("Standalone search query"):
                st.code(result.get("standalone_question", ""))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources_payload}
        )
