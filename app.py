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
import doc_summaries

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

            # Generate a doc-level summary right after indexing. The chain
            # is already loaded (we just embedded), so this is one extra
            # short LLM call per PDF. Cached to disk; idempotent.
            try:
                chain = get_chain()
                full_text = result.markdown or "\n\n".join(
                    d.page_content for d in result.documents
                )
                chain.summarize_document(p.name, full_text)
            except Exception:
                logger.exception("Summary generation failed for %s", p.name)
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
                    doc_summaries.clear_all()
                    st.session_state.indexed_files = []
                paths = _save_uploaded_files(uploaded)
                st.write(f"Saved {len(paths)} file(s) to {UPLOAD_DIR}.")
                ok, chunks = _ingest_and_index(paths)
                status.update(
                    label=f"Indexed {ok}/{len(paths)} file(s) ({chunks} chunks). "
                          f"Doc-level summaries generated.",
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
                     help="Permanently delete every chunk + summary from the vector store."):
            store.reset()
            doc_summaries.clear_all()
            st.session_state.indexed_files = []
            st.success("Vector store wiped.")
            st.rerun()

    # Per-PDF scope selector. Without this, retrieval pools chunks from
    # every indexed doc and mixes facts across them ("cross-talk"). The
    # multiselect lets the user constrain each turn to specific files.
    indexed_files = _list_indexed_filenames()
    if indexed_files:
        if "scope" not in st.session_state:
            # Default: search across everything that's indexed.
            st.session_state.scope = list(indexed_files)
        else:
            # Drop any selections that no longer exist (file was unindexed).
            st.session_state.scope = [
                f for f in st.session_state.scope if f in indexed_files
            ]

        st.session_state.scope = st.multiselect(
            f"Search in ({len(indexed_files)} indexed)",
            options=indexed_files,
            default=st.session_state.scope,
            help="Restrict each query to specific PDFs. Leave empty or "
                 "select all to search across the whole vector store.",
        )
    else:
        st.session_state.scope = []


# ---------------------------------------------------------------------------
# Main pane — chat
# ---------------------------------------------------------------------------
st.title("\U0001F4DA Local PDF RAG Assistant")
st.caption("Offline RAG over your PDFs — Docling + ChromaDB + llama.cpp + LangChain.")

# Render past messages.
for i, msg in enumerate(st.session_state.messages):
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
        # Only the LAST assistant message gets clickable follow-up chips;
        # earlier suggestions are stale once the conversation moves on.
        if (
            msg["role"] == "assistant"
            and i == len(st.session_state.messages) - 1
            and msg.get("suggestions")
        ):
            st.caption("✨ Suggested follow-ups:")
            cols = st.columns(min(3, len(msg["suggestions"])))
            for j, q in enumerate(msg["suggestions"][:3]):
                if cols[j % len(cols)].button(
                    q, key=f"sugg_{i}_{j}", use_container_width=True,
                ):
                    # Re-inject as the next user message via session state.
                    st.session_state["__pending_prompt__"] = q
                    st.rerun()

# Tiny scope indicator so users understand what each query will search.
_indexed = _list_indexed_filenames()
_selected = st.session_state.get("scope") or []
if _indexed:
    if not _selected or set(_selected) == set(_indexed):
        st.caption(f"\U0001F50D Searching: **all {len(_indexed)} indexed document(s)**")
    elif len(_selected) == 1:
        st.caption(f"\U0001F50D Searching only: **{_selected[0]}**")
    else:
        st.caption(f"\U0001F50D Searching: **{len(_selected)} of {len(_indexed)} document(s)**")
else:
    st.caption("\U0001F4C2 No documents indexed yet — upload a PDF to start.")

# Input. A pending suggestion-chip click is replayed as if the user typed it.
prompt = st.chat_input("Ask a question about your PDFs...")
if not prompt and st.session_state.get("__pending_prompt__"):
    prompt = st.session_state.pop("__pending_prompt__")
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

        # Resolve scope: empty selection or all-selected both mean "search
        # everything" - we pass None in those cases so the underlying chain
        # knows it can skip the filename filter entirely.
        scope = st.session_state.get("scope") or []
        all_indexed = _list_indexed_filenames()
        scope_arg = None if (not scope or set(scope) == set(all_indexed)) else list(scope)

        # The chain.stream() generator yields tagged events:
        #   ('standalone', str), ('sources', List[Doc]), ('token', str), ('done', dict)
        # We feed only the 'token' events into st.write_stream for the
        # typewriter effect, while side-channel-capturing the rest.
        captures: dict = {}
        def _token_only():
            try:
                for kind, payload in chain.stream(
                    prompt,
                    session_id=st.session_state.session_id,
                    filenames=scope_arg,
                ):
                    if kind == "token":
                        yield payload
                    else:
                        captures[kind] = payload
            except Exception as exc:
                logger.exception("Streaming RAG failed")
                captures["error"] = str(exc)

        # Optional pre-stream hint while standalone+retrieval run.
        with st.spinner("Retrieving and generating..."):
            answer = st.write_stream(_token_only())

        if captures.get("error"):
            err = f"Error during RAG query: {captures['error']}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.stop()

        # The chain emits a final ('done', result) with everything reassembled.
        result = captures.get("done", {
            "answer": answer,
            "source_documents": captures.get("sources", []),
            "all_source_documents": captures.get("sources", []),
            "standalone_question": captures.get("standalone", prompt),
            "chitchat": False,
            "scope": scope_arg,
        })
        if not answer:
            answer = result.get("answer") or "_(empty response)_"

        # Cited sources (filtered by [filename p.X] in the answer) are the
        # default view; the full list is accessible via a checkbox below.
        cited_sources = result.get("source_documents", [])
        all_sources = result.get("all_source_documents", cited_sources)

        sources_payload = []
        if cited_sources:
            label = (
                f"Sources cited ({len(cited_sources)})"
                if len(cited_sources) < len(all_sources)
                else "Sources"
            )
            with st.expander(label):
                if len(cited_sources) < len(all_sources):
                    st.caption(
                        f"Showing only the {len(cited_sources)} excerpts the "
                        f"answer cites; {len(all_sources) - len(cited_sources)} "
                        f"additional retrieved excerpts were not cited."
                    )
                for d in cited_sources:
                    meta = d.metadata or {}
                    head = f"**{meta.get('filename', 'unknown')}** — p.{meta.get('page', '?')}"
                    if meta.get("headings"):
                        head += f" — {meta['headings']}"
                    snippet = d.page_content[:400] + ("..." if len(d.page_content) > 400 else "")
                    st.markdown(head)
                    st.markdown(f"> {snippet}")
                    sources_payload.append({"metadata": meta, "snippet": snippet})

        # Hide the standalone-query debug expander on chitchat turns.
        if not result.get("chitchat"):
            with st.expander("Standalone search query"):
                st.code(result.get("standalone_question", ""))

        # Suggested follow-up questions: 3 short, document-anchored prompts
        # the user might naturally ask next. Hidden during chitchat turns.
        suggestions: List[str] = []
        if not result.get("chitchat") and answer:
            try:
                suggestions = chain.suggest_followups(
                    last_question=prompt,
                    last_answer=answer,
                )
            except Exception:
                logger.exception("Failed to generate follow-up suggestions")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
            "suggestions": suggestions,
        })
