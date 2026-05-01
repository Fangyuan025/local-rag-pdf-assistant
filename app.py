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

import base64
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List

import streamlit as st
import streamlit.components.v1 as components

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
def _autoplay_audio(audio_bytes: bytes, key: str) -> None:
    """Play `audio_bytes` immediately on render with NO visible UI.

    Uses a hidden HTML5 `<audio autoplay>` element instead of `st.audio`
    so the user doesn't see Streamlit's full-width progress bar after
    every answer. The `key` query-string forces the browser to treat
    each render as a fresh element so replay clicks restart playback
    even if the same bytes are re-rendered.
    """
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    components.html(
        f"""
        <audio id="hushdoc-{key}" autoplay style="display:none"
               src="data:audio/wav;base64,{b64}#k={key}"></audio>
        """,
        height=0,
    )


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
    # ── Documents
    st.subheader("\U0001F4C2 Documents")

    # Docling natively handles both PDFs and document images: image inputs
    # go through the OCR pipeline (RapidOCR), so a phone snap of a page
    # gets indexed alongside real PDFs without any extra config.
    uploaded = st.file_uploader(
        "Upload PDFs or document photos",
        type=["pdf", "jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="PDFs are layout-parsed; JPG/PNG/TIFF photos go through OCR.",
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
        clear_chat_clicked = st.button("Clear Chat", use_container_width=True)

    if ingest_clicked and uploaded:
        with st.status(
            "Parsing PDFs with Docling and indexing into Chroma...",
            expanded=True,
        ) as status:
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
                    label=f"Indexed {ok}/{len(paths)} file(s) "
                          f"({chunks} chunks, summaries generated).",
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
        st.toast("Chat cleared.", icon="🧹")

    try:
        store = get_vector_store()
        store_count = store.count()
    except Exception:
        store = None
        store_count = -1
    st.caption(f"Vector store: **{store_count}** chunks")

    # Wipe-everything button — separate from "Clear Chat" to avoid accidents.
    if store is not None and store_count > 0:
        if st.button(
            "\U0001F5D1 Clear all documents",
            use_container_width=True,
            help="Permanently delete every chunk + summary from the vector store.",
        ):
            store.reset()
            doc_summaries.clear_all()
            st.session_state.indexed_files = []
            st.toast("Vector store wiped.", icon="🗑")
            st.rerun()

    # ── Search scope (only shown when there's something to scope to)
    indexed_files = _list_indexed_filenames()
    if indexed_files:
        st.subheader("\U0001F50D Search scope")
        if "scope" not in st.session_state:
            st.session_state.scope = list(indexed_files)
        else:
            # Drop any selections that no longer exist (file was unindexed).
            st.session_state.scope = [
                f for f in st.session_state.scope if f in indexed_files
            ]

        st.session_state.scope = st.multiselect(
            f"In {len(indexed_files)} indexed PDF(s)",
            options=indexed_files,
            default=st.session_state.scope,
            help="Restrict each query to specific PDFs. Leave empty or select "
                 "all to search across the whole vector store.",
            label_visibility="collapsed",
        )
    else:
        st.session_state.scope = []

    # ── Voice mode (default OFF)
    st.subheader("\U0001F3A4 Voice")
    voice_mode = st.toggle(
        "Voice mode",
        value=st.session_state.get("voice_mode", False),
        help=(
            "Speak your question with the inline mic button. The answer "
            "auto-plays after streaming and a 🔊 replay icon stays beside it."
        ),
    )
    st.session_state.voice_mode = voice_mode
    if voice_mode:
        st.caption(
            "\U0001F310 Voice features are English-only for now "
            "(Whisper-base.en in, Kokoro-82M out)."
        )


# ---------------------------------------------------------------------------
# Main pane — chat
# ---------------------------------------------------------------------------
st.title("\U0001F910 Hushdoc")
st.caption(
    "A local-only PDF assistant that keeps every word between you and your machine."
)

# Render past messages. Each assistant message that has cached TTS audio
# gets a 🔊 replay icon next to its source-citation row.
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(msg.get("sources_label") or "Sources"):
                for s in msg["sources"]:
                    meta = s.get("metadata", {})
                    label = f"**{meta.get('filename', 'unknown')}** — p.{meta.get('page', '?')}"
                    if meta.get("headings"):
                        label += f" — {meta['headings']}"
                    st.markdown(label)
                    st.markdown(f"> {s.get('snippet', '')}")
        if msg.get("audio_b64"):
            if st.button("🔊", key=f"replay_{i}", help="Replay audio"):
                st.session_state["__pending_audio_b64__"] = msg["audio_b64"]
                st.session_state["__pending_audio_key__"] = f"replay_{i}_{uuid.uuid4().hex[:6]}"
                st.rerun()

# Scope indicator + (when voice mode is on) inline mic. Both sit
# immediately above the chat input so the mic visually belongs to it.
_indexed = _list_indexed_filenames()
_selected = st.session_state.get("scope") or []
if _indexed:
    if not _selected or set(_selected) == set(_indexed):
        scope_msg = f"\U0001F50D Searching: **all {len(_indexed)} indexed document(s)**"
    elif len(_selected) == 1:
        scope_msg = f"\U0001F50D Searching only: **{_selected[0]}**"
    else:
        scope_msg = f"\U0001F50D Searching: **{len(_selected)} of {len(_indexed)} document(s)**"
else:
    scope_msg = "\U0001F4C2 No documents indexed yet — upload a PDF to start."

if st.session_state.get("voice_mode"):
    # Mic icon sits in a narrow column on the left; the scope caption
    # fills the rest of the row so the mic visually anchors to the
    # chat input directly below.
    from audio_recorder_streamlit import audio_recorder
    col_mic, col_msg = st.columns([1, 11])
    with col_mic:
        audio_bytes = audio_recorder(
            text="",
            icon_name="microphone",
            icon_size="2x",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            pause_threshold=2.0,   # auto-stop after 2 s of silence
            sample_rate=16000,
            key="hushdoc_mic",
        )
    with col_msg:
        st.caption(scope_msg + " — speak then pause to send.")
    if audio_bytes:
        # audio_recorder returns the same bytes across reruns until the
        # user records again. Hash them to fire transcription only once.
        import hashlib
        audio_id = hashlib.sha1(audio_bytes).hexdigest()
        if st.session_state.get("__last_audio_id__") != audio_id:
            try:
                from voice import transcribe
                with st.spinner("Transcribing..."):
                    text = transcribe(audio_bytes)
                st.session_state["__last_audio_id__"] = audio_id
                if text:
                    st.session_state["__pending_prompt__"] = text
                    st.rerun()
                else:
                    st.warning("Could not transcribe — try recording again.")
            except Exception as exc:
                logger.exception("Voice input failed")
                st.error(f"Voice input failed: {exc}")
else:
    st.caption(scope_msg)

# Input. A pending voice-transcribed prompt is replayed as if typed.
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
        sources_label = "Sources"
        if cited_sources:
            sources_label = (
                f"Sources cited ({len(cited_sources)})"
                if len(cited_sources) < len(all_sources)
                else "Sources"
            )
            with st.expander(sources_label):
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
                # Standalone-query debug info: folded inside Sources rather
                # than a second top-level expander, to keep each turn tidy.
                if not result.get("chitchat"):
                    st.divider()
                    st.caption("Standalone search query (debug):")
                    st.code(
                        result.get("standalone_question", ""),
                        language=None,
                    )

        # Voice output: synthesize once after streaming completes. We use
        # an invisible HTML5 <audio autoplay> via components.html instead
        # of st.audio so the user doesn't see Streamlit's full-width
        # progress bar after every answer. The bytes are cached on the
        # message so a 🔊 replay icon next to the answer can re-play the
        # same audio without re-running the model.
        audio_b64: str = ""
        if st.session_state.get("voice_mode") and answer:
            from llm_chain import detect_language
            if detect_language(answer) == "en":
                try:
                    from voice import synthesize
                    with st.spinner("\U0001F50A Generating audio..."):
                        audio_bytes = synthesize(answer)
                    if audio_bytes:
                        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                        # Autoplay invisibly this turn.
                        _autoplay_audio(audio_bytes, key=f"new_{uuid.uuid4().hex[:6]}")
                        # Persistent replay icon for the just-rendered message
                        # (the message-loop render of the same turn on the
                        # next rerun will show another one keyed by index).
                        if st.button(
                            "🔊", key=f"replay_live_{uuid.uuid4().hex[:6]}",
                            help="Replay audio",
                        ):
                            st.session_state["__pending_audio_b64__"] = audio_b64
                            st.session_state["__pending_audio_key__"] = (
                                f"replay_live_{uuid.uuid4().hex[:6]}"
                            )
                            st.rerun()
                except Exception:
                    logger.exception("Voice output failed")
            else:
                st.caption(
                    "\U0001F507 Voice output skipped — Kokoro-82M is English-only."
                )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_payload,
            "sources_label": sources_label,
            "audio_b64": audio_b64,
        })


# Handle a 🔊 replay click queued from the message-render loop. Renders
# the invisible autoplay HTML one final time at end of script.
_pending_b64 = st.session_state.pop("__pending_audio_b64__", None)
_pending_key = st.session_state.pop("__pending_audio_key__", None)
if _pending_b64 and _pending_key:
    try:
        _autoplay_audio(base64.b64decode(_pending_b64), key=_pending_key)
    except Exception:
        logger.exception("Replay autoplay failed")
