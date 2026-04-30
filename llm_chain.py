"""
Step 3: Local LLM Engine, Conversational Memory, and RAG Chain.

- Loads a local GGUF model via llama-cpp-python (no network calls).
- Maintains chat history with LangChain's RunnableWithMessageHistory.
- Implements a "Standalone Query Generator" that rewrites the latest user
  question into a context-independent search query using the chat history.
- Retrieves relevant chunks from the local Chroma store and grounds the
  final answer on them.
"""
from __future__ import annotations

import os
# llama-cpp-python and PyTorch (via sentence-transformers) both ship their own
# OpenMP runtimes on Windows. Loading both into one process triggers a duplicate
# OpenMP runtime check that segfaults. This flag tells Intel OpenMP to tolerate
# it. MUST be set BEFORE numpy/torch/llama_cpp are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from vector_store import LocalVectorStore, build_default_store
from llama_server import (
    LlamaServer,
    ServerConfig,
    get_shared_server,
    DEFAULT_MODEL_PATH as SERVER_DEFAULT_MODEL_PATH,
)
import doc_summaries

logger = logging.getLogger("llm_chain")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = SERVER_DEFAULT_MODEL_PATH


@dataclass
class LLMConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    # Qwen3 1.7B supports up to 40K. n_ctx is TOTAL across llama-server
    # slots; per-slot ctx = n_ctx / parallel. With server parallel=4 the
    # 32768 here gives 8192 per slot, plenty for ragas judge prompts.
    # KV cache lives on the GPU; on a 4GB card the model + 32K KV ≈ 2-3GB.
    n_ctx: int = 32768
    # -1 = offload ALL layers to GPU (requires CUDA-enabled llama-server).
    # Set to 0 for CPU-only.
    n_gpu_layers: int = -1
    temperature: float = 0.2
    # Generous so answers + ragas judge JSON don't get truncated.
    max_tokens: int = 2048
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    # llama-server HTTP endpoint config (port etc.) is in ServerConfig.
    server_config: Optional[ServerConfig] = None
    extra_kwargs: dict = field(default_factory=dict)


# Qwen3 chat-template directive that disables the <think> reasoning block.
# Append this to a user message and the model will answer directly.
# Reference: Qwen3 uses /think and /no_think soft switches in the prompt.
QWEN3_NO_THINK = "/no_think"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Putting /no_think at the very START of the system message is the most
# reliable way to disable Qwen3's <think> block - putting it inside or at
# the end of the user message can cause the model to echo it back as content.
CONDENSE_QUESTION_SYSTEM = (
    QWEN3_NO_THINK + "\n"
    "You rewrite the user's latest message into a single self-contained "
    "search query that captures what they actually want to look up.\n\n"
    "Rules:\n"
    "- Resolve every pronoun, ellipsis, and implicit reference using the "
    "  chat history. Short follow-ups like 'why?', 'and?', 'tell me more', "
    "  '为什么', '继续', '再说说' MUST be expanded using what was just "
    "  discussed.\n"
    "- Output ONLY the rewritten query in plain prose (no quotes, no "
    "  slashes, no special tokens, no preamble like 'Standalone query:').\n"
    "- If the latest message is already self-contained, return it UNCHANGED.\n"
    "- Keep the rewrite under 25 words.\n\n"
    "Examples:\n"
    "  History: 'Q: What is the Transformer architecture? A: It is a model "
    "  that uses self-attention...'\n"
    "  Latest: '为什么？'\n"
    "  Output: Why does the Transformer architecture use self-attention "
    "  instead of recurrence?\n\n"
    "  History: (empty)\n"
    "  Latest: 'How does multi-head attention work?'\n"
    "  Output: How does multi-head attention work?"
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONDENSE_QUESTION_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "Latest message: {question}\n\nRewritten query:"),
    ]
)

ANSWER_SYSTEM = (
    QWEN3_NO_THINK + "\n"
    "You are a precise assistant answering questions about PDF documents.\n\n"
    "LANGUAGE: Reply in the SAME natural language as the user's latest "
    "question. If the user writes in Chinese, answer in Chinese; if in "
    "English, answer in English. Do not switch languages mid-answer.\n\n"
    "Follow these rules:\n"
    "1. Use ONLY the facts contained in the provided context (document "
    "   summaries + retrieved excerpts). If the answer is not present in "
    "   the context, reply exactly: 'I don't know based on the provided "
    "   documents.' (or its Chinese equivalent: '根据提供的文档我无法回答。')\n"
    "   Do NOT use outside knowledge. Do NOT guess author names, dates, or "
    "   numbers that are not literally present in the context. If you "
    "   mention a dataset / model / number / year, it MUST appear verbatim "
    "   in the context - otherwise omit it.\n"
    "2. The context block starts with a 'Documents in scope' summary list "
    "   (one line per file describing what that document is about), "
    "   followed by retrieved excerpts each prefixed with "
    "   '--- From <filename> (page <n>) ---'. Use the summaries to answer "
    "   high-level questions ('which one is about X', 'what is the topic', "
    "   'summarize this paper'); use the excerpts for specific details.\n"
    "3. For comparative or cross-document questions, attribute every claim "
    "   to the file it actually came from. Do NOT invent commonalities. "
    "   Do NOT mix facts across files. If the documents have nothing "
    "   meaningful in common, say so plainly.\n"
    "4. Synthesize ONE coherent prose answer. Do NOT enumerate context "
    "   excerpts one-by-one - the user wants the conclusion, not a "
    "   chunk-by-chunk commentary.\n"
    "5. When the context contains tables, code, or formulas, preserve their "
    "   structure verbatim. For LaTeX formulas in the context, keep them in "
    "   LaTeX form in your answer.\n"
    "6. Cite sources inline as [filename p.<page>] using the page numbers "
    "   shown in each excerpt header. Only cite excerpts you actually used.\n"
    "7. Be concise. Do not repeat yourself. Do not fabricate quotes."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Context:\n----------\n{context}\n----------\n\n"
            "Question: {question}\n\nAnswer:",
        ),
    ]
)

# Suggestion generator — produces 3 short follow-up questions a user might
# naturally ask next, given the previous Q+A and the documents in scope.
SUGGESTIONS_SYSTEM = (
    QWEN3_NO_THINK + "\n"
    "Suggest 3 short follow-up questions the user might naturally ask next, "
    "based on the most recent Q+A and the documents available. "
    "Output each question on its own line, no numbering, no bullets, no "
    "preamble. Each question must be answerable from the documents the "
    "user is already working with. Match the language the user is using "
    "(Chinese for Chinese conversations, English for English). "
    "Each question should be under 15 words."
)

SUGGESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUGGESTIONS_SYSTEM),
        (
            "human",
            "Documents available:\n{documents_overview}\n\n"
            "Most recent Q: {question}\n"
            "Most recent A: {answer}\n\n"
            "Three follow-up questions:",
        ),
    ]
)

# Conversational fallback prompt for greetings / meta-questions / chitchat
# where doing a vector search would be silly. Reply in the user's language.
CHITCHAT_SYSTEM = (
    QWEN3_NO_THINK + "\n"
    "Reply in the SAME language as the user's message (Chinese for Chinese, "
    "English for English).\n"
    "You are a friendly assistant for a local PDF question-answering app. "
    "The user has just sent a greeting, an introduction, a thank-you, or a "
    "meta-question about your capabilities - NOT a question about a specific "
    "document. Reply briefly and naturally in the same language as the user. "
    "If appropriate, mention that you can answer questions about PDFs they "
    "upload. Do NOT refuse, do NOT say 'I don't know based on the provided "
    "documents'. Keep it under 3 sentences."
)

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CHITCHAT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Chitchat detector — short messages that match a greeting / thanks / meta
# pattern in CN or EN, and shouldn't trigger document retrieval.
# ---------------------------------------------------------------------------
# Two pattern groups:
#   STRICT  — unambiguously chitchat (greetings, thanks, farewells, identity
#             questions). These short-circuit retrieval regardless of history.
#   META    — ambiguous on their own ('why?', 'really?', 'how?'). These are
#             chitchat ONLY when there's no chat history; with history they
#             are follow-ups that need retrieval-with-rewrite.
_CHITCHAT_STRICT_PATTERNS = [
    # English greetings / thanks / farewells
    r"^\s*(hi|hello|hey|yo|sup|good\s+(morning|afternoon|evening|night))\b",
    r"^\s*(thanks|thank\s+you|thx|cheers|bye|goodbye|see\s+you|nice\s+to\s+meet)\b",
    r"^\s*(how\s+are\s+you|how's\s+it\s+going|what'?s?\s+up|how\s+do\s+you\s+do)\b",
    # English identity / capability questions
    r"^\s*(who\s+are\s+you|what\s+(are\s+you|can\s+you\s+do|do\s+you\s+do)|what\s+is\s+this)\b",
    r"^\s*(introduce\s+yourself|tell\s+me\s+about\s+yourself|please\s+introduce)\b",
    r"^\s*(help|what\s+(should|can)\s+i\s+(do|ask)|how\s+do\s+i\s+use\s+(this|you))\b",
    # Chinese greetings / thanks / farewells
    r"^\s*(你好|您好|嗨|哈喽|早上?好|中午好|下午好|晚上好|晚安)",
    r"^\s*(谢谢|多谢|感谢|拜拜|再见|回头见|辛苦了)",
    r"^\s*(在吗|你在吗|忙吗|最近怎么样)",
    # Chinese identity / capability questions
    r"^\s*(你是谁|你叫什么|你能(做|干)什么|你是什么|介绍(一?下)?(你?自己|你))",
    r"^\s*(你会(做)?什么|你能帮(我|忙)|帮助|怎么(用|玩))",
    r"(请|麻烦)?介绍(一?下)?(你?自己|你)",
]
_CHITCHAT_STRICT_RE = re.compile("|".join(_CHITCHAT_STRICT_PATTERNS), re.IGNORECASE)


def is_chitchat(text: str, has_history: bool = False) -> bool:
    """Return True if the message looks like a greeting / chitchat / meta-q
    that should bypass document retrieval.

    Parameters
    ----------
    text : str
        The user's message.
    has_history : bool
        Whether there's any prior chat history. Affects ambiguous short
        utterances: a bare 'why?' / '为什么' counts as chitchat only when
        there is no history; with history it's a follow-up that needs
        retrieval-with-rewrite, NOT a fresh greeting.
    """
    if not text:
        return False
    t = text.strip()
    # Long messages are very unlikely to be pure chitchat - assume they're
    # real questions even if they happen to start with a greeting word.
    if len(t) > 40:
        return False
    return bool(_CHITCHAT_STRICT_RE.search(t))


# ---------------------------------------------------------------------------
# Streaming <think>...</think> filter for reasoning models.
# ---------------------------------------------------------------------------
class _ThinkStripFilter:
    """
    Incremental filter that swallows the first ``<think>...</think>`` block
    of a streamed model output. Used to keep CoT tokens out of the user's
    chat bubble while still streaming the actual answer.

    Handles partial tags split across chunk boundaries (e.g. one chunk ends
    with ``<thi`` and the next starts with ``nk>``) by buffering up to
    ``len('</think>') - 1`` characters at the tail.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._post_think = False  # already passed through one think block

    def feed(self, chunk: str) -> str:
        """Consume the next streamed chunk, return text safe to display."""
        if self._post_think:
            return chunk

        self._buf += chunk
        out = ""
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self.CLOSE)
                if idx < 0:
                    # Keep just enough tail in case </think> straddles.
                    keep = max(0, len(self._buf) - (len(self.CLOSE) - 1))
                    self._buf = self._buf[keep:]
                    return out
                self._buf = self._buf[idx + len(self.CLOSE):]
                self._in_think = False
                self._post_think = True
                out += self._buf
                self._buf = ""
                return out
            else:
                idx = self._buf.find(self.OPEN)
                if idx < 0:
                    # No opener seen yet. Emit everything except a possible
                    # partial-prefix tail like "<thin".
                    keep = max(0, len(self._buf) - (len(self.OPEN) - 1))
                    out += self._buf[:keep]
                    self._buf = self._buf[keep:]
                    return out
                # Emit pre-tag text, then enter think state.
                out += self._buf[:idx]
                self._buf = self._buf[idx + len(self.OPEN):]
                self._in_think = True
        return out

    def flush(self) -> str:
        """Emit anything still buffered when the stream ends."""
        if self._in_think:
            return ""  # truncated mid-think → drop
        out, self._buf = self._buf, ""
        return out


# ---------------------------------------------------------------------------
# LLM loader
# ---------------------------------------------------------------------------
def load_local_llm(config: Optional[LLMConfig] = None) -> ChatOpenAI:
    """
    Start (or reuse) the local llama-server process and return a langchain
    ChatOpenAI pointed at its OpenAI-compatible endpoint.

    Why a server, not in-process bindings? The Windows CUDA wheels for
    llama-cpp-python are stuck at v0.3.4 which doesn't know the qwen3 GGUF
    architecture. The standalone llama-server.exe from upstream llama.cpp
    is current, GPU-enabled, and OpenAI-compatible.
    """
    cfg = config or LLMConfig()
    model_path = Path(cfg.model_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model not found at: {model_path}\n"
            "Place a quantized .gguf file at ./models/model.gguf or set the "
            "LLAMA_MODEL_PATH environment variable."
        )

    # Build the server config and start the subprocess.
    server_cfg = cfg.server_config or ServerConfig(
        model_path=model_path,
        n_ctx=cfg.n_ctx,
        n_gpu_layers=cfg.n_gpu_layers,
    )
    try:
        server = get_shared_server(server_cfg)
    except Exception as exc:
        logger.exception("Failed to start llama-server.")
        raise RuntimeError(f"Could not start llama-server: {exc}") from exc

    try:
        logger.info(
            "Connecting ChatOpenAI client to llama-server at %s "
            "(n_ctx=%d, n_gpu_layers=%d)",
            server_cfg.openai_base_url,
            cfg.n_ctx,
            cfg.n_gpu_layers,
        )
        # llama-server doesn't enforce auth; ChatOpenAI requires *some* key.
        llm = ChatOpenAI(
            base_url=server_cfg.openai_base_url,
            api_key="not-needed",
            # The model name is whatever llama-server is configured with - it
            # ignores the field but langchain validates non-emptiness.
            model="local",
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            # repeat_penalty maps onto OpenAI's frequency_penalty loosely;
            # llama-server passes through model_kwargs as sampler params.
            model_kwargs={
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            timeout=600,
            max_retries=2,
            **cfg.extra_kwargs,
        )
        logger.info("ChatOpenAI client ready.")
        return llm
    except Exception as exc:
        logger.exception("Failed to build ChatOpenAI client.")
        raise RuntimeError(f"Could not build ChatOpenAI client: {exc}") from exc


# ---------------------------------------------------------------------------
# Helper: format retrieved docs for the prompt
# ---------------------------------------------------------------------------
def format_documents(
    docs: List[Document],
    summaries: Optional[Dict[str, str]] = None,
) -> str:
    """Render retrieved chunks (and optional doc-level summaries) for the
    answer prompt.

    Avoids leading [1]/[2]/[3] numbering on purpose: small models tend to
    mirror that pattern and produce 'enumerate-each-chunk' answers instead
    of synthesizing. Natural-language headers keep the citations available
    without inviting the format to be copied verbatim into the reply.

    When `summaries` is provided, the rendered context begins with a
    compact 'Documents in scope' overview followed by '## Excerpts'. This
    gives the model document-level awareness even when the retrieved
    excerpts skew toward one file.
    """
    parts: List[str] = []
    if summaries:
        parts.append(doc_summaries.format_overview(summaries))
        parts.append("")  # blank line
        parts.append("Excerpts:")
    for d in docs:
        meta = d.metadata or {}
        filename = meta.get("filename", "unknown")
        page = meta.get("page") or meta.get("pages", "?")
        heading = meta.get("headings", "")
        header = f"--- From {filename} (page {page})"
        if heading:
            header += f", section: {heading}"
        header += " ---"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts) if parts else "(no relevant context found)"


# ---------------------------------------------------------------------------
# Citation parsing for source filtering.
# ---------------------------------------------------------------------------
# Match `[filename.pdf p.5]`, `[filename.pdf p. 5]`, `[filename.pdf, p.5]`,
# `[filename.pdf p.5-7]`. Filenames in our metadata always end in a known
# extension; require ".pdf" to keep the pattern strict.
_CITATION_RE = re.compile(
    r"\[([^\[\]]+?\.pdf)\s*[, ]\s*(?:p\.?|page)\s*(\d+(?:\s*[-–]\s*\d+)?)\]",
    re.IGNORECASE,
)


def parse_citations(answer_text: str) -> List[tuple[str, str]]:
    """Extract (filename, page-or-range) tuples cited in the answer."""
    if not answer_text:
        return []
    return [(m.group(1).strip(), m.group(2).strip())
            for m in _CITATION_RE.finditer(answer_text)]


def filter_sources_by_citations(
    docs: List[Document],
    answer_text: str,
) -> List[Document]:
    """Return only the source documents whose (filename, page) is mentioned
    inline in the answer. Falls back to ALL docs if no citations parsed
    (so the user still sees evidence even if the model forgot to cite)."""
    cites = parse_citations(answer_text)
    if not cites:
        return docs

    cited_pairs: set[tuple[str, str]] = set()
    for fn, page in cites:
        # Range like "5-7" → all pages in [5,7]
        if "-" in page or "–" in page:
            try:
                lo, hi = re.split(r"[-–]", page)
                for p in range(int(lo.strip()), int(hi.strip()) + 1):
                    cited_pairs.add((fn.lower(), str(p)))
            except Exception:
                cited_pairs.add((fn.lower(), page.strip()))
        else:
            cited_pairs.add((fn.lower(), page.strip()))

    cited_files = {fn for fn, _ in cited_pairs}
    out: List[Document] = []
    for d in docs:
        meta = d.metadata or {}
        fn = str(meta.get("filename", "")).lower()
        page = str(meta.get("page", "") or "")
        if (fn, page) in cited_pairs or fn in cited_files:
            # Match by exact (file,page) first, else fall back to file-level
            # so users still get the right file even if the page is off-by-one.
            out.append(d)
    return out or docs  # never return empty; user wants to see SOMEthing


# ---------------------------------------------------------------------------
# Stateful RAG chain
# ---------------------------------------------------------------------------
class RAGChain:
    """
    Stateful RAG chain combining:
      - per-session chat history
      - standalone-question rewriting
      - local Chroma retrieval
      - local llama.cpp answer generation
    """

    def __init__(
        self,
        vector_store: Optional[LocalVectorStore] = None,
        llm: Optional[ChatOpenAI] = None,
        llm_config: Optional[LLMConfig] = None,
        k: int = 6,
    ) -> None:
        self.vector_store = vector_store or build_default_store()
        self.llm = llm or load_local_llm(llm_config)
        self.k = k
        self._sessions: Dict[str, BaseChatMessageHistory] = {}

        self._chain = self._build_chain()
        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Chitchat short-circuit: skips retrieval, calls LLM directly with
        # the conversational prompt. Same memory backing as the main chain.
        self._chitchat_chain = (
            CHITCHAT_PROMPT
            | self.llm
            | StrOutputParser()
            | RunnableLambda(self._strip_reasoning)
        )
        self._chitchat_with_history = RunnableWithMessageHistory(
            self._chitchat_chain,
            self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )

    # ----------------------------------------------------------- session mgmt
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ----------------------------------------------------------- chain wiring
    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """
        Remove <think>...</think> blocks emitted by reasoning models
        (DeepSeek-R1, Qwen-Thinking, etc.). Also handles the case where
        the model gets truncated mid-think and never closes the tag —
        in which case we drop everything from <think> onward.
        """
        if not text:
            return ""
        # Closed think blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Unclosed leading think (model still inside CoT when output ended)
        text = re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip stray tags
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _build_chain(self) -> Runnable:
        condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()

        def _post_condense_factory():
            """Closure that has access to the original question for sanity checks."""
            def _post(inputs: dict) -> str:
                raw = inputs["_raw"]
                original = inputs["question"]
                cleaned = self._strip_reasoning(raw)
                # Strip quotes / labels the model loves to add.
                cleaned = re.sub(r'^["\'\s]+|["\'\s]+$', "", cleaned)
                cleaned = re.sub(
                    r"^(query|search query|standalone query)\s*[:\-]\s*",
                    "", cleaned, flags=re.IGNORECASE,
                )

                # Defensive fallback: small models often regurgitate prior
                # turns into the rewrite. If the rewrite is much longer than
                # the original, or is empty, prefer the user's actual question.
                if not cleaned:
                    logger.info("Rewriter empty -> using original question.")
                    return original
                if len(cleaned) > max(120, 3 * len(original)):
                    logger.info(
                        "Rewriter output suspiciously long (%d vs original %d) "
                        "-> using original question.",
                        len(cleaned), len(original),
                    )
                    return original

                logger.info("Query rewritten to: %s", cleaned)
                return cleaned
            return _post

        post = _post_condense_factory()
        # Keep both the raw rewrite and the original question in scope.
        condense_chain = (
            RunnablePassthrough.assign(_raw=condense_chain)
            | RunnableLambda(post)
        )

        def _retrieve(state: dict) -> List[Document]:
            # Fall back to the raw question if the rewriter produced nothing
            # useful (common with reasoning models that get truncated).
            query = state.get("standalone") or state["question"]
            if not query.strip():
                query = state["question"]

            # Follow-up boost: when the user's actual message is very short
            # AND chat history has a recent assistant turn, append a snippet
            # of that turn to the search query. Small models often produce
            # a weak rewrite for bare 'why?' / '为什么', and similarity
            # search on those alone returns garbage.
            original = state.get("question", "")
            if len(original.strip()) < 15:
                history = state.get("chat_history") or []
                last_ai = next(
                    (m for m in reversed(history)
                     if getattr(m, "type", "") == "ai"
                     or m.__class__.__name__ == "AIMessage"),
                    None,
                )
                if last_ai is not None:
                    ai_snippet = (getattr(last_ai, "content", "") or "")[:300]
                    if ai_snippet:
                        query = f"{query} (context: {ai_snippet})"
                        logger.info("Follow-up boost engaged for short query.")

            # Optional per-call scope: list of filenames to restrict
            # retrieval to. Prevents cross-document interference when the
            # vector store holds multiple PDFs.
            filenames = state.get("filenames") or None

            # Decide effective scope: explicit list, else everything indexed.
            effective_scope = (
                filenames if filenames else self.vector_store.list_filenames()
            )

            # Balanced retrieval whenever 2+ docs are in play. Keeps a single
            # semantically-dominant doc from crowding out the rest, which
            # otherwise makes 'what's common between the two' or 'which one
            # is about X' unanswerable.
            if len(effective_scope) >= 2:
                docs = self.vector_store.similarity_search_balanced(
                    query, k=self.k, filenames=effective_scope,
                )
                mode = "balanced"
            else:
                docs = self.vector_store.similarity_search(
                    query, k=self.k, filenames=filenames,
                )
                mode = "topk"

            scope_label = ",".join(filenames) if filenames else "ALL"
            logger.info(
                "Retrieved %d chunks (%s, scope: %s).",
                len(docs), mode, scope_label,
            )
            return docs

        def _build_context(state: dict) -> str:
            # Pull doc summaries for whichever filenames showed up in the
            # retrieved excerpts (so the model sees both excerpt details
            # AND a doc-level overview to disambiguate cross-doc questions).
            seen = []
            for d in state["source_documents"]:
                fn = (d.metadata or {}).get("filename")
                if fn and fn not in seen:
                    seen.append(fn)
            summaries = doc_summaries.get_summaries_for(seen)
            return format_documents(state["source_documents"], summaries=summaries)

        # Bind helpers we'll need from the streaming path too.
        self._post_condense = post
        self._retrieve_fn = _retrieve
        self._condense_chain = condense_chain
        self._build_context = _build_context

        # Pipeline: produce {question, chat_history, standalone, context, source_documents}
        pipeline = (
            RunnablePassthrough.assign(standalone=condense_chain)
            .assign(source_documents=_retrieve)
            .assign(context=_build_context)
        )

        answer_chain = (
            ANSWER_PROMPT
            | self.llm
            | StrOutputParser()
            | RunnableLambda(self._strip_reasoning)
        )

        chain = pipeline.assign(answer=answer_chain)
        return chain

    # --------------------------------------------------------------- main API
    def ask(
        self,
        question: str,
        session_id: str = "default",
        filenames: Optional[List[str]] = None,
    ) -> dict:
        """Run a single conversational turn. Returns the full pipeline state.

        Parameters
        ----------
        question : str
            User input.
        session_id : str
            Conversational memory key.
        filenames : list[str], optional
            Restrict retrieval to these source files. None / empty = search
            the entire vector store. Greetings short-circuit and ignore this.

        Greetings / meta-questions short-circuit to a chitchat reply that
        skips vector retrieval entirely.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = self._get_session_history(session_id)
        has_history = bool(history.messages)

        if is_chitchat(question, has_history=has_history):
            logger.info("Chitchat detected, bypassing retrieval.")
            try:
                answer = self._chitchat_with_history.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": session_id}},
                )
            except Exception as exc:
                logger.exception("Chitchat chain invocation failed.")
                raise RuntimeError("Chitchat chain failed.") from exc
            return {
                "question": question,
                "standalone_question": question,
                "answer": answer,
                "source_documents": [],
                "all_source_documents": [],
                "chitchat": True,
                "scope": None,
            }

        inputs: Dict[str, object] = {"question": question}
        if filenames:
            inputs["filenames"] = list(filenames)

        try:
            result = self._chain_with_history.invoke(
                inputs,
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as exc:
            logger.exception("RAG chain invocation failed.")
            raise RuntimeError("RAG chain failed to produce an answer.") from exc

        all_docs = result.get("source_documents", [])
        answer_text = result.get("answer", "")
        # Show only sources actually cited in the answer; keep the full list
        # under all_source_documents for callers that want to see everything.
        cited = filter_sources_by_citations(all_docs, answer_text)

        return {
            "question": question,
            "standalone_question": result.get("standalone", question),
            "answer": answer_text,
            "source_documents": cited,
            "all_source_documents": all_docs,
            "chitchat": False,
            "scope": list(filenames) if filenames else None,
        }

    def ask_no_memory(
        self,
        question: str,
        filenames: Optional[List[str]] = None,
    ) -> dict:
        """Stateless one-shot query, useful for evaluation."""
        inputs: Dict[str, object] = {"question": question, "chat_history": []}
        if filenames:
            inputs["filenames"] = list(filenames)
        try:
            result = self._chain.invoke(inputs)
        except Exception as exc:
            logger.exception("Stateless RAG chain invocation failed.")
            raise RuntimeError("Stateless RAG chain failed.") from exc
        return {
            "question": question,
            "standalone_question": result.get("standalone", question),
            "answer": result.get("answer", ""),
            "source_documents": result.get("source_documents", []),
            "scope": list(filenames) if filenames else None,
        }

    # ------------------------------------------------------------ streaming
    def stream(
        self,
        question: str,
        session_id: str = "default",
        filenames: Optional[List[str]] = None,
    ):
        """
        Token-by-token streaming variant of ``ask``. Yields tagged events:

          - ``("standalone", str)``   — the rewritten / fallback search query
          - ``("sources", List[Doc])``— retrieved chunks (RAG path only)
          - ``("token", str)``        — incremental answer text (with
            ``<think>...</think>`` blocks already stripped)
          - ``("done", dict)``        — final result, same shape as ``ask``

        Memory is updated with the cleaned full answer at the end of the
        stream, mirroring what RunnableWithMessageHistory does for ``ask``.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        history = self._get_session_history(session_id)
        chat_history = list(history.messages)
        has_history = bool(chat_history)

        # ---- chitchat path ----------------------------------------------
        if is_chitchat(question, has_history=has_history):
            logger.info("Chitchat detected, bypassing retrieval (streaming).")
            yield ("standalone", question)
            yield ("sources", [])

            messages = CHITCHAT_PROMPT.format_messages(
                chat_history=chat_history,
                question=question,
            )
            full_raw = ""
            stripper = _ThinkStripFilter()
            for chunk in self.llm.stream(messages):
                piece = getattr(chunk, "content", str(chunk)) or ""
                if not piece:
                    continue
                full_raw += piece
                visible = stripper.feed(piece)
                if visible:
                    yield ("token", visible)
            tail = stripper.flush()
            if tail:
                yield ("token", tail)

            cleaned = self._strip_reasoning(full_raw)
            history.add_user_message(question)
            history.add_ai_message(cleaned)

            yield ("done", {
                "question": question,
                "standalone_question": question,
                "answer": cleaned,
                "source_documents": [],
                "all_source_documents": [],
                "chitchat": True,
                "scope": None,
            })
            return

        # ---- RAG path ---------------------------------------------------
        # 1. Standalone-question rewrite (non-streaming, short).
        try:
            raw = self._condense_chain.invoke(
                {"question": question, "chat_history": chat_history}
            )
        except Exception as exc:
            logger.exception("Condense chain failed during streaming.")
            raise RuntimeError("Failed to rewrite query.") from exc
        standalone = self._post_condense({"_raw": raw, "question": question})
        yield ("standalone", standalone)

        # 2. Retrieve. Pass chat_history so the follow-up retrieval boost
        # can pick up the previous assistant message for short questions.
        state = {
            "question": question,
            "standalone": standalone,
            "chat_history": chat_history,
        }
        if filenames:
            state["filenames"] = list(filenames)
        docs = self._retrieve_fn(state)
        yield ("sources", docs)

        # 3. Stream the grounded answer (with doc summaries prepended).
        seen_files: List[str] = []
        for d in docs:
            fn = (d.metadata or {}).get("filename")
            if fn and fn not in seen_files:
                seen_files.append(fn)
        summaries = doc_summaries.get_summaries_for(seen_files)
        messages = ANSWER_PROMPT.format_messages(
            chat_history=chat_history,
            context=format_documents(docs, summaries=summaries),
            question=question,
        )
        full_raw = ""
        stripper = _ThinkStripFilter()
        try:
            for chunk in self.llm.stream(messages):
                piece = getattr(chunk, "content", str(chunk)) or ""
                if not piece:
                    continue
                full_raw += piece
                visible = stripper.feed(piece)
                if visible:
                    yield ("token", visible)
            tail = stripper.flush()
            if tail:
                yield ("token", tail)
        except Exception as exc:
            logger.exception("Streaming answer generation failed.")
            raise RuntimeError("Streaming generation failed.") from exc

        cleaned = self._strip_reasoning(full_raw)
        history.add_user_message(question)
        history.add_ai_message(cleaned)

        # Filter sources to those actually cited in the final answer; keep
        # the full list under all_source_documents for callers that want it.
        cited = filter_sources_by_citations(docs, cleaned)

        yield ("done", {
            "question": question,
            "standalone_question": standalone,
            "answer": cleaned,
            "source_documents": cited,
            "all_source_documents": docs,
            "chitchat": False,
            "scope": list(filenames) if filenames else None,
        })

    # ----------------------------------------------------------- summaries
    SUMMARY_SYSTEM = (
        QWEN3_NO_THINK + "\n"
        "Summarize the document in 2-3 sentences. Be concrete: name the "
        "topic / domain, the methodology or approach, and the main "
        "contribution or finding. No filler, no preamble. Keep it under "
        "60 words."
    )

    def summarize_document(self, filename: str, full_text: str) -> str:
        """Generate a 2-3 sentence summary of a document and cache it.
        Idempotent — returns the cached summary if one already exists."""
        existing = doc_summaries.get_summary(filename)
        if existing:
            return existing
        # 12K chars covers most paper abstracts + intros; the model only
        # needs the gist, not the entire body.
        excerpt = (full_text or "")[:12000]
        if not excerpt.strip():
            return ""
        try:
            messages = [
                SystemMessage(content=self.SUMMARY_SYSTEM),
                HumanMessage(content=(
                    f"Document: {filename}\n\nContent:\n{excerpt}\n\n"
                    "Summary:"
                )),
            ]
            resp = self.llm.invoke(messages)
            text = self._strip_reasoning(getattr(resp, "content", str(resp)))
            text = " ".join(text.split())
            doc_summaries.set_summary(filename, text)
            logger.info("Summarized %s: %s", filename, text[:80])
            return text
        except Exception as exc:
            logger.exception("Failed to summarize %s", filename)
            return ""

    # --------------------------------------------------------- suggestions
    def suggest_followups(
        self,
        last_question: str,
        last_answer: str,
        max_suggestions: int = 3,
    ) -> List[str]:
        """Generate up to N short follow-up questions the user might ask
        next, anchored on the documents currently indexed."""
        if not last_answer:
            return []
        summaries = doc_summaries.all_summaries()
        if summaries:
            overview = doc_summaries.format_overview(summaries)
        else:
            files = self.vector_store.list_filenames()
            overview = "\n".join(f"- {fn}" for fn in files) or "(no docs)"
        try:
            messages = SUGGESTIONS_PROMPT.format_messages(
                documents_overview=overview,
                question=last_question,
                answer=last_answer[:600],
            )
            resp = self.llm.invoke(messages)
            text = self._strip_reasoning(getattr(resp, "content", str(resp)))
        except Exception:
            logger.exception("Suggestion generation failed.")
            return []
        # Pick at most N non-empty lines that look like questions.
        out: List[str] = []
        for line in text.splitlines():
            s = re.sub(r"^[\s\-\d\.\)、]+", "", line).strip()
            s = re.sub(r"^['\"]+|['\"]+$", "", s)
            if not s or len(s) < 4:
                continue
            out.append(s)
            if len(out) >= max_suggestions:
                break
        return out


# ---------------------------------------------------------------------------
# CLI: simple REPL for local testing
# ---------------------------------------------------------------------------
def main() -> None:
    chain = RAGChain()
    session_id = "cli"
    print("Local RAG REPL. Ctrl-C to quit.\n")
    try:
        while True:
            q = input("you> ").strip()
            if not q:
                continue
            result = chain.ask(q, session_id=session_id)
            print(f"\nbot> {result['answer']}\n")
            srcs = result["source_documents"]
            if srcs:
                print("sources:")
                for d in srcs:
                    meta = d.metadata
                    print(f"  - {meta.get('filename')} p.{meta.get('page', '?')}")
                print()
    except (KeyboardInterrupt, EOFError):
        print("\nbye.")


if __name__ == "__main__":
    main()
