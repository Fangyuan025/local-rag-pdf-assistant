# 🤫 Hushdoc

> **A local-only PDF assistant that keeps every word between you and your machine.**

Privacy-first, fully offline, GPU-accelerated, stateful RAG over your own
documents. Nothing about your files — not the bytes, not the chunks, not
your questions, not the answers — ever leaves your computer. The only
network calls are the one-time HuggingFace downloads of the embedding /
ASR / TTS models; once those are cached, you can run completely
air-gapped.

## What's in the box

- **Document ingestion**: PDF, DOCX, and document photos (JPG / PNG /
  TIFF / BMP) via [IBM Docling](https://github.com/DS4SD/docling) —
  preserves tables, code, LaTeX math, and OCRs images via RapidOCR
- **Embedding + vector store**: `sentence-transformers/all-MiniLM-L6-v2`
  → persistent ChromaDB at `./chroma_db`
- **LLM**: any GGUF model served by the upstream
  [`llama.cpp`](https://github.com/ggerganov/llama.cpp) standalone
  binary, with full GPU offload
- **HTTP API**: a small **FastAPI** app that wraps every backend
  capability behind JSON + SSE endpoints (`/api/chat`, `/api/documents`,
  `/api/voice/*`, `/api/health`)
- **UI**: a modern **React + Tailwind + Shadcn** frontend with
  token-by-token streaming, multi-document scope filtering, automatic
  Chinese / English language matching, citation-anchored sources, and
  an optional voice mode (Whisper-base.en speech-in, Kokoro-82M
  speech-out)
- **Eval**: offline [Ragas](https://github.com/explodinggradients/ragas)
  scoring with the local LLM as judge

## Architecture

```
   ┌───────────────────────────────────────┐    HTTP / SSE     ┌─────────────────────────┐
   │  web/  (React + Vite + Tailwind)      │ ───────────────► │  server/  (FastAPI)     │
   │  ─ ChatPane (streaming markdown)      │ ◄─────────────── │  ─ /api/chat (SSE)       │
   │  ─ Sidebar (upload · scope · voice)   │     /api/*       │  ─ /api/documents/*      │
   │  ─ Browser VAD + hidden TTS autoplay  │     proxy        │  ─ /api/voice/*          │
   └─────────────────────┬─────────────────┘                  └────────────┬────────────┘
                         │                                                  │ imports
                         ▼ (via Vite dev proxy)                             ▼
                http://localhost:8000                  ┌──────────────────────────────────┐
                                                       │ ingest · vector_store · llm_chain│
                                                       │ doc_summaries · voice            │
                                                       │ llama_server (subprocess mgmt)   │
                                                       └──────────────────────────────────┘
                                                                          │
                                                                          ▼
                                                       ┌──────────────────────────────────┐
                                                       │ llama-server.exe (GPU CUDA, GGUF)│
                                                       │ ChromaDB · summaries.json sidecar│
                                                       └──────────────────────────────────┘
```

Per-turn flow inside `llm_chain.py`:

1. **Language detection** — classifies the user message as `zh` or `en`
   and prepares the language directive for the prompt.
2. **Routing** — `is_chitchat(text, has_history=...)`:
   - **Chitchat** ("你好" / "Morning!" / "introduce yourself" / thanks)
     → bypasses retrieval and hits a friendly prompt.
   - **Document query** → continues to step 3.
3. **Standalone-question rewrite** — resolves pronouns and short
   follow-ups; defensive fallback to the raw question on suspicious
   rewrites; for very short follow-ups, a snippet of the previous
   assistant message is appended to the search query as a safety net.
4. **Retrieval** — top-k similarity for single-doc scopes, *balanced*
   per-doc retrieval when 2+ docs are in scope.
5. **Context build** — prepends a "Documents in scope" overview from
   the per-PDF summary cache (`chroma_db/summaries.json`).
6. **Streaming answer** — tokens stream through SSE straight from
   `llama-server` into the React `<ChatPane>`; an FSM strips
   `<think>...</think>` blocks inline so reasoning never reaches the UI.
7. **Citation filter** — Sources panel shows only the excerpts the
   answer actually cites with `[filename p.<page>]`.

## Project layout

```
hushdoc/
├── server/                # FastAPI backend
│   ├── main.py            # routes (health, documents, chat, voice)
│   ├── deps.py            # lazy singletons (chain, store, ingestor)
│   ├── schemas.py         # Pydantic request/response models
│   └── streaming.py       # SSE adapter for RAGChain.stream()
├── web/                   # React + Vite frontend
│   ├── src/
│   │   ├── App.tsx        # shell + theme + global shortcuts
│   │   ├── components/
│   │   │   ├── ChatPane / ChatMessage / ChatInput / Sources
│   │   │   ├── Sidebar (Documents · Scope · Voice)
│   │   │   ├── DocumentUpload / ScopeSelector / MicButton
│   │   │   └── ui/            # shadcn primitives
│   │   ├── hooks/
│   │   │   ├── useChat / useDocuments / useScope / useVoice
│   │   │   ├── useKeyboardShortcuts / useStickyBottom
│   │   ├── lib/{api, audio, vad, utils}.ts
│   │   └── types.ts
│   ├── vite.config.ts     # /api/* → :8000 proxy
│   └── package.json
├── ingest.py              # Docling parsing + HybridChunker
├── vector_store.py        # ChromaDB + balanced retrieval
├── llm_chain.py           # RAGChain (streaming, scope, follow-up, voice)
├── llama_server.py        # llama-server.exe lifecycle
├── doc_summaries.py       # per-PDF summary cache
├── voice.py               # Whisper-base.en ASR + Kokoro-82M TTS
├── evaluate.py            # offline Ragas scoring
├── test_api.py            # E2E HTTP smoke
├── smoke_test.py          # E2E chain smoke
├── dev.ps1 / dev.sh       # one-command dev launcher
└── requirements.txt
```

## Requirements

- **Python 3.12** (3.13/3.14 lack prebuilt wheels for `scikit-network`,
  a transitive dep of Docling)
- **Node.js 20+**
- **NVIDIA GPU** with CUDA 12.x or newer driver (CPU also works — set
  `n_gpu_layers=0` in `LLMConfig`)
- A GGUF model and the standalone `llama-server.exe` binary

## Setup

### 1. Backend (Python)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Frontend (Node)

```powershell
cd web
npm install
cd ..
```

### 3. `llama-server.exe`

Download a recent build from
<https://github.com/ggerganov/llama.cpp/releases> (pick a
`*-bin-win-cuda-*.zip` matching your CUDA version). Extract anywhere
and point Hushdoc at it:

```powershell
$env:LLAMA_SERVER_EXE = "C:\path\to\llama-server.exe"
```

Or edit `DEFAULT_SERVER_EXE` in `llama_server.py`.

### 4. GGUF model

Drop any GGUF model at `./models/model.gguf` (overridable via the
`LLAMA_MODEL_PATH` env var). Pick a quantization that fits your VRAM.

## Run

### One command (dev)

```powershell
.\dev.ps1            # PowerShell
# or
./dev.sh             # bash / Git Bash / Linux / macOS
```

Starts the FastAPI backend on `:8000` and the Vite dev server on
`:5173`, opens `http://localhost:5173/` in your browser, and forwards
Ctrl+C to both processes on shutdown.

### Two terminals (manual)

```powershell
# Terminal A — backend
.\.venv\Scripts\python.exe -m uvicorn server.main:app --port 8000

# Terminal B — frontend
cd web
npm run dev
```

Open <http://localhost:5173/>.

### CLI helpers

```powershell
# Index a file from the command line (no UI)
python vector_store.py path\to\your.pdf
python vector_store.py path\to\your.docx
python vector_store.py path\to\photo.jpg     # OCR via RapidOCR

# End-to-end smoke test (one chat turn over what's currently indexed)
python smoke_test.py

# E2E exercise of every HTTP route (requires uvicorn already running)
python test_api.py

# Offline Ragas eval
python evaluate.py --test-set eval_dataset.json
```

## Using the UI

**Sidebar** (left rail on desktop, hamburger drawer on mobile):

| Section | What it does |
|---|---|
| **Documents** | Drag-and-drop or pick PDFs / DOCX / photos. "Replace existing index on upload" wipes the store first; uncheck to add alongside what's already indexed. |
| **Search scope** | Multi-select of indexed files. Empty or all-selected ≡ "search the whole store"; partial selection restricts each query to the picked files. |
| **Voice** | Off by default. When on: a 🎤 mic icon appears beside the chat input (auto-stops on 1.5 s of silence), and the assistant answer auto-plays via a hidden `<audio>` element. **English only for now.** |

**Bottom of sidebar**: **New chat** (clears history) and **Clear all
documents** (wipes store + summaries; click-to-confirm).

**Main pane**:
- Empty state when nothing's been asked yet
- Streaming assistant answers with markdown, code, GFM tables, an
  inline `▍` cursor, and `[filename p.X]` citation chips below
- Per-message `🔊` replay icon when voice mode synthesised audio
- Floating "Jump to latest" pill if you scroll up mid-stream

**Keyboard shortcuts**:

| Key | Action |
|---|---|
| `Cmd/Ctrl + K` | Focus the chat input |
| `Cmd/Ctrl + L` | New chat (reset memory) |
| `Esc` | Cancel anything in flight (streaming or recording) |
| `Enter` | Send message |
| `Shift + Enter` | Newline in the input |

## Configuration knobs

`LLMConfig` (in `llm_chain.py`):

| Field | Default | Notes |
|---|---|---|
| `n_ctx` | `8192` | Context window |
| `n_gpu_layers` | `-1` | All layers on GPU; `0` = CPU only |
| `temperature` | `0.2` | Low for deterministic answers |
| `max_tokens` | `2048` | Generous so Ragas judge JSON doesn't truncate |

`ServerConfig` (in `llama_server.py`):

| Field | Default | Notes |
|---|---|---|
| `port` | `8765` | llama-server's HTTP port (separate from FastAPI's 8000) |
| `parallel` | `4` | Lets Ragas's answer_relevancy fan out N completions |
| `startup_timeout_s` | `90.0` | Cold cache may need this long |

## Notes on the design choices

**Why two processes instead of one?** Python is fine for the model
plumbing (Docling, ChromaDB, ragas, transformers), but a Vite + React
front-end is the only sane way to ship a modern streaming chat UI.
Splitting them lets each side use the tooling that fits — and since
Vite proxies `/api/*` to `:8000` in dev, the browser still sees a
single origin. A future Docker image would bundle them together
(FastAPI serving the built `web/dist/`).

**Why a subprocess llama.cpp server instead of `llama-cpp-python`?** On
Windows, the prebuilt CUDA wheels for `llama-cpp-python` lag the
upstream `llama.cpp` release cadence by months, so newer GGUF
architectures often fail to load. Using the standalone `llama-server.exe`
from upstream releases sidesteps that and gives you the latest model
support for free.

**Why HybridChunker?** `HierarchicalChunker` produces lots of tiny
single-sentence chunks. `HybridChunker` merges adjacent peers up to a
token budget so each chunk carries enough context for high-precision
retrieval.

**Multi-format ingestion (PDF / DOCX / image).** Docling auto-detects
format from the extension and routes:
- **PDF** → layout pipeline (Heron + TableFormer)
- **DOCX** → native XML parse (no OCR)
- **image** → RapidOCR

All three emit the same DoclingDocument structure, which then flows
through HybridChunker → embedding → RAG with no special-casing
downstream.

**Per-PDF doc summaries.** Vanilla top-k similarity is bad at
high-level questions ("which one is about ML?", "summarize this
paper") because chunks alone don't carry document themes. At ingest we
make one short LLM call per file for a 2–3 sentence summary, cache it
in `chroma_db/summaries.json`, and prepend it to the answer prompt as
a "Documents in scope" overview.

**Balanced multi-doc retrieval.** With 2+ PDFs in scope, retrieval
allocates the budget evenly across filenames so a single
semantically-dominant document can not crowd the others out — making
"what's common between the two?" actually answerable.

**Citation-filtered Sources.** The model is told to cite as
`[filename p.<page>]`. After streaming, a regex pulls those citations
out and the Sources panel shows only the cited excerpts. The full
retrieved set is still kept under `all_source_documents` for callers
that want it.

**Per-turn language directive.** Small models drift toward the
language of the source documents (usually English) regardless of the
user's language. Each turn we detect the question as `zh` or `en` and
splice the directive into the user message *immediately before*
`Answer:` — the most-recent-token slot, which small models obey most
reliably.

**Defensive standalone-query rewrite + follow-up boost.** Small
reasoning models sometimes regurgitate prior turns into the rewrite,
emit SQL, or get truncated mid-`<think>`. The chain falls back to the
raw question when the rewrite is suspicious, and for very short
follow-ups (`why?` / `为什么?`) it appends a snippet of the previous
assistant message to the search query as a safety net.

**SSE over WebSocket.** Chat is one-way (server → client) — perfect
SSE fit. `EventSource` doesn't support POST bodies though, so the React
client uses `fetch` + `ReadableStream` and hand-parses SSE frames.

**Streaming with inline `<think>` stripping.** Tokens stream from
`llama-server` through FastAPI's SSE bridge into the React ChatPane.
A small FSM strips the first `<think>...</think>` block as it goes,
including the case where the open or close tag is split across two
streamed chunks. First token visible in ~0.5 s for chitchat, ~2 s for
RAG.

**Voice mode (optional, English only).** The microphone capture and
voice activity detection (auto-stop on 1.5 s of silence) run entirely
in the browser via the Web Audio API. The recorded audio is
re-encoded to PCM WAV in-process (no ffmpeg dep) and POSTed to
`/api/voice/transcribe` (Whisper-base.en on the CPU). After the
assistant's answer streams in, `/api/voice/synthesize` (Kokoro-82M)
returns a WAV that auto-plays via a hidden `<audio>` element — no
visible progress bar — and a small 🔊 icon next to the message
re-plays the cached blob URL.

**Chitchat short-circuit.** Common greetings, thanks, farewells, and
identity / capability questions in CN+EN ("hello", "Morning!",
"你好", "早安", "介绍一下你自己", "thank you", "Howdy", etc.)
bypass retrieval entirely and hit a friendly conversational prompt.
Ambiguous short utterances ("why?", "为什么?") are NOT treated as
chitchat when there is prior chat history — they go through the RAG
path so the rewriter can expand them.

## Producing in production (later)

The repo is dev-mode only right now: `dev.ps1` / `dev.sh` start the
two processes side by side. A production build would `npm run build`
to produce `web/dist/`, then have FastAPI serve it from the same
origin and drop the CORS allowlist + the Vite proxy. A `Dockerfile`
that bundles the venv, llama-server binary, model, and built React
assets into one image is on the future-work list.

## License

MIT.
