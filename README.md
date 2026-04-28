# Local PDF RAG Assistant

A fully offline, GPU-accelerated, stateful RAG assistant for PDFs. Parses
documents with **IBM Docling** (preserves tables, code, math), embeds with
**sentence-transformers**, persists in **ChromaDB**, and answers via a local
**llama.cpp** server with any GGUF model. Streamlit UI on top, Ragas
evaluation on the side. No cloud, no API keys.

## Architecture

```
   ┌────────────┐    HTTP / OpenAI API    ┌──────────────────┐
   │  app.py    │  ─────────────────────► │ llama-server.exe │
   │ (Streamlit)│                         │  GPU CUDA, GGUF  │
   └─────┬──────┘                         └──────────────────┘
         │ uses
         ▼
   ┌──────────────────┐    embed     ┌────────────────┐
   │   llm_chain.py   │  ─────────►  │   ChromaDB     │
   │ (LangChain RAG)  │              │  (persistent)  │
   └─────┬────────────┘              └────────────────┘
         │ retrieve
         ▼
   ┌──────────────┐
   │  ingest.py   │  Docling + HybridChunker → LangChain Documents
   └──────────────┘
```

The chain has two paths:
- **Chitchat** ("你好" / "hello" / "thank you" / meta-questions) → bypasses
  retrieval, replies conversationally.
- **Document query** → standalone-question rewrite → vector search → grounded
  answer with inline citations `[file p.<page>]`.

## Project layout

| File | What it does |
|---|---|
| `ingest.py` | PDF → Docling → HybridChunker → LangChain `Document`s with rich metadata (headings, page numbers, content type). |
| `vector_store.py` | HuggingFace `all-MiniLM-L6-v2` embeddings + persistent ChromaDB collection at `./chroma_db`. Idempotent upserts via SHA-256 IDs. |
| `llama_server.py` | Lifecycle manager for the standalone `llama-server.exe` (subprocess + `/health` polling + auto-shutdown). |
| `llm_chain.py` | `LLMConfig`, `RAGChain` (memory + standalone-question rewriter + chitchat short-circuit), `ChatOpenAI` client pointed at the local server. |
| `evaluate.py` | Offline Ragas evaluation. Computes Context Precision / Faithfulness / Answer Relevancy with the LOCAL judge LLM. Writes JSON+CSV to `eval_results/`. |
| `app.py` | Streamlit chat UI. Sidebar: upload + replace-or-append index + clear-all-documents button. Main pane: stateful chat with source citations. |
| `smoke_test.py` | End-to-end smoke test (3 questions against the indexed PDFs). |
| `eval_dataset.json` | Sample test set for `evaluate.py`. |

## Requirements

- **Windows** (Linux/macOS works for the Python parts; you'd swap the
  `llama-server.exe` path for the native binary)
- **Python 3.12** (3.13/3.14 lack prebuilt wheels for `scikit-network`, a
  transitive dependency of Docling)
- **NVIDIA GPU** with recent driver (CUDA 12.x or newer). CPU also works,
  just slower — set `n_gpu_layers=0` in `LLMConfig`.
- Free VRAM matched to your model + KV cache. The defaults assume a
  ~4 GB card; bigger models or longer context need more.

## Setup

### 1. Python venv

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

Note: `requirements.txt` does **not** pin `llama-cpp-python` because we don't
use it. We talk to `llama-server.exe` over HTTP instead — see step 3.

### 3. Get `llama-server.exe`

Download a recent build from <https://github.com/ggerganov/llama.cpp/releases>
(pick a `*-bin-win-cuda-*.zip` matching your CUDA version). Extract anywhere
and set the path:

```powershell
$env:LLAMA_SERVER_EXE = "C:\path\to\llama-server.exe"
```

Or edit `DEFAULT_SERVER_EXE` in `llama_server.py`.

### 4. Place a GGUF model

Drop any GGUF model at `./models/model.gguf` (overridable via
`LLAMA_MODEL_PATH`). Any architecture supported by your `llama-server.exe`
build will work — pick a quantization that fits your VRAM.

### 5. (One-time) Index a PDF

```powershell
python vector_store.py path\to\your.pdf
```

…or just upload from the Streamlit UI.

## Run

### Streamlit UI

```powershell
streamlit run app.py
```

Sidebar:
- **Upload PDF(s)** + **Ingest & Index** to add to the vector store
- **Replace existing index on upload** (default on) — wipes ChromaDB before
  ingesting so queries only see the new docs
- **🗑 Clear all documents** — manual nuke button
- **Indexed documents (N)** — live list of what's currently searchable

Main pane: chat. Each answer shows expandable **Sources** (file + page +
heading + snippet) and **Standalone search query** (the rewritten query that
drove retrieval).

### CLI smoke test

```powershell
python smoke_test.py
```

Runs 3 questions end-to-end against whatever's currently indexed. Prints
timings and source citations.

### Ragas evaluation

```powershell
python evaluate.py --test-set eval_dataset.json
```

Outputs metrics to `eval_results/ragas_results_<timestamp>.json` and `.csv`.
Default metrics: `context_precision` + `faithfulness` + `answer_relevancy`.

## Configuration knobs

`LLMConfig` (in `llm_chain.py`):

| Field | Default | Notes |
|---|---|---|
| `n_ctx` | `32768` | Total context across server slots |
| `n_gpu_layers` | `-1` | All layers on GPU; set `0` for CPU |
| `temperature` | `0.2` | Low for deterministic answers |
| `max_tokens` | `2048` | Generous so Ragas judge JSON doesn't truncate |

`ServerConfig` (in `llama_server.py`):

| Field | Default | Notes |
|---|---|---|
| `port` | `8765` | Avoids common Streamlit/Jupyter conflicts |
| `parallel` | `4` | Lets Ragas's answer_relevancy fan out N completions |
| `startup_timeout_s` | `90.0` | First load can be slow on cold cache |

## Notes on the design choices

**Why a subprocess server instead of `llama-cpp-python`?** On Windows, the
prebuilt CUDA wheels for `llama-cpp-python` lag the upstream `llama.cpp`
release cadence by months, so newer GGUF architectures often fail to load.
Building from source needs MSVC + CUDA toolkit. Using the standalone
`llama-server.exe` from upstream releases sidesteps all of this and gives
you the latest model support for free.

**Why HybridChunker?** Docling's `HierarchicalChunker` produces lots of tiny
single-sentence chunks (e.g. just a section header). `HybridChunker` merges
adjacent peers up to a token budget, so each chunk carries enough context
for high-precision retrieval.

**Defensive standalone-query rewrite.** Small reasoning models sometimes
regurgitate prior conversation turns into the rewritten query, or emit
SQL, or get truncated mid-`<think>`. The chain detects these failure
modes (length heuristic + `<think>` stripping) and falls back to the raw
user question.

**Chitchat short-circuit.** A regex-based detector matches common greetings,
thanks, farewells, and meta-questions in CN+EN. These bypass retrieval and
hit a friendly conversational prompt instead — so "你好" no longer gets
answered with "I don't know based on the provided documents."

## License

MIT.
