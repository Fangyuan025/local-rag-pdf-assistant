"""FastAPI backend for Hushdoc.

Run in dev with:
    uvicorn server.main:app --reload --port 8000

Pairs with the React frontend in ``web/`` (Vite dev server on :5173,
which proxies /api/* to this backend).
"""
import os

# llama-cpp-python (when imported transitively) and PyTorch ship duplicate
# OpenMP runtimes on Windows; loading both segfaults without this flag.
# MUST be set before any heavy imports below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
