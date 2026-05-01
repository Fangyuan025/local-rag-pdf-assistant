"""FastAPI backend for Hushdoc.

Run in dev with:
    uvicorn server.main:app --reload --port 8000

The Streamlit ``app.py`` is kept untouched during the React-frontend
migration; both UIs can run side by side until ``app.py`` is removed
in the final cleanup commit.
"""
import os

# llama-cpp-python (when imported transitively) and PyTorch ship duplicate
# OpenMP runtimes on Windows; loading both segfaults without this flag.
# MUST be set before any heavy imports below.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
