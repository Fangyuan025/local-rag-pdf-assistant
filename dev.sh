#!/usr/bin/env bash
# Hushdoc dev launcher (bash).
# Starts FastAPI on :8000 + Vite on :5173, wires Ctrl+C to stop both.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Pick a Python: prefer the project venv, fall back to system python.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
    PY="$ROOT/.venv/Scripts/python.exe"   # Git Bash on Windows
else
    echo "[hushdoc] venv not found; create one with:" >&2
    echo "    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -d "$ROOT/web/node_modules" ]]; then
    echo "[hushdoc] web/node_modules missing — running 'npm install'..."
    (cd "$ROOT/web" && npm install)
fi

cleanup() {
    echo
    echo "[hushdoc] stopping..."
    [[ -n "${BACK_PID:-}" ]] && kill "$BACK_PID" 2>/dev/null || true
    [[ -n "${FRONT_PID:-}" ]] && kill "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[hushdoc] starting FastAPI backend on http://localhost:8000 ..."
( cd "$ROOT" && "$PY" -m uvicorn server.main:app --port 8000 ) &
BACK_PID=$!

echo "[hushdoc] starting Vite frontend on http://localhost:5173 ..."
( cd "$ROOT/web" && npm run dev ) &
FRONT_PID=$!

wait "$BACK_PID" "$FRONT_PID"
