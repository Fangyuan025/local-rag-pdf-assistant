"""End-to-end exercise of every P1/P2 route via httpx.

Run after `uvicorn server.main:app --port 8000` is up. Tests:
  GET  /api/health            (cold)
  GET  /api/documents
  DELETE /api/documents
  POST /api/documents/upload  (real PDF ingest, SSE)
  POST /api/chat              (real RAG turn, SSE)
  POST /api/chat/clear
  GET  /api/voice/health
  POST /api/voice/synthesize  (real Kokoro)
  POST /api/voice/transcribe  (round-trip the synthesized WAV)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

import httpx

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).parent
PDF = ROOT / "data" / "attention.pdf"


def parse_sse(text: str) -> list[tuple[str, object]]:
    """Line-oriented SSE parser per the spec — flushes one event per
    blank line. Keeps state so multi-data-line frames (uncommon for us)
    parse correctly, and so concurrent frames don't get merged when
    `iter_text` chunk boundaries fall mid-stream."""
    events: list[tuple[str, object]] = []
    cur_event = "message"
    cur_data: list[str] = []

    def flush():
        nonlocal cur_event, cur_data
        if cur_data:
            body = "\n".join(cur_data)
            try:
                payload: object = json.loads(body)
            except Exception:
                payload = body
            events.append((cur_event, payload))
        cur_event, cur_data = "message", []

    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if line == "":
            flush()
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            cur_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            cur_data.append(line[len("data:"):].lstrip())
    flush()
    return events


with httpx.Client(base_url=BASE, timeout=600) as client:
    print("=" * 70)
    print("HEALTH (cold)")
    print("=" * 70)
    r = client.get("/api/health")
    r.raise_for_status()
    print(r.json())

    print("\n" + "=" * 70)
    print("DELETE all docs (clean slate)")
    print("=" * 70)
    print(client.delete("/api/documents").json())

    print("\n" + "=" * 70)
    print("UPLOAD attention.pdf  (SSE)")
    print("=" * 70)
    with PDF.open("rb") as fh:
        with client.stream(
            "POST",
            "/api/documents/upload",
            files={"files": (PDF.name, fh, "application/pdf")},
            data={"replace": "true"},
        ) as resp:
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
        for ev, payload in parse_sse(buf):
            print(f"  [{ev}] {payload}")

    print("\n" + "=" * 70)
    print("LIST docs")
    print("=" * 70)
    print(client.get("/api/documents").json())

    print("\n" + "=" * 70)
    print("CHAT (SSE)  Q: What is the Transformer architecture?")
    print("=" * 70)
    with client.stream(
        "POST",
        "/api/chat",
        json={"question": "What is the Transformer architecture?",
              "session_id": "api_test"},
    ) as resp:
        full = ""
        token_count = 0
        for chunk in resp.iter_text():
            full += chunk
        for ev, payload in parse_sse(full):
            if ev == "token":
                token_count += 1
            elif ev == "standalone":
                print(f"  [standalone] {payload}")
            elif ev == "sources":
                print(f"  [sources] {len(payload.get('docs', []))} docs")
            elif ev == "done":
                print(f"  [done] answer ({len(payload.get('answer',''))} chars), "
                      f"sources={len(payload.get('source_documents', []))}, "
                      f"chitchat={payload.get('chitchat')}")
                print(f"         answer preview: {payload.get('answer','')[:200]}")
        print(f"  [stats] {token_count} token events received")

    print("\n" + "=" * 70)
    print("CHAT clear")
    print("=" * 70)
    print(client.post("/api/chat/clear", json={"session_id": "api_test"}).json())

    print("\n" + "=" * 70)
    print("VOICE health (cold)")
    print("=" * 70)
    print(client.get("/api/voice/health").json())

    print("\n" + "=" * 70)
    print("VOICE synthesize → transcribe round-trip")
    print("=" * 70)
    sample = "Hello, this is a Hushdoc API integration test."
    t0 = time.time()
    syn = client.post("/api/voice/synthesize", json={"text": sample})
    syn.raise_for_status()
    print(f"  synthesized {len(syn.content)} bytes in {time.time()-t0:.1f}s")

    t0 = time.time()
    tr = client.post(
        "/api/voice/transcribe",
        files={"audio": ("hello.wav", syn.content, "audio/wav")},
    )
    tr.raise_for_status()
    print(f"  transcribed in {time.time()-t0:.1f}s -> {tr.json()!r}")

    print("\n" + "=" * 70)
    print("HEALTH (warm)")
    print("=" * 70)
    print(client.get("/api/health").json())
    print(client.get("/api/voice/health").json())

print("\nALL ROUTES OK")
