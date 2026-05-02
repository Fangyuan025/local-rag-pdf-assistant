"""
SSE adapter for ``RAGChain.stream``.

The chain emits tagged 2-tuples: ``("standalone"|"sources"|"token"|"done", payload)``.
We yield dicts with ``event`` + ``data`` keys; ``sse_starlette.EventSourceResponse``
takes care of wrapping them as proper SSE frames on the wire. (Hand-rolling the
``event: x\\ndata: y\\n\\n`` text ourselves causes double-encoding because
EventSourceResponse re-wraps each line under ``data:``.)

Source documents (LangChain ``Document``) are flattened to a small dict the
React client can render directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Iterator

from langchain_core.documents import Document

logger = logging.getLogger("server.streaming")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _doc_to_dict(d: Document) -> Dict[str, Any]:
    """Flatten a LangChain Document into a JSON-serialisable shape the
    frontend can render without extra processing."""
    meta = d.metadata or {}
    snippet = d.page_content
    if len(snippet) > 400:
        snippet = snippet[:400] + "..."
    return {
        "filename": meta.get("filename", "unknown"),
        "page": meta.get("page", None),
        "headings": meta.get("headings", ""),
        "snippet": snippet,
    }


def _sse(event: str, payload: Any) -> Dict[str, str]:
    """Build an EventSourceResponse-shaped dict. Payload is JSON-encoded
    so the React client can ``JSON.parse(e.data)`` uniformly across all
    event kinds."""
    return {
        "event": event,
        "data": json.dumps(payload, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Bridges
# ---------------------------------------------------------------------------
async def chain_stream_to_sse(
    events: Iterator[tuple],
) -> AsyncIterator[Dict[str, str]]:
    """
    Bridge the synchronous ``RAGChain.stream`` generator to an async SSE
    stream. Heavy LLM work runs in a background thread so the event loop
    stays responsive while tokens trickle in.
    """
    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL = object()
    loop = asyncio.get_running_loop()

    def producer() -> None:
        try:
            for kind, payload in events:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, payload))
        except Exception as exc:  # pragma: no cover
            logger.exception("Chain stream producer crashed.")
            loop.call_soon_threadsafe(
                queue.put_nowait, ("error", {"message": str(exc)})
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)

    asyncio.create_task(asyncio.to_thread(producer))

    while True:
        item = await queue.get()
        if item is SENTINEL:
            return
        kind, payload = item
        if kind == "token":
            yield _sse("token", {"text": payload})
        elif kind == "standalone":
            yield _sse("standalone", {"query": payload})
        elif kind == "sources":
            yield _sse("sources", {"docs": [_doc_to_dict(d) for d in payload]})
        elif kind == "done":
            done = dict(payload)
            done["source_documents"] = [
                _doc_to_dict(d) for d in done.get("source_documents", [])
            ]
            done.pop("all_source_documents", None)
            yield _sse("done", done)
        elif kind == "error":
            yield _sse("error", payload)
        else:
            yield _sse(kind, payload)


async def events_to_sse(
    items: AsyncIterator[tuple],
) -> AsyncIterator[Dict[str, str]]:
    """Wrap an async (kind, payload) iterator as an SSE-shaped dict
    iterator for upload progress + similar non-chain streams."""
    async for kind, payload in items:
        yield _sse(kind, payload)
