"""
Document-level summary cache.

Stores a 2-3 sentence summary per indexed PDF in a single JSON file
alongside the Chroma vector store. Summaries are generated once at ingest
time using the local LLM, then injected into the answer-prompt context as
a high-level "Documents in scope" overview. This lets the model answer
questions like "which one is about X?" or "summarize this paper" using
something denser than scattered chunks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

logger = logging.getLogger("doc_summaries")

# Co-located with chroma_db so wiping the DB also drops summaries cleanly.
DEFAULT_SUMMARY_PATH = Path("./chroma_db/summaries.json")


def _load(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read summaries cache at %s", path)
        return {}


def _save(path: Path, data: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_summary(filename: str, path: Path = DEFAULT_SUMMARY_PATH) -> str | None:
    return _load(path).get(filename)


def get_summaries_for(
    filenames: Iterable[str],
    path: Path = DEFAULT_SUMMARY_PATH,
) -> Dict[str, str]:
    data = _load(path)
    return {fn: data[fn] for fn in filenames if fn in data}


def all_summaries(path: Path = DEFAULT_SUMMARY_PATH) -> Dict[str, str]:
    return _load(path)


def set_summary(filename: str, summary: str, path: Path = DEFAULT_SUMMARY_PATH) -> None:
    data = _load(path)
    data[filename] = summary
    _save(path, data)


def remove_summary(filename: str, path: Path = DEFAULT_SUMMARY_PATH) -> None:
    data = _load(path)
    if data.pop(filename, None) is not None:
        _save(path, data)


def remove_missing(
    keep: Iterable[str],
    path: Path = DEFAULT_SUMMARY_PATH,
) -> List[str]:
    """Drop summaries for filenames not in `keep`. Returns removed names."""
    data = _load(path)
    keep_set = set(keep)
    removed = [fn for fn in list(data) if fn not in keep_set]
    if removed:
        for fn in removed:
            data.pop(fn, None)
        _save(path, data)
    return removed


def clear_all(path: Path = DEFAULT_SUMMARY_PATH) -> None:
    if path.exists():
        path.unlink()


def format_overview(summaries: Dict[str, str]) -> str:
    """Render a compact 'Documents in scope' block for the answer prompt."""
    if not summaries:
        return ""
    lines = ["Documents in scope:"]
    for fn, summ in summaries.items():
        # Each summary should already be 2-3 sentences; keep one bullet line.
        compact = " ".join(summ.split())
        lines.append(f"- {fn}: {compact}")
    return "\n".join(lines)
