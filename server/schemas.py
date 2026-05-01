"""Pydantic models for the Hushdoc HTTP API.

Kept deliberately small — the SSE chat stream uses dicts directly, but
everything else flows through these typed schemas so the OpenAPI spec
served at ``/docs`` is useful and the React client can codegen types
later if we want.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    ok: bool
    chain_loaded: bool
    store_loaded: bool
    vector_count: int
    indexed_files: List[str]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
class DocumentsResponse(BaseModel):
    filenames: List[str]
    chunk_count: int
    summaries: Dict[str, str]


class DeleteDocumentsResponse(BaseModel):
    ok: bool
    was_count: int


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    filenames: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to these source files. None / empty "
                    "means search the whole vector store.",
    )


class ChatClearRequest(BaseModel):
    session_id: str = "default"


class ChatClearResponse(BaseModel):
    ok: bool


# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------
class VoiceTranscribeResponse(BaseModel):
    text: str


class VoiceSynthesizeRequest(BaseModel):
    text: str


class VoiceHealthResponse(BaseModel):
    whisper_ready: bool
    kokoro_ready: bool
