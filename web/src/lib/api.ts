/**
 * Tiny typed wrapper around the FastAPI backend.
 *
 * All paths go through the Vite dev proxy at /api/* (see vite.config.ts),
 * so in dev the browser stays on a single origin and in prod we'd serve
 * the built React bundle from FastAPI itself.
 *
 * Streaming endpoints (chat, upload) bypass these helpers and use
 * fetch + ReadableStream directly inside the matching React hook —
 * EventSource doesn't support POST bodies, so we hand-parse SSE frames.
 */

import type {
  DocumentsResponse,
  HealthResponse,
} from "@/types"

const BASE = "/api"

async function jsonGet<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json() as Promise<T>
}

async function jsonSend<T>(
  path: string,
  body: unknown,
  method: "POST" | "PUT" | "DELETE" = "POST",
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => "")
    throw new Error(`${path} -> ${res.status} ${detail}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Health & documents
// ---------------------------------------------------------------------------
export const apiHealth = () => jsonGet<HealthResponse>("/health")
export const apiListDocuments = () => jsonGet<DocumentsResponse>("/documents")
export const apiDeleteDocuments = () =>
  jsonSend<{ ok: boolean; was_count: number }>("/documents", undefined, "DELETE")

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
export const apiClearChat = (sessionId: string) =>
  jsonSend<{ ok: boolean }>("/chat/clear", { session_id: sessionId })

// ---------------------------------------------------------------------------
// Voice
// ---------------------------------------------------------------------------
export const apiVoiceHealth = () =>
  jsonGet<{ whisper_ready: boolean; kokoro_ready: boolean }>("/voice/health")

export async function apiTranscribe(audio: Blob): Promise<string> {
  const fd = new FormData()
  fd.append("audio", audio, "speech.wav")
  const res = await fetch(`${BASE}/voice/transcribe`, {
    method: "POST",
    body: fd,
  })
  if (!res.ok) throw new Error(`transcribe -> ${res.status}`)
  const json = (await res.json()) as { text: string }
  return json.text
}

export async function apiSynthesize(text: string): Promise<Blob> {
  const res = await fetch(`${BASE}/voice/synthesize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`synthesize -> ${res.status}`)
  return res.blob()
}
