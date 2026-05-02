/** API + chat types shared across hooks and components. */

export interface SourceDoc {
  filename: string
  page: number | null
  headings: string
  snippet: string
}

export interface HealthResponse {
  ok: boolean
  chain_loaded: boolean
  store_loaded: boolean
  vector_count: number
  indexed_files: string[]
}

export interface DocumentsResponse {
  filenames: string[]
  chunk_count: number
  summaries: Record<string, string>
}

export interface DoneEvent {
  question: string
  standalone_question: string
  answer: string
  source_documents: SourceDoc[]
  chitchat: boolean
  scope: string[] | null
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  /** True while the assistant message is still being streamed. */
  streaming?: boolean
  /** Set to true when the chain short-circuited the chitchat path. */
  chitchat?: boolean
  /** Final cited sources (filtered to those mentioned in the answer). */
  sources?: SourceDoc[]
  /** Standalone search query that drove retrieval (debug). */
  standaloneQuery?: string
  /** Cached TTS audio for the replay icon, if voice mode synthesised one. */
  audioUrl?: string
}
