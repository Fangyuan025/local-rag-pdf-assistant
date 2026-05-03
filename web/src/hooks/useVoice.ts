import { useCallback, useEffect, useRef, useState } from "react"
import { toast } from "sonner"

import { apiSynthesize, apiTranscribe } from "@/lib/api"
import { isEnglish } from "@/lib/audio"
import { recordWithVAD } from "@/lib/vad"

const STORAGE_KEY = "hushdoc-voice-mode"

export type VoiceState = "idle" | "recording" | "processing"

/**
 * Top-level voice-mode coordination.
 *
 * `enabled` is the user-controlled toggle (off by default, persisted in
 * sessionStorage so it survives a soft reload).
 *
 * `record()` opens the mic, runs the VAD loop, posts the resulting WAV to
 * /api/voice/transcribe, and returns the recognised English text. The
 * caller (ChatInput) is expected to send that text as the next chat turn
 * immediately so the user "talks then waits" instead of "talks then
 * hits send".
 *
 * `synthesizeAndPlay(text)` posts the assistant answer to
 * /api/voice/synthesize and pipes the WAV into a hidden global <audio>
 * element so the answer auto-plays without any visible progress bar.
 * Returns the blob URL so the message can cache it for replay.
 */
export function useVoice() {
  const [enabled, setEnabled] = useState<boolean>(() => {
    try {
      return sessionStorage.getItem(STORAGE_KEY) === "true"
    } catch {
      return false
    }
  })
  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, String(enabled))
    } catch {
      /* ignore */
    }
  }, [enabled])

  const [state, setState] = useState<VoiceState>("idle")
  const [error, setError] = useState<string | null>(null)
  const [level, setLevel] = useState(0) // last RMS for the mic-pulse meter
  const abortRef = useRef<AbortController | null>(null)

  const cancel = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setState("idle")
    setLevel(0)
  }, [])

  const record = useCallback(async (): Promise<string | null> => {
    if (state !== "idle") return null
    setError(null)
    setState("recording")
    setLevel(0)
    const ac = new AbortController()
    abortRef.current = ac
    try {
      const wav = await recordWithVAD(ac.signal, {
        onLevel: (rms) => setLevel(rms),
      })
      setState("processing")
      const text = await apiTranscribe(wav)
      return text || null
    } catch (err) {
      if ((err as Error).name === "AbortError") return null
      const msg = err instanceof Error ? err.message : String(err)
      setError(msg)
      // Common case: user denied mic permission. Surface a friendlier note.
      if (/permission|denied|NotAllowed/i.test(msg)) {
        toast.error(
          "Microphone access denied. Allow it in your browser to use voice mode.",
        )
      } else if (/no speech/i.test(msg)) {
        toast.warning("Didn't hear anything — try again?")
      } else {
        toast.error(`Voice input failed: ${msg}`)
      }
      return null
    } finally {
      setState("idle")
      setLevel(0)
      abortRef.current = null
    }
  }, [state])

  // Hidden audio element used for autoplay. Created once, reused.
  const playerRef = useRef<HTMLAudioElement | null>(null)
  if (typeof document !== "undefined" && !playerRef.current) {
    const el = document.createElement("audio")
    el.style.display = "none"
    document.body.appendChild(el)
    playerRef.current = el
  }

  const playUrl = useCallback((url: string) => {
    const el = playerRef.current
    if (!el) return
    el.src = url
    el.currentTime = 0
    void el.play().catch(() => undefined)
  }, [])

  const synthesizeAndPlay = useCallback(
    async (text: string): Promise<string | null> => {
      if (!enabled || !text || !isEnglish(text)) return null
      try {
        const wav = await apiSynthesize(text)
        const url = URL.createObjectURL(wav)
        playUrl(url)
        return url
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        setError(msg)
        toast.error(`TTS failed: ${msg}`)
        return null
      }
    },
    [enabled, playUrl],
  )

  return {
    enabled,
    setEnabled,
    state,
    level,
    error,
    record,
    cancel,
    synthesizeAndPlay,
    playUrl,
  }
}
