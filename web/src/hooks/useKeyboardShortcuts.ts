import { useEffect } from "react"

interface ShortcutOptions {
  /** Cmd/Ctrl + K — focus the chat input. */
  onFocusInput?: () => void
  /** Cmd/Ctrl + L — clear the current chat. */
  onClearChat?: () => void
  /** Esc — cancel anything in flight (streaming, recording, processing). */
  onEscape?: () => void
}

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/i.test(navigator.platform)

const inEditableField = (el: EventTarget | null): boolean => {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName
  if (tag === "INPUT" || tag === "TEXTAREA") return true
  if (el.isContentEditable) return true
  return false
}

/** Global keyboard shortcuts. Cmd on mac, Ctrl elsewhere. */
export function useKeyboardShortcuts(opts: ShortcutOptions) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = isMac ? e.metaKey : e.ctrlKey

      // Cmd/Ctrl+K — focus input. Works even from inside another input,
      // since the convention is "jump back to chat".
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault()
        opts.onFocusInput?.()
        return
      }
      // Cmd/Ctrl+L — clear chat. Skip when inside another text field so we
      // don't blow away whatever the user was about to type.
      if (mod && e.key.toLowerCase() === "l" && !inEditableField(e.target)) {
        e.preventDefault()
        opts.onClearChat?.()
        return
      }
      // Esc — cancel anything in flight.
      if (e.key === "Escape") {
        opts.onEscape?.()
        return
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [opts])
}
