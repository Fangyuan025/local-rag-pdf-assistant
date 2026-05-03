import { useCallback, useEffect, useRef, useState } from "react"

/**
 * Smart auto-scroll for chat panes.
 *
 * Auto-scrolls to bottom only when the user is already within `threshold`
 * pixels of the bottom — so manually scrolling up to re-read a previous
 * answer is respected and the chain doesn't yank you back to the latest
 * tokens.
 */
export function useStickyBottom<TScroll extends HTMLElement>(
  /** Anything whose change should trigger a re-evaluation (e.g. messages). */
  trigger: unknown,
  threshold = 80,
) {
  const scrollRef = useRef<TScroll | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const [autoFollow, setAutoFollow] = useState(true)

  // Watch the user's scroll position and toggle autoFollow.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    // Radix ScrollArea wraps the actual scroll element; find it.
    const viewport =
      (el.querySelector("[data-radix-scroll-area-viewport]") as HTMLElement) ||
      el
    const onScroll = () => {
      const distance =
        viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight
      setAutoFollow(distance < threshold)
    }
    viewport.addEventListener("scroll", onScroll)
    return () => viewport.removeEventListener("scroll", onScroll)
  }, [threshold])

  // Auto-scroll when the trigger changes AND we're allowed to.
  useEffect(() => {
    if (!autoFollow) return
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [trigger, autoFollow])

  const jumpToBottom = useCallback(() => {
    setAutoFollow(true)
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [])

  return { scrollRef, bottomRef, autoFollow, jumpToBottom }
}
