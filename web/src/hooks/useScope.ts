import { useCallback, useEffect, useMemo, useState } from "react"

const STORAGE_KEY = "hushdoc-scope"

/**
 * Per-session scope: which indexed filenames the next chat turn will
 * search across. Defaults to "all selected" — when scope === indexed,
 * the chat hook passes null upstream so the backend can skip the
 * filename filter entirely.
 *
 * Persisted in sessionStorage so the selection survives a soft reload
 * but resets on a new browser tab.
 */
export function useScope(indexed: string[]) {
  const [selected, setSelected] = useState<string[]>(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY)
      if (raw) return JSON.parse(raw) as string[]
    } catch {
      /* ignore */
    }
    return []
  })

  // Drop selections that no longer exist; default to ALL when unset.
  // Depend on a stable string key so a new array reference with the same
  // contents (TanStack Query returns fresh refs on every refetch) doesn't
  // re-fire the effect and trigger an infinite render loop.
  const indexedKey = indexed.slice().sort().join("|")
  useEffect(() => {
    if (indexed.length === 0) {
      setSelected((prev) => (prev.length === 0 ? prev : []))
      return
    }
    setSelected((prev) => {
      const stillThere = prev.filter((f) => indexed.includes(f))
      if (stillThere.length === 0) return [...indexed]
      // Avoid emitting a new array reference if nothing actually changed.
      if (
        stillThere.length === prev.length &&
        stillThere.every((f, i) => f === prev[i])
      ) {
        return prev
      }
      return stillThere
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexedKey])

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(selected))
    } catch {
      /* ignore */
    }
  }, [selected])

  const toggle = useCallback((filename: string) => {
    setSelected((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename],
    )
  }, [])

  const selectAll = useCallback(() => setSelected([...indexed]), [indexed])
  const selectNone = useCallback(() => setSelected([]), [])

  // The value we send to the chain: null when "search everything"
  // (either none selected or all selected), otherwise the explicit list.
  const effectiveScope = useMemo<string[] | null>(() => {
    if (selected.length === 0) return null
    if (selected.length === indexed.length) return null
    return selected
  }, [selected, indexed])

  const allSelected = selected.length === indexed.length && indexed.length > 0

  return {
    selected,
    effectiveScope,
    toggle,
    selectAll,
    selectNone,
    allSelected,
  }
}
