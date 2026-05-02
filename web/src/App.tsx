import { useEffect, useMemo, useRef, useState } from "react"
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import { AlertTriangle, Loader2, Moon, ShieldCheck, Sun } from "lucide-react"

import { ChatPane, type ChatPaneHandle } from "@/components/ChatPane"
import { Sidebar } from "@/components/Sidebar"
import { Button } from "@/components/ui/button"
import { apiHealth } from "@/lib/api"
import { cn } from "@/lib/utils"

const qc = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5_000 },
  },
})

function HealthPill() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["health"],
    queryFn: apiHealth,
    refetchInterval: 5_000,
  })

  if (isLoading)
    return (
      <Pill icon={<Loader2 className="h-3.5 w-3.5 animate-spin" />}>
        Connecting…
      </Pill>
    )
  if (error)
    return (
      <Pill
        icon={<AlertTriangle className="h-3.5 w-3.5" />}
        variant="destructive"
      >
        Backend offline
      </Pill>
    )
  return (
    <Pill
      icon={<ShieldCheck className="h-3.5 w-3.5" />}
      variant={data?.chain_loaded ? "ready" : "loading"}
    >
      {data?.vector_count ?? 0} chunks ·{" "}
      {data?.chain_loaded ? "ready" : "warming up"}
    </Pill>
  )
}

function Pill({
  children,
  icon,
  variant = "loading",
}: {
  children: React.ReactNode
  icon?: React.ReactNode
  variant?: "loading" | "ready" | "destructive"
}) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium",
        variant === "ready" &&
          "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300",
        variant === "loading" &&
          "border-border bg-muted text-muted-foreground",
        variant === "destructive" &&
          "border-destructive/30 bg-destructive/10 text-destructive",
      )}
    >
      {icon}
      {children}
    </div>
  )
}

function Shell() {
  const [dark, setDark] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches,
  )
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  // One stable session id per browser tab. "New chat" in the sidebar
  // resets the chain's server-side memory but keeps this id.
  const sessionId = useMemo(() => {
    const k = "hushdoc-session-id"
    const existing = sessionStorage.getItem(k)
    if (existing) return existing
    const fresh = `web-${crypto.randomUUID()}`
    sessionStorage.setItem(k, fresh)
    return fresh
  }, [])

  // Lifted state: the sidebar reports the current scope; the chat consumes it.
  // useState's referential identity is stable, so passing setScope directly
  // to the sidebar avoids triggering a new effect each render.
  const [scope, setScope] = useState<string[] | null>(null)
  const chatRef = useRef<ChatPaneHandle>(null)

  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-center justify-between border-b px-5 py-2.5">
        <div className="flex items-center gap-2.5">
          <span className="text-xl">🤫</span>
          <h1 className="text-base font-semibold tracking-tight">Hushdoc</h1>
          <span className="hidden text-xs text-muted-foreground sm:inline">
            local-only PDF assistant
          </span>
        </div>
        <div className="flex items-center gap-3">
          <HealthPill />
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            onClick={() => setDark((d) => !d)}
            title={dark ? "Switch to light" : "Switch to dark"}
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
        </div>
      </header>

      <main className="flex min-h-0 flex-1">
        <Sidebar
          onClearChat={() => chatRef.current?.clear()}
          onScopeChange={setScope}
        />
        <div className="flex min-w-0 flex-1">
          <ChatPane ref={chatRef} sessionId={sessionId} scope={scope} />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <Shell />
    </QueryClientProvider>
  )
}
