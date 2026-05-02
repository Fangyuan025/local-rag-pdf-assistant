import { useEffect, useState } from "react"
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query"
import { Loader2, ShieldCheck, AlertTriangle } from "lucide-react"

import { apiHealth } from "@/lib/api"
import { cn } from "@/lib/utils"

// One QueryClient for the whole app — TanStack Query handles caching,
// retries, and background refetch for the few JSON endpoints we have.
const qc = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5_000 },
  },
})

/**
 * Phase-3 placeholder shell. Just proves the React → Vite proxy →
 * FastAPI plumbing works end to end. Real chat / sidebar / voice land
 * in P4-P6.
 */
function HealthBanner() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["health"],
    queryFn: apiHealth,
    refetchInterval: 5_000,
  })

  if (isLoading)
    return (
      <Pill icon={<Loader2 className="h-3.5 w-3.5 animate-spin" />}>
        Connecting to backend…
      </Pill>
    )
  if (error)
    return (
      <Pill
        icon={<AlertTriangle className="h-3.5 w-3.5" />}
        variant="destructive"
      >
        Backend unreachable — start uvicorn on :8000
      </Pill>
    )
  return (
    <Pill
      icon={<ShieldCheck className="h-3.5 w-3.5" />}
      variant={data?.chain_loaded ? "ready" : "loading"}
    >
      Backend OK · {data?.vector_count} chunks ·{" "}
      {data?.chain_loaded ? "chain ready" : "chain idle"}
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
  const [dark, setDark] = useState(() =>
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  )
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
  }, [dark])

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b px-6 py-3">
        <div className="flex items-center gap-2">
          <span className="text-xl">🤫</span>
          <h1 className="text-base font-semibold tracking-tight">Hushdoc</h1>
          <span className="hidden text-xs text-muted-foreground sm:inline">
            local-only PDF assistant
          </span>
        </div>
        <div className="flex items-center gap-3">
          <HealthBanner />
          <button
            type="button"
            onClick={() => setDark((d) => !d)}
            className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
            title="Toggle theme"
          >
            {dark ? "☀ Light" : "🌙 Dark"}
          </button>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 md:grid-cols-[16rem_1fr]">
        <aside className="hidden border-r p-4 text-sm md:block">
          <p className="text-muted-foreground">
            Sidebar (documents · scope · voice) lands in P5–P6.
          </p>
        </aside>

        <section className="flex flex-col items-center justify-center p-8">
          <div className="max-w-md space-y-3 text-center">
            <h2 className="text-2xl font-semibold tracking-tight">
              React skeleton wired up.
            </h2>
            <p className="text-sm text-muted-foreground">
              The pill in the top-right is hitting{" "}
              <code className="rounded bg-muted px-1 py-0.5 text-xs">
                /api/health
              </code>{" "}
              every 5 seconds through the Vite proxy. Chat lands in P4.
            </p>
          </div>
        </section>
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
