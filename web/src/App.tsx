import { useEffect, useMemo, useRef, useState } from "react"
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query"
import {
  AlertTriangle,
  Loader2,
  Menu,
  Moon,
  ShieldCheck,
  Sun,
} from "lucide-react"
import { Toaster } from "sonner"

import { ChatPane, type ChatPaneHandle } from "@/components/ChatPane"
import { Sidebar, SidebarContent } from "@/components/Sidebar"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts"
import { useVoice } from "@/hooks/useVoice"
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

  // One stable session id per browser tab.
  const sessionId = useMemo(() => {
    const k = "hushdoc-session-id"
    const existing = sessionStorage.getItem(k)
    if (existing) return existing
    const fresh = `web-${crypto.randomUUID()}`
    sessionStorage.setItem(k, fresh)
    return fresh
  }, [])

  // Lifted state.
  const [scope, setScope] = useState<string[] | null>(null)
  const chatRef = useRef<ChatPaneHandle>(null)
  const voice = useVoice()
  const [drawerOpen, setDrawerOpen] = useState(false)

  // Global keyboard shortcuts.
  useKeyboardShortcuts({
    onFocusInput: () => chatRef.current?.focusInput(),
    onClearChat: () => chatRef.current?.clear(),
    onEscape: () => chatRef.current?.cancel(),
  })

  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-center justify-between gap-2 border-b px-4 py-2.5 sm:px-5">
        <div className="flex min-w-0 items-center gap-2.5">
          {/* Hamburger — mobile only */}
          <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
            <SheetTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="md:hidden"
              >
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="p-0">
              <SidebarContent
                onClearChat={() => {
                  chatRef.current?.clear()
                  setDrawerOpen(false)
                }}
                onScopeChange={setScope}
                voice={voice}
              />
            </SheetContent>
          </Sheet>

          <span className="text-xl">🤫</span>
          <h1 className="text-base font-semibold tracking-tight">Hushdoc</h1>
          <span className="hidden truncate text-xs text-muted-foreground sm:inline">
            local-only PDF assistant
          </span>
        </div>
        <div className="flex shrink-0 items-center gap-2 sm:gap-3">
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
          voice={voice}
        />
        <div className="flex min-w-0 flex-1">
          <ChatPane
            ref={chatRef}
            sessionId={sessionId}
            scope={scope}
            voice={voice}
          />
        </div>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <Shell />
      <Toaster
        position="bottom-right"
        toastOptions={{
          classNames: {
            toast:
              "border bg-background text-foreground shadow-md rounded-md text-sm",
          },
        }}
      />
    </QueryClientProvider>
  )
}
