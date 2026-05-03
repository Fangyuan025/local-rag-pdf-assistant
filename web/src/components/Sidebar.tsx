import { useEffect, useState } from "react"
import {
  Folder,
  Loader2,
  MessageSquarePlus,
  Mic,
  Search,
  Trash2,
} from "lucide-react"

import { DocumentUpload } from "@/components/DocumentUpload"
import { ScopeSelector } from "@/components/ScopeSelector"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Switch } from "@/components/ui/switch"
import { useDocuments } from "@/hooks/useDocuments"
import { useScope } from "@/hooks/useScope"
import type { useVoice } from "@/hooks/useVoice"

export interface SidebarProps {
  onClearChat: () => void
  onScopeChange: (scope: string[] | null) => void
  voice: ReturnType<typeof useVoice>
}

/** Inner content — shared between desktop sidebar and mobile drawer. */
export function SidebarContent({
  onClearChat,
  onScopeChange,
  voice,
}: SidebarProps) {
  const { list, del } = useDocuments()
  const indexed = list.data?.filenames ?? []
  const summaries = list.data?.summaries ?? {}
  const scope = useScope(indexed)

  // Mirror the effective scope upward whenever it changes.
  useEffect(() => {
    onScopeChange(scope.effectiveScope)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(scope.effectiveScope)])

  const [confirming, setConfirming] = useState(false)

  return (
    <div className="flex h-full min-h-0 w-full flex-col">
      <ScrollArea className="flex-1">
        <div className="space-y-5 p-4">
          {/* Documents */}
          <Section
            icon={<Folder className="h-3.5 w-3.5" />}
            title="Documents"
            badge={
              list.data
                ? `${list.data.chunk_count} chunks`
                : list.isLoading
                  ? "…"
                  : ""
            }
          >
            <DocumentUpload />
          </Section>

          {/* Search scope */}
          <Section
            icon={<Search className="h-3.5 w-3.5" />}
            title="Search scope"
          >
            {list.isLoading ? (
              <ScopeSkeleton />
            ) : (
              <ScopeSelector
                indexed={indexed}
                selected={scope.selected}
                summaries={summaries}
                allSelected={scope.allSelected}
                onToggle={scope.toggle}
                onSelectAll={scope.selectAll}
                onSelectNone={scope.selectNone}
              />
            )}
          </Section>

          {/* Voice */}
          <Section icon={<Mic className="h-3.5 w-3.5" />} title="Voice">
            <label className="flex cursor-pointer items-center justify-between gap-2 rounded-md border bg-card px-2.5 py-2 text-xs">
              <span>Voice mode</span>
              <Switch
                checked={voice.enabled}
                onCheckedChange={(v) => voice.setEnabled(v)}
              />
            </label>
            {voice.enabled && (
              <p className="px-1 text-[11px] leading-snug text-muted-foreground">
                🌐 English only — Whisper-base.en in, Kokoro-82M out. Mic
                appears beside the chat input; auto-stops after 1.5 s of
                silence.
              </p>
            )}
          </Section>
        </div>
      </ScrollArea>

      {/* Sticky bottom: chat + danger actions */}
      <div className="space-y-2 border-t bg-card/50 p-3">
        <Button
          size="sm"
          variant="secondary"
          className="w-full"
          onClick={onClearChat}
          title="Cmd/Ctrl + L"
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />
          New chat
        </Button>
        {indexed.length > 0 &&
          (confirming ? (
            <div className="flex gap-1">
              <Button
                size="sm"
                variant="destructive"
                className="flex-1"
                onClick={() => {
                  del.mutate()
                  setConfirming(false)
                }}
              >
                {del.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Confirm wipe
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setConfirming(false)}
              >
                Cancel
              </Button>
            </div>
          ) : (
            <Button
              size="sm"
              variant="ghost"
              className="w-full text-muted-foreground hover:text-destructive"
              onClick={() => setConfirming(true)}
            >
              <Trash2 className="h-3.5 w-3.5" />
              Clear all documents
            </Button>
          ))}
      </div>
    </div>
  )
}

/** Desktop variant — fixed-width left rail, hidden below md. */
export function Sidebar(props: SidebarProps) {
  return (
    <aside className="hidden w-64 shrink-0 border-r md:block">
      <SidebarContent {...props} />
    </aside>
  )
}

function Section({
  icon,
  title,
  badge,
  children,
}: {
  icon: React.ReactNode
  title: string
  badge?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {icon}
        {title}
        {badge && (
          <span className="ml-auto rounded bg-muted px-1.5 py-0.5 text-[10px] font-normal lowercase tracking-normal text-muted-foreground/80">
            {badge}
          </span>
        )}
      </div>
      {children}
    </div>
  )
}

function ScopeSkeleton() {
  return (
    <ul className="space-y-1.5">
      {[0, 1, 2].map((i) => (
        <li
          key={i}
          className="h-6 animate-pulse rounded-md bg-muted/60"
          style={{ animationDelay: `${i * 80}ms` }}
        />
      ))}
    </ul>
  )
}
