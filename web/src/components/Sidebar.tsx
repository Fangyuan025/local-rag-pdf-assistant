import { useEffect, useState } from "react"
import {
  Folder,
  Loader2,
  MessageSquarePlus,
  Search,
  Trash2,
} from "lucide-react"

import { DocumentUpload } from "@/components/DocumentUpload"
import { ScopeSelector } from "@/components/ScopeSelector"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useDocuments } from "@/hooks/useDocuments"
import { useScope } from "@/hooks/useScope"

interface SidebarProps {
  onClearChat: () => void
  onScopeChange: (scope: string[] | null) => void
}

export function Sidebar({ onClearChat, onScopeChange }: SidebarProps) {
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
    <aside className="hidden w-64 shrink-0 border-r md:flex md:flex-col">
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
            <ScopeSelector
              indexed={indexed}
              selected={scope.selected}
              summaries={summaries}
              allSelected={scope.allSelected}
              onToggle={scope.toggle}
              onSelectAll={scope.selectAll}
              onSelectNone={scope.selectNone}
            />
          </Section>
        </div>
      </ScrollArea>

      {/* Sticky bottom: chat + danger actions */}
      <div className="border-t bg-card/50 p-3 space-y-2">
        <Button
          size="sm"
          variant="secondary"
          className="w-full"
          onClick={onClearChat}
        >
          <MessageSquarePlus className="h-3.5 w-3.5" />
          New chat
        </Button>
        {indexed.length > 0 && (
          <>
            {confirming ? (
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
            )}
          </>
        )}
      </div>
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

