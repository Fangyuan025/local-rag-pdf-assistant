import { CheckSquare, FileText, Square } from "lucide-react"

import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface ScopeSelectorProps {
  indexed: string[]
  selected: string[]
  summaries: Record<string, string>
  allSelected: boolean
  onToggle: (filename: string) => void
  onSelectAll: () => void
  onSelectNone: () => void
}

export function ScopeSelector({
  indexed,
  selected,
  summaries,
  allSelected,
  onToggle,
  onSelectAll,
  onSelectNone,
}: ScopeSelectorProps) {
  if (indexed.length === 0) {
    return (
      <p className="rounded-md border border-dashed bg-muted/30 px-2.5 py-2 text-[11px] text-muted-foreground">
        Upload a document above to populate this list.
      </p>
    )
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-muted-foreground">
          {selected.length}/{indexed.length} in scope
        </span>
        <button
          type="button"
          onClick={allSelected ? onSelectNone : onSelectAll}
          className="text-[11px] text-muted-foreground hover:text-foreground"
        >
          {allSelected ? "Select none" : "Select all"}
        </button>
      </div>

      <TooltipProvider delayDuration={300}>
        <ul className="space-y-0.5">
          {indexed.map((fn) => {
            const checked = selected.includes(fn)
            const summary = summaries[fn]
            return (
              <li key={fn}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={() => onToggle(fn)}
                      className={cn(
                        "flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-xs transition-colors",
                        "hover:bg-accent hover:text-accent-foreground",
                        !checked && "text-muted-foreground",
                      )}
                    >
                      {checked ? (
                        <CheckSquare className="h-3.5 w-3.5 shrink-0 text-primary" />
                      ) : (
                        <Square className="h-3.5 w-3.5 shrink-0" />
                      )}
                      <FileText className="h-3 w-3 shrink-0 opacity-70" />
                      <span className="truncate">{fn}</span>
                    </button>
                  </TooltipTrigger>
                  {summary && (
                    <TooltipContent side="right" className="max-w-sm">
                      {summary}
                    </TooltipContent>
                  )}
                </Tooltip>
              </li>
            )
          })}
        </ul>
      </TooltipProvider>
    </div>
  )
}
