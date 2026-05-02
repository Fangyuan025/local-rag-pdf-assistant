import { useCallback, useEffect, useRef, useState } from "react"
import {
  CheckCircle2,
  FileText,
  Loader2,
  Upload,
  X,
  XCircle,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { useDocuments } from "@/hooks/useDocuments"
import { cn } from "@/lib/utils"

const ACCEPT = [
  ".pdf",
  ".docx",
  ".jpg",
  ".jpeg",
  ".png",
  ".tif",
  ".tiff",
  ".bmp",
]
const ACCEPT_ATTR = ACCEPT.join(",")

export function DocumentUpload() {
  const [queue, setQueue] = useState<File[]>([])
  const [replace, setReplace] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const { upload, uploading, uploadError, progress, dismissProgress } =
    useDocuments()

  // Auto-clear the queue once an upload finishes successfully so the
  // sidebar collapses back to the dropzone idle state.
  useEffect(() => {
    if (progress.done && !uploading) {
      const t = setTimeout(() => {
        setQueue([])
        dismissProgress()
      }, 1500)
      return () => clearTimeout(t)
    }
  }, [progress.done, uploading, dismissProgress])

  const addFiles = useCallback((files: FileList | File[]) => {
    const ok = Array.from(files).filter((f) =>
      ACCEPT.some((ext) => f.name.toLowerCase().endsWith(ext)),
    )
    setQueue((prev) => {
      const seen = new Set(prev.map((f) => f.name))
      return [...prev, ...ok.filter((f) => !seen.has(f.name))]
    })
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files)
    },
    [addFiles],
  )

  const removeFromQueue = (name: string) =>
    setQueue((prev) => prev.filter((f) => f.name !== name))

  const onSubmit = () => {
    if (queue.length === 0) return
    void upload(queue, replace)
  }

  const showProgress = progress.files.length > 0

  return (
    <div className="space-y-2">
      {/* Dropzone */}
      <label
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-md border border-dashed px-3 py-4 text-center text-xs transition-colors",
          dragOver
            ? "border-primary/50 bg-primary/5 text-foreground"
            : "border-border bg-muted/30 text-muted-foreground hover:bg-muted/50",
        )}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files)
            e.currentTarget.value = "" // allow re-adding the same file
          }}
        />
        <Upload className="h-4 w-4" />
        <div>
          <span className="font-medium text-foreground">Drop</span> PDFs,
          DOCX, or photos
        </div>
        <div className="text-[10px] text-muted-foreground/70">
          PDF · DOCX · JPG · PNG · TIFF
        </div>
      </label>

      {/* Replace toggle */}
      <label className="flex cursor-pointer items-start gap-2 px-1 text-[11px] leading-snug text-muted-foreground">
        <input
          type="checkbox"
          checked={replace}
          onChange={(e) => setReplace(e.target.checked)}
          className="mt-0.5 h-3.5 w-3.5 accent-primary"
        />
        <span>
          <span className="font-medium text-foreground">Replace</span> existing
          index on upload
        </span>
      </label>

      {/* Queue list (pre-upload) */}
      {!showProgress && queue.length > 0 && (
        <ul className="space-y-1">
          {queue.map((f) => (
            <li
              key={f.name}
              className="flex items-center gap-2 rounded-md border bg-card px-2 py-1.5 text-xs"
            >
              <FileText className="h-3 w-3 shrink-0 text-muted-foreground" />
              <span className="truncate">{f.name}</span>
              <span className="ml-auto shrink-0 text-[10px] text-muted-foreground">
                {(f.size / 1024).toFixed(0)} KB
              </span>
              <button
                type="button"
                onClick={() => removeFromQueue(f.name)}
                className="text-muted-foreground hover:text-destructive"
                title="Remove"
              >
                <X className="h-3 w-3" />
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Progress list (during/after upload) */}
      {showProgress && (
        <ul className="space-y-1">
          {progress.files.map((f) => (
            <li
              key={f.filename}
              className="flex items-start gap-2 rounded-md border bg-card px-2 py-1.5 text-xs"
            >
              <ProgressIcon status={f.status} />
              <div className="min-w-0 flex-1">
                <div className="truncate">{f.filename}</div>
                {f.status === "indexed" && (
                  <div className="text-[10px] text-muted-foreground">
                    {f.chunks} chunks indexed
                  </div>
                )}
                {f.status === "error" && (
                  <div className="text-[10px] text-destructive">{f.error}</div>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}

      {uploadError && (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1.5 text-xs text-destructive">
          {uploadError}
        </div>
      )}

      <Button
        size="sm"
        onClick={onSubmit}
        disabled={uploading || queue.length === 0}
        className="w-full"
      >
        {uploading ? (
          <>
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Indexing…
          </>
        ) : (
          <>
            <Upload className="h-3.5 w-3.5" />
            Ingest &amp; index ({queue.length || 0})
          </>
        )}
      </Button>
    </div>
  )
}

function ProgressIcon({ status }: { status: string }) {
  if (status === "indexed")
    return <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-emerald-500" />
  if (status === "error")
    return <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-destructive" />
  if (status === "uploading" || status === "queued")
    return (
      <Loader2 className="mt-0.5 h-3 w-3 shrink-0 animate-spin text-muted-foreground" />
    )
  return <FileText className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
}
