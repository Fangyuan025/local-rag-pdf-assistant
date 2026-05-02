import { forwardRef, useEffect, useImperativeHandle, useRef } from "react"
import { Sparkles } from "lucide-react"

import { ScrollArea } from "@/components/ui/scroll-area"
import { useChat } from "@/hooks/useChat"

import { ChatInput } from "./ChatInput"
import { ChatMessage } from "./ChatMessage"

interface ChatPaneProps {
  sessionId: string
  scope?: string[] | null
}

export interface ChatPaneHandle {
  clear: () => void
}

export const ChatPane = forwardRef<ChatPaneHandle, ChatPaneProps>(
  function ChatPane({ sessionId, scope }, ref) {
    const { messages, send, stop, clear, streaming, error } = useChat({
      sessionId,
      scope,
    })
    const bottomRef = useRef<HTMLDivElement>(null)

    // Pin the scrollbar to bottom while streaming so new tokens stay in view.
    useEffect(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
    }, [messages])

    useImperativeHandle(ref, () => ({ clear }), [clear])

    return (
      <div className="flex h-full min-h-0 flex-1 flex-col">
        <ScrollArea className="flex-1">
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6">
            {messages.length === 0 ? (
              <EmptyState />
            ) : (
              messages.map((msg) => <ChatMessage key={msg.id} msg={msg} />)
            )}
            {error && (
              <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {error}
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </ScrollArea>
        <ChatInput streaming={streaming} onSend={send} onStop={stop} />
      </div>
    )
  },
)

function EmptyState() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-3 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Sparkles className="h-6 w-6" />
      </div>
      <h2 className="text-xl font-semibold tracking-tight">
        Ask anything about your PDFs.
      </h2>
      <p className="max-w-md text-sm text-muted-foreground">
        Upload a PDF, DOCX, or document photo from the sidebar, then start
        asking. Answers stream in with inline source citations. Everything
        runs on your machine — nothing leaves it.
      </p>
    </div>
  )
}
