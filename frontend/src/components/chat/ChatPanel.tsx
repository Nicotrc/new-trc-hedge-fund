import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Bot, User, X, MessageSquare } from 'lucide-react'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

interface ChatPanelProps {
  defaultTicker?: string
}

export function ChatPanel({ defaultTicker = '' }: ChatPanelProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [ticker, setTicker] = useState(defaultTicker)
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100)
  }, [isOpen])

  const sendMessage = useCallback(async () => {
    const text = input.trim()
    if (!text || !ticker.trim() || loading) return

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    const assistantId = crypto.randomUUID()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '', streaming: true }

    setMessages((prev) => [...prev, userMsg, assistantMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/v1/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: ticker.trim().toUpperCase(), message: text }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          for (const line of chunk.split('\n')) {
            if (!line.startsWith('data: ')) continue
            const payload = line.slice(6)
            if (payload === '[DONE]') break
            try {
              const { text: delta } = JSON.parse(payload) as { text: string }
              accumulated += delta
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, content: accumulated } : m
                )
              )
            } catch { /* skip malformed */ }
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: 'Error: could not reach the server.', streaming: false }
            : m
        )
      )
    } finally {
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId ? { ...m, streaming: false } : m)
      )
      setLoading(false)
    }
  }, [input, ticker, loading])

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void sendMessage() }
  }

  return (
    <>
      {/* Floating button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-50 flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 shadow-lg hover:bg-blue-500 transition-colors"
          title="Ask AI about a ticker"
        >
          <MessageSquare className="h-5 w-5 text-white" />
        </button>
      )}

      {/* Chat drawer */}
      {isOpen && (
        <div className="fixed bottom-0 right-0 z-50 flex h-[540px] w-[400px] flex-col rounded-tl-xl border border-zinc-700 bg-zinc-950 shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
            <div className="flex items-center gap-2">
              <Bot className="h-4 w-4 text-blue-400" />
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-zinc-300">
                Analyst AI
              </span>
            </div>
            <div className="flex items-center gap-2">
              {/* Ticker input */}
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="TICKER"
                maxLength={8}
                className="w-20 rounded border border-zinc-700 bg-zinc-900 px-2 py-1 font-mono text-xs text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
              />
              <button onClick={() => setIsOpen(false)} className="text-zinc-500 hover:text-zinc-300">
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
                <Bot className="h-8 w-8 text-zinc-700" />
                <p className="text-xs text-zinc-500">
                  Inserisci un ticker e fai una domanda.<br />
                  Es: <span className="text-zinc-400 italic">"Qual è il sentiment degli agenti su NVDA?"</span>
                </p>
              </div>
            )}
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div className="mt-0.5 shrink-0">
                  {msg.role === 'user'
                    ? <User className="h-4 w-4 text-zinc-500" />
                    : <Bot className="h-4 w-4 text-blue-400" />
                  }
                </div>
                <div
                  className={`max-w-[85%] rounded-lg px-3 py-2 text-xs leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-blue-900/40 text-zinc-200'
                      : 'bg-zinc-900 text-zinc-300'
                  }`}
                >
                  {msg.content || (msg.streaming && (
                    <span className="animate-pulse text-zinc-500">●●●</span>
                  ))}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="border-t border-zinc-800 px-3 py-3">
            <div className="flex items-center gap-2">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder={ticker ? `Domanda su ${ticker}…` : 'Prima inserisci un ticker…'}
                disabled={loading || !ticker.trim()}
                className="flex-1 rounded border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-100 placeholder-zinc-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              />
              <button
                onClick={() => void sendMessage()}
                disabled={loading || !input.trim() || !ticker.trim()}
                className="flex h-8 w-8 items-center justify-center rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-40 transition-colors"
              >
                <Send className="h-3.5 w-3.5 text-white" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
