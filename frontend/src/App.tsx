import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Menu, MessageCircle, X } from 'lucide-react'
import { Sidebar } from './components/Sidebar'
import { DisclaimerBanner } from './components/DisclaimerBanner'
import { QuickPrompts } from './components/QuickPrompts'
import { ChatMessageView, TypingIndicator } from './components/ChatMessage'
import { Composer } from './components/Composer'
import { QUICK_PROMPTS, sendChatMessage } from './services/chatService'
import {
  createEmptyConversation,
  createWelcomeConversation,
  titleFromMessage,
} from './lib/chatHelpers'
import { createId, loadConversations, saveConversations } from './lib/storage'
import type { ChatMessage, Conversation, FeedbackValue } from './types/chat'

function usePersistedConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const stored = loadConversations()
    if (stored && stored.length > 0) return stored
    return [createWelcomeConversation()]
  })

  useEffect(() => {
    saveConversations(conversations)
  }, [conversations])

  return [conversations, setConversations] as const
}

export default function App() {
  const [conversations, setConversations] = usePersistedConversations()
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? '')
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [lastFollowUps, setLastFollowUps] = useState<string[]>([
    ...QUICK_PROMPTS.slice(0, 3),
  ])
  const scrollRef = useRef<HTMLDivElement>(null)

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0] ?? null,
    [conversations, activeId],
  )

  useEffect(() => {
    if (!active && conversations[0]) {
      setActiveId(conversations[0].id)
    }
  }, [active, conversations])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [active?.messages, busy])

  const updateConversation = useCallback(
    (id: string, updater: (conv: Conversation) => Conversation) => {
      setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)))
    },
    [setConversations],
  )

  const handleNewChat = () => {
    const conv = createEmptyConversation()
    setConversations((prev) => [conv, ...prev])
    setActiveId(conv.id)
    setLastFollowUps([...QUICK_PROMPTS.slice(0, 3)])
    setDraft('')
    setSidebarOpen(false)
  }

  const handleSelect = (id: string) => {
    setActiveId(id)
    setSidebarOpen(false)
  }

  const handleDelete = (id: string) => {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id)
      if (next.length === 0) {
        const welcome = createWelcomeConversation()
        setActiveId(welcome.id)
        return [welcome]
      }
      if (id === activeId) {
        setActiveId(next[0].id)
      }
      return next
    })
  }

  const handleRename = (id: string) => {
    const current = conversations.find((c) => c.id === id)
    if (!current) return
    const next = window.prompt('Sohbet başlığı', current.title)
    if (!next || !next.trim()) return
    updateConversation(id, (c) => ({
      ...c,
      title: next.trim().slice(0, 80),
      updatedAt: new Date().toISOString(),
    }))
  }

  const handleClearAll = () => {
    if (!window.confirm('Tüm sohbet geçmişi silinsin mi?')) return
    const welcome = createWelcomeConversation()
    setConversations([welcome])
    setActiveId(welcome.id)
    setLastFollowUps([...QUICK_PROMPTS.slice(0, 3)])
  }

  const sendMessage = async (rawText: string, options?: { regenerateOf?: string }) => {
    const text = rawText.trim()
    if (!text || busy || !active) return

    const conversationId = active.id
    const now = new Date().toISOString()
    const userMessage: ChatMessage = {
      id: createId('msg'),
      role: 'user',
      content: text,
      createdAt: now,
    }

    setBusy(true)
    setDraft('')

    updateConversation(conversationId, (c) => {
      let messages = c.messages
      if (options?.regenerateOf) {
        const idx = messages.findIndex((m) => m.id === options.regenerateOf)
        if (idx >= 0) {
          messages = messages.slice(0, idx)
        }
      } else {
        messages = [...messages, userMessage]
      }

      const isFirstUser = !c.messages.some((m) => m.role === 'user')
      return {
        ...c,
        title: isFirstUser && !options?.regenerateOf ? titleFromMessage(text) : c.title,
        updatedAt: now,
        messages,
      }
    })

    try {
      const { response } = await sendChatMessage({
        message: text,
        conversation_id: conversationId,
      })

      const assistantMessage: ChatMessage = {
        id: createId('msg'),
        role: 'assistant',
        content: response.answer,
        createdAt: new Date().toISOString(),
        triageLevel: response.triage_level,
        sources: response.sources,
        disclaimer: response.disclaimer,
        feedback: null,
      }

      updateConversation(conversationId, (c) => ({
        ...c,
        updatedAt: new Date().toISOString(),
        messages: [...c.messages, assistantMessage],
      }))
      setLastFollowUps(response.follow_ups ?? [])
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Yanıt alınamadı. Lütfen tekrar deneyin.'
      const errorMessage: ChatMessage = {
        id: createId('msg'),
        role: 'assistant',
        content: message,
        createdAt: new Date().toISOString(),
        isError: true,
        feedback: null,
      }
      updateConversation(conversationId, (c) => ({
        ...c,
        updatedAt: new Date().toISOString(),
        messages: [...c.messages, errorMessage],
      }))
      setLastFollowUps([])
    } finally {
      setBusy(false)
    }
  }

  const handleFeedback = (messageId: string, value: FeedbackValue) => {
    if (!active) return
    updateConversation(active.id, (c) => ({
      ...c,
      messages: c.messages.map((m) => (m.id === messageId ? { ...m, feedback: value } : m)),
    }))
  }

  const handleRegenerate = () => {
    if (!active || busy) return
    const messages = active.messages
    let lastAssistantIdx = -1
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && !messages[i].isError) {
        lastAssistantIdx = i
        break
      }
    }
    if (lastAssistantIdx < 0) return

    let lastUser: ChatMessage | undefined
    for (let i = lastAssistantIdx - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'user') {
        lastUser = messages[i]
        break
      }
    }
    if (!lastUser) return

    void sendMessage(lastUser.content, { regenerateOf: messages[lastAssistantIdx].id })
  }

  const lastAssistantId = useMemo(() => {
    if (!active) return null
    for (let i = active.messages.length - 1; i >= 0; i -= 1) {
      const m = active.messages[i]
      if (m.role === 'assistant' && !m.isError) return m.id
    }
    return null
  }, [active])

  const isEmpty = !active || active.messages.length === 0
  const apiConfigured = Boolean(import.meta.env.VITE_API_URL)

  return (
    <div className="app-shell">
      <div
        className={`overlay${sidebarOpen ? ' visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
        aria-hidden={!sidebarOpen}
      />

      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={active?.id ?? null}
        searchQuery={searchQuery}
        searchOpen={searchOpen}
        onSearchQueryChange={setSearchQuery}
        onToggleSearch={() => setSearchOpen((v) => !v)}
        onNewChat={handleNewChat}
        onSelect={handleSelect}
        onDelete={handleDelete}
        onRename={handleRename}
        onClearAll={handleClearAll}
        onOpenSettings={() => setSettingsOpen(true)}
        onCloseMobile={() => setSidebarOpen(false)}
      />

      <main className="main">
        <div className="main-topbar">
          <button
            type="button"
            className="menu-btn"
            aria-label="Menüyü aç"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={18} />
          </button>
          <div className="topbar-title">{active?.title ?? 'DiyabetAsistan'}</div>
          <button type="button" className="menu-btn" aria-label="Yeni sohbet" onClick={handleNewChat}>
            <MessageCircle size={18} />
          </button>
        </div>

        <div className="chat-scroll" ref={scrollRef}>
          <DisclaimerBanner />

          {isEmpty ? (
            <div className="empty-state">
              <div className="empty-icon" aria-hidden>
                <MessageCircle size={28} />
              </div>
              <h1>Tip 2 diyabet eğitim asistanı</h1>
              <p>
                Güvenilir eğitim kaynaklarından bilgi alın. Tanı, doz veya acil müdahale için
                kullanılmaz. Aşağıdaki sorulardan biriyle başlayabilirsiniz.
              </p>
              <QuickPrompts
                prompts={QUICK_PROMPTS}
                disabled={busy}
                onSelect={(prompt) => void sendMessage(prompt)}
              />
            </div>
          ) : (
            <>
              {active.messages.map((message) => (
                <ChatMessageView
                  key={message.id}
                  message={message}
                  isLastAssistant={message.id === lastAssistantId}
                  followUps={message.id === lastAssistantId ? lastFollowUps : []}
                  onFeedback={(value) => handleFeedback(message.id, value)}
                  onRegenerate={handleRegenerate}
                  onFollowUp={(text) => void sendMessage(text)}
                  busy={busy}
                />
              ))}
              {busy && <TypingIndicator />}
            </>
          )}
        </div>

        <Composer
          value={draft}
          onChange={setDraft}
          onSubmit={() => void sendMessage(draft)}
          disabled={busy}
        />
      </main>

      {settingsOpen && (
        <div className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title">
          <div className="settings-backdrop" onClick={() => setSettingsOpen(false)} />
          <div className="settings-card">
            <button
              type="button"
              className="settings-close"
              aria-label="Kapat"
              onClick={() => setSettingsOpen(false)}
            >
              <X size={16} />
            </button>
            <h2 id="settings-title">Ayarlar</h2>
            <p>
              Bu arayüz FastAPI <code>/chat</code> sözleşmesine hazırdır. Şu an{' '}
              <strong>{apiConfigured ? 'gerçek API' : 'mock yanıt'}</strong> modunda çalışıyor.
            </p>
            <p>
              Gerçek backend için <code>frontend/.env</code> dosyasına{' '}
              <code>VITE_API_URL=http://localhost:8000</code> ekleyin.
            </p>
            <p>
              Sohbet geçmişi yalnızca tarayıcınızda (<code>localStorage</code>) saklanır; sunucuya
              gönderilmez.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
