import {
  Bot,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage, FeedbackValue } from '../types/chat'
import { formatTime, triageLabel } from '../lib/chatHelpers'
import { SourceCard } from './SourceCard'

interface ChatMessageProps {
  message: ChatMessage
  followUps?: string[]
  isLastAssistant?: boolean
  onFeedback?: (value: FeedbackValue) => void
  onRegenerate?: () => void
  onFollowUp?: (text: string) => void
  busy?: boolean
}

export function ChatMessageView({
  message,
  followUps = [],
  isLastAssistant = false,
  onFeedback,
  onRegenerate,
  onFollowUp,
  busy,
}: ChatMessageProps) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="user-bubble">{message.content}</div>
      </div>
    )
  }

  const sources = message.sources ?? []

  return (
    <div className="message-row assistant">
      <article className="assistant-card" aria-live="polite">
        <div className="assistant-meta">
          <div className="assistant-identity">
            <div className="bot-avatar" aria-hidden>
              <Bot size={16} />
            </div>
            <span>DiyabetAsistan</span>
            <span className="message-time">{formatTime(message.createdAt)}</span>
          </div>
          {message.triageLevel && (
            <span className={`triage-badge ${message.triageLevel}`}>
              {triageLabel(message.triageLevel)}
            </span>
          )}
        </div>

        <div className={`message-body markdown-body${message.isError ? ' error' : ''}`}>
          {message.isError ? (
            message.content
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer">
                    {children}
                  </a>
                ),
                table: ({ children }) => (
                  <div className="table-wrap">
                    <table>{children}</table>
                  </div>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {sources.length > 0 && (
          <div className="sources-block">
            <h3>Kaynaklar</h3>
            <div className="sources-grid">
              {sources.map((source) => (
                <SourceCard key={`${source.document}-${source.section}`} source={source} />
              ))}
            </div>
          </div>
        )}

        {message.disclaimer && (
          <p className="inline-disclaimer">{message.disclaimer}</p>
        )}

        {!message.isError && (
          <div className="message-actions">
            <button
              type="button"
              className={`action-chip${message.feedback === 'up' ? ' active' : ''}`}
              aria-pressed={message.feedback === 'up'}
              onClick={() => onFeedback?.(message.feedback === 'up' ? null : 'up')}
              disabled={busy}
            >
              <ThumbsUp size={14} />
              Faydalı
            </button>
            <button
              type="button"
              className={`action-chip${message.feedback === 'down' ? ' active' : ''}`}
              aria-pressed={message.feedback === 'down'}
              onClick={() => onFeedback?.(message.feedback === 'down' ? null : 'down')}
              disabled={busy}
            >
              <ThumbsDown size={14} />
              Değil
            </button>
            {isLastAssistant && onRegenerate && (
              <button
                type="button"
                className="action-chip regenerate"
                onClick={onRegenerate}
                disabled={busy}
              >
                <RefreshCw size={14} />
                Yeniden üret
              </button>
            )}
          </div>
        )}

        {isLastAssistant && followUps.length > 0 && (
          <div className="followups-block">
            <h3>Devam soruları</h3>
            {followUps.map((q) => (
              <button
                key={q}
                type="button"
                className="followup-chip"
                disabled={busy}
                onClick={() => onFollowUp?.(q)}
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </article>
    </div>
  )
}

export function TypingIndicator() {
  return (
    <div className="message-row assistant" aria-live="polite" aria-busy="true">
      <div className="typing-card">
        <div className="typing-dots" aria-hidden>
          <span />
          <span />
          <span />
        </div>
        Yanıt hazırlanıyor…
      </div>
    </div>
  )
}
