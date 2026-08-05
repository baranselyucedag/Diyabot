import {
  FileText,
  MessageCircle,
  Pencil,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from 'lucide-react'
import type { Conversation } from '../types/chat'
import { groupConversationsByRecency } from '../lib/chatHelpers'

interface SidebarProps {
  open: boolean
  conversations: Conversation[]
  activeId: string | null
  searchQuery: string
  searchOpen: boolean
  onSearchQueryChange: (value: string) => void
  onToggleSearch: () => void
  onNewChat: () => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  onRename: (id: string) => void
  onClearAll: () => void
  onOpenSettings: () => void
  onCloseMobile: () => void
}

export function Sidebar({
  open,
  conversations,
  activeId,
  searchQuery,
  searchOpen,
  onSearchQueryChange,
  onToggleSearch,
  onNewChat,
  onSelect,
  onDelete,
  onRename,
  onClearAll,
  onOpenSettings,
  onCloseMobile,
}: SidebarProps) {
  const filtered = conversations.filter((c) =>
    c.title.toLocaleLowerCase('tr-TR').includes(searchQuery.toLocaleLowerCase('tr-TR')),
  )
  const groups = groupConversationsByRecency(filtered)

  return (
    <aside className={`sidebar${open ? ' open' : ''}`} aria-label="Sohbet geçmişi">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-hidden>
          <MessageCircle size={18} strokeWidth={2.4} />
        </div>
        <div className="brand-text">
          <span className="brand-title">DiyabetAsistan</span>
          <span className="brand-subtitle">Tip 2 eğitim chatbot’u</span>
        </div>
        <button
          type="button"
          className="menu-btn"
          aria-label="Menüyü kapat"
          onClick={onCloseMobile}
          id="sidebar-close-mobile"
        >
          <X size={18} />
        </button>
      </div>

      <div className="sidebar-actions">
        <button type="button" className="btn-new-chat" onClick={onNewChat}>
          <Plus size={18} strokeWidth={2.5} />
          Yeni sohbet
        </button>
        <button
          type="button"
          className={`btn-icon${searchOpen ? ' active' : ''}`}
          aria-label="Sohbet ara"
          aria-pressed={searchOpen}
          onClick={onToggleSearch}
        >
          <Search size={18} />
        </button>
      </div>

      {searchOpen && (
        <div className="search-box">
          <Search size={16} color="#8a97a8" aria-hidden />
          <input
            type="search"
            placeholder="Sohbetlerde ara…"
            value={searchQuery}
            onChange={(e) => onSearchQueryChange(e.target.value)}
            aria-label="Sohbet ara"
            autoFocus
          />
        </div>
      )}

      <div className="conversation-panel">
        <div className="conversation-header">
          <h2>Sohbetleriniz</h2>
          {conversations.length > 0 && (
            <button type="button" className="link-btn" onClick={onClearAll}>
              Tümünü sil
            </button>
          )}
        </div>

        <div className="conversation-list" role="list">
          {groups.length === 0 && (
            <p
              style={{
                padding: '0.75rem',
                color: 'var(--color-text-muted)',
                fontSize: '0.85rem',
              }}
            >
              Henüz sohbet yok. Yeni bir sohbet başlatın.
            </p>
          )}

          {groups.map((group) => (
            <div key={group.label}>
              <div className="group-label">{group.label}</div>
              {group.items.map((conv) => {
                const active = conv.id === activeId
                return (
                  <div
                    key={conv.id}
                    role="listitem"
                    className={`conversation-item${active ? ' active' : ''}`}
                    onClick={() => onSelect(conv.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        onSelect(conv.id)
                      }
                    }}
                    tabIndex={0}
                    aria-current={active ? 'page' : undefined}
                  >
                    <FileText size={16} className="item-icon" aria-hidden />
                    <span className="item-title">{conv.title}</span>
                    <div
                      className="item-actions"
                      onClick={(e) => e.stopPropagation()}
                      onKeyDown={(e) => e.stopPropagation()}
                    >
                      <button
                        type="button"
                        className="item-action"
                        aria-label="Yeniden adlandır"
                        onClick={() => onRename(conv.id)}
                      >
                        <Pencil size={14} />
                      </button>
                      <button
                        type="button"
                        className="item-action danger"
                        aria-label="Sohbeti sil"
                        onClick={() => onDelete(conv.id)}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                    {active && <span className="active-dot" aria-hidden />}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      <div className="sidebar-footer">
        <button type="button" className="settings-btn" onClick={onOpenSettings}>
          <Settings size={18} />
          Ayarlar
        </button>
        <div className="user-card">
          <div className="avatar" aria-hidden>
            HY
          </div>
          <div className="user-meta">
            <div className="user-name">Hasta Kullanıcı</div>
            <div className="user-role">Eğitim modu</div>
          </div>
        </div>
      </div>
    </aside>
  )
}
