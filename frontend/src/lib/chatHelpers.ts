import type { ChatMessage, Conversation, TriageLevel } from '../types/chat'
import { DEFAULT_DISCLAIMER } from '../services/chatService'
import { createId } from './storage'

export function triageLabel(level: TriageLevel): string {
  switch (level) {
    case 'GREEN':
      return 'Genel eğitim'
    case 'YELLOW':
      return 'Hekim yönlendirme'
    case 'RED':
      return 'Tedavi değişikliği yok'
    case 'REFUSE':
      return 'Doz / tedavi kapsamı dışı'
    case 'EMERGENCY':
      return 'Acil durum'
  }
}

export function titleFromMessage(text: string): string {
  const cleaned = text.replace(/\s+/g, ' ').trim()
  if (cleaned.length <= 42) return cleaned || 'Yeni sohbet'
  return `${cleaned.slice(0, 42).trim()}…`
}

export function createWelcomeConversation(): Conversation {
  const now = new Date().toISOString()
  const assistant: ChatMessage = {
    id: createId('msg'),
    role: 'assistant',
    content:
      'Merhaba! Ben tip 2 diyabet hasta eğitim asistanıyım. Beslenme, egzersiz, kan şekeri takibi ve öz-yönetim konularında güvenilir eğitim kaynaklarından bilgi paylaşırım.\n\nTanı koyamam, ilaç dozu öneremem ve acil durum yönetemem. Acil belirtilerde lütfen 112’yi arayın.',
    createdAt: now,
    triageLevel: 'GREEN',
    sources: [],
    disclaimer: DEFAULT_DISCLAIMER,
    feedback: null,
  }

  return {
    id: createId('conv'),
    title: 'Hoş geldiniz',
    createdAt: now,
    updatedAt: now,
    messages: [assistant],
  }
}

export function createEmptyConversation(): Conversation {
  const now = new Date().toISOString()
  return {
    id: createId('conv'),
    title: 'Yeni sohbet',
    createdAt: now,
    updatedAt: now,
    messages: [],
  }
}

export function groupConversationsByRecency(conversations: Conversation[]): {
  label: string
  items: Conversation[]
}[] {
  const now = Date.now()
  const day = 24 * 60 * 60 * 1000
  const recent: Conversation[] = []
  const older: Conversation[] = []

  for (const c of conversations) {
    const age = now - new Date(c.updatedAt).getTime()
    if (age <= 7 * day) recent.push(c)
    else older.push(c)
  }

  const groups = []
  if (recent.length) groups.push({ label: 'Son 7 gün', items: recent })
  if (older.length) groups.push({ label: 'Daha eski', items: older })
  return groups
}

export function formatTime(iso: string): string {
  try {
    return new Intl.DateTimeFormat('tr-TR', {
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(iso))
  } catch {
    return ''
  }
}
