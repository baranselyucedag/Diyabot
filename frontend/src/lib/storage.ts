import type { Conversation } from '../types/chat'

const STORAGE_KEY = 't2dm-chatbot-conversations-v1'

export function loadConversations(): Conversation[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Conversation[]
    if (!Array.isArray(parsed)) return null
    return parsed
  } catch {
    return null
  }
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
  } catch {
    // Quota / private mode — sessizce yoksay
  }
}

export function createId(prefix = 'id'): string {
  return `${prefix}_${crypto.randomUUID()}`
}
