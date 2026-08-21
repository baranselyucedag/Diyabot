export type TriageLevel = 'GREEN' | 'YELLOW' | 'RED' | 'REFUSE' | 'EMERGENCY'

export type MessageRole = 'user' | 'assistant' | 'system'

export type FeedbackValue = 'up' | 'down' | null

export interface Source {
  document: string
  section: string
  section_label?: string | null
  snippet?: string
}

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  triageLevel?: TriageLevel
  sources?: Source[]
  disclaimer?: string
  feedback?: FeedbackValue
  isError?: boolean
}

export interface Conversation {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
}

export interface ChatRequest {
  message: string
  conversation_id?: string
}

export interface ChatResponse {
  answer: string
  triage_level: TriageLevel
  sources: Source[]
  disclaimer: string
  follow_ups?: string[]
}

export interface SendMessageResult {
  response: ChatResponse
  usedMock: boolean
}
