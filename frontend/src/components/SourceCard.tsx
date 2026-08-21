import { BookOpen } from 'lucide-react'
import type { Source } from '../types/chat'

interface SourceCardProps {
  source: Source
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <article className="source-card">
      <div className="source-icon" aria-hidden>
        <BookOpen size={16} />
      </div>
      <div className="source-body">
        <div className="source-doc">{source.document}</div>
        {source.section_label ? (
          <div className="source-section">Bölüm: {source.section_label}</div>
        ) : null}
        {source.snippet && <div className="source-snippet">{source.snippet}</div>}
      </div>
    </article>
  )
}
