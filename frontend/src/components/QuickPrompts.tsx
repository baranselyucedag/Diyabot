import { Sparkles } from 'lucide-react'

interface QuickPromptsProps {
  prompts: readonly string[]
  onSelect: (prompt: string) => void
  disabled?: boolean
}

export function QuickPrompts({ prompts, onSelect, disabled }: QuickPromptsProps) {
  return (
    <div className="quick-prompts" role="list" aria-label="Hızlı sorular">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          type="button"
          className="quick-prompt"
          role="listitem"
          disabled={disabled}
          onClick={() => onSelect(prompt)}
        >
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <Sparkles size={14} aria-hidden />
            {prompt}
          </span>
        </button>
      ))}
    </div>
  )
}
