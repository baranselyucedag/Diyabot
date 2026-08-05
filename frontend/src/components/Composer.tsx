import { useEffect, useRef } from 'react'
import { Brain, SendHorizontal } from 'lucide-react'

interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled?: boolean
  placeholder?: string
}

export function Composer({
  value,
  onChange,
  onSubmit,
  disabled,
  placeholder = 'Aklınızdaki soruyu yazın…',
}: ComposerProps) {
  const ref = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`
  }, [value])

  return (
    <div className="composer-wrap">
      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault()
          if (!disabled && value.trim()) onSubmit()
        }}
      >
        <div className="composer-icon" aria-hidden>
          <Brain size={18} />
        </div>
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          aria-label="Mesajınız"
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (!disabled && value.trim()) onSubmit()
            }
          }}
        />
        <button
          type="submit"
          className="send-btn"
          aria-label="Gönder"
          disabled={disabled || !value.trim()}
        >
          <SendHorizontal size={18} />
        </button>
      </form>
      <p className="composer-hint">
        <span className="kbd">Enter</span> gönder · <span className="kbd">Shift + Enter</span> yeni
        satır · Eğitim amaçlıdır, tıbbi tavsiye değildir
      </p>
    </div>
  )
}
