import { AlertTriangle } from 'lucide-react'

export function DisclaimerBanner() {
  return (
    <div className="disclaimer-banner" role="note">
      <AlertTriangle size={18} aria-hidden style={{ flexShrink: 0, marginTop: 2 }} />
      <div>
        <strong>Önemli:</strong> Bu bir yapay zekâ eğitim asistanıdır; tıbbi tavsiye, tanı veya tedavi
        yerine geçmez. Acil belirtilerde <strong>112</strong>’yi arayın.
      </div>
    </div>
  )
}
