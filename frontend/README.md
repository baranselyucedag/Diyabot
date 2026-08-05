# DiyabetAsistan Frontend

Tip 2 diyabet hasta eğitim chatbot’u için React + Vite + TypeScript arayüzü.

Referans tasarımdaki sol sohbet geçmişi, ana konuşma kartı ve alt mesaj kutusu düzeni; Türkçe içerik, kaynak kartları, triage rozeti ve sağlık feragatnamesi ile uyarlanmıştır.

## Kurulum

```bash
cd frontend
npm install
npm run dev
```

Tarayıcıda varsayılan adres: `http://localhost:5173`

## Ortam değişkenleri

`.env.example` dosyasını `.env` olarak kopyalayın:

```bash
cp .env.example .env
```

| Değişken | Açıklama |
|---|---|
| `VITE_API_URL` | FastAPI base URL. Boş bırakılırsa **mock** yanıtlar kullanılır. |

Örnek:

```env
VITE_API_URL=http://localhost:8000
```

## Beklenen API sözleşmesi

`POST {VITE_API_URL}/chat`

İstek:

```json
{
  "message": "Prediyabet nedir?",
  "conversation_id": "opsiyonel-uuid"
}
```

Yanıt:

```json
{
  "answer": "Prediyabet, kan şekeri düzeyinin normalden yüksek ancak diyabet tanısı için yeterince yüksek olmadığı durumdur.",
  "triage_level": "GREEN",
  "sources": [
    {
      "document": "1.hafta-Diyabet ve Prediyabet Hakkında Bilgi",
      "section": "2.1 Prediyabet Tanımı",
      "snippet": "opsiyonel kısa alıntı"
    }
  ],
  "disclaimer": "Bu bilgi genel eğitim amaçlıdır; tanı veya tedavi önerisi değildir.",
  "follow_ups": ["Egzersize nasıl başlamalıyım?"]
}
```

`triage_level` değerleri: `GREEN` | `YELLOW` | `RED` | `REFUSE` | `EMERGENCY`

## Özellikler

- Yeni sohbet, arama, yeniden adlandırma, silme
- `localStorage` ile sohbet geçmişi
- Mock RAG yanıtları (prediyabet, egzersiz, beslenme, takip, ayak bakımı + güvenlik kuralları)
- Faydalı / değil geri bildirimi ve yeniden üret
- Kaynak kartları ve devam soruları
- Responsive mobil menü

## Komutlar

```bash
npm run dev      # geliştirme sunucusu
npm run build    # üretim derlemesi
npm run preview  # derlenmiş önizleme
npm run lint     # oxlint
```

## Not

Bu arayüz eğitim amaçlıdır; tıbbi tavsiye, tanı veya tedavi yerine geçmez.
