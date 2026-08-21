import type { ChatRequest, ChatResponse, SendMessageResult, TriageLevel } from '../types/chat'

export type ChatHistoryItem = {
  turn_id: string
  role: string
  content: string
  timestamp: string
  triage?: string
}

const DEFAULT_DISCLAIMER =
  'Bu bilgi genel eğitim amaçlıdır; tanı veya tedavi önerisi değildir. Kişisel kararlar için hekiminize danışın.'

const EMERGENCY_KEYWORDS = [
  '112',
  'bayıl',
  'bilinç',
  'titriyorum',
  'konuşamıyorum',
  'göğüs ağrısı',
  'nefes alamıyorum',
  'şekerim 40',
  'şekerim 42',
  'şekerim 45',
]

const DOSE_KEYWORDS = [
  'kaç ünite',
  'doz',
  'insülin artır',
  'ilaç artır',
  'ilaç azalt',
  'metformin doz',
]

const MOCK_KB: Array<{
  triggers: string[]
  answer: string
  triage_level: TriageLevel
  sources: ChatResponse['sources']
  follow_ups: string[]
}> = [
  {
    triggers: ['prediyabet', 'prediyabet nedir'],
    answer:
      'Prediyabet, kan şekeri düzeyinin normalden yüksek ancak diyabet tanısı için yeterince yüksek olmadığı durumdur. Bu dönemde sağlıklı beslenme, düzenli fiziksel aktivite ve kilo yönetimi ile tip 2 diyabet riski azaltılabilir.\n\nÖnemli noktalar:\n• Açlık kan şekeri ve HbA1c değerleri hekim kontrolünde izlenmelidir.\n• Küçük, sürdürülebilir yaşam tarzı değişiklikleri büyük fark yaratır.\n• Bu durum bir tanı veya tedavi planı değildir; takip için hekiminize danışın.',
    triage_level: 'GREEN',
    sources: [
      {
        document: '1.hafta-Diyabet ve Prediyabet Hakkında Bilgi',
        section: '2.1 Prediyabet Tanımı',
        snippet: 'Prediyabet, kan şekeri düzeyinin normalden yüksek ancak diyabet tanısı için yeterince yüksek olmadığı durumdur.',
      },
    ],
    follow_ups: [
      'Tip 2 diyabet nedir?',
      'Kan şekerimi nasıl takip etmeliyim?',
      'Egzersize nasıl başlamalıyım?',
    ],
  },
  {
    triggers: ['egzersiz', 'fiziksel aktivite', 'yürüyüş'],
    answer:
      'Tip 2 diyabet yönetiminde haftada en az 150 dakika orta şiddette fiziksel aktivite önerilir. Yürüyüş, yüzme veya bisiklet gibi aktivitelerle başlamak güvenli bir yaklaşımdır.\n\nBaşlarken dikkat edilecekler:\n• Egzersiz öncesi ve sonrası kan şekeri izlemi önemlidir.\n• Ani, yoğun egzersiz yerine kademeli artış tercih edin.\n• Ayak bakımı ve uygun ayakkabı seçimi komplikasyon riskini azaltır.\n• Göğüs ağrısı, aşırı nefes darlığı veya çok düşük/yüksek şeker belirtilerinde aktiviteyi bırakıp destek alın.',
    triage_level: 'GREEN',
    sources: [
      {
        document: '4.hafta-Fiziksel aktivite ve egzersiz',
        section: 'Güvenli Başlangıç',
        snippet: 'Haftada en az 150 dakika orta şiddette fiziksel aktivite önerilir.',
      },
      {
        document: 'Hemşire Eğitim Rehberi',
        section: 'Yaşam Tarzı ve Egzersiz',
      },
    ],
    follow_ups: [
      'Egzersiz öncesi kan şekeri ne olmalı?',
      'Ayak bakımında nelere dikkat etmeliyim?',
      'Beslenmede nelere dikkat etmeliyim?',
    ],
  },
  {
    triggers: ['beslenme', 'diyet', 'karbonhidrat', 'yemek'],
    answer:
      'Tip 2 diyabette beslenme; porsiyon kontrolü, dengeli öğünler ve düzenli karbonhidrat dağılımı üzerine kuruludur. Tek bir “yasak liste” yerine sürdürülebilir alışkanlıklar hedeflenir.\n\nPratik öneriler:\n• Sebze, tam tahıl ve protein içeren dengeli tabaklar tercih edin.\n• Şekerli içecekleri azaltın.\n• Öğün atlamak yerine düzenli beslenmeyi sürdürün.\n• Kişiye özel diyet planı için diyetisyen/hekim yönlendirmesi alın.',
    triage_level: 'GREEN',
    sources: [
      {
        document: 'Beslenme ve Egzersiz Rehberi',
        section: 'Öğün Planlama',
      },
    ],
    follow_ups: [
      'Prediyabet nedir?',
      'Egzersize nasıl başlamalıyım?',
      'Kan şekerimi nasıl takip etmeliyim?',
    ],
  },
  {
    triggers: ['kan şekeri', 'glukoz', 'ölçüm', 'takip', 'hba1c'],
    answer:
      'Kan şekeri takibi, tip 2 diyabet öz-yönetiminin temel parçalarından biridir. Ölçüm zamanları ve hedefler kişiye göre hekim tarafından belirlenir.\n\nGenel eğitim notları:\n• Açlık, tokluk ve gerektiğinde yatmadan önce ölçüm yapılabilir.\n• Sonuçları tarih/saat ile kaydetmek trendleri görmeyi kolaylaştırır.\n• Çok düşük veya çok yüksek değerlerde acil belirtiler varsa gecikmeden yardım alın.\n• Hedef aralıkları ve cihaz kullanımı için sağlık ekibinize danışın.',
    triage_level: 'GREEN',
    sources: [
      {
        document: '1.hafta-Diyabet ve Prediyabet Hakkında Bilgi',
        section: 'Öz-İzlem',
      },
    ],
    follow_ups: [
      'Hipoglisemi belirtileri nelerdir?',
      'Egzersize nasıl başlamalıyım?',
      'Beslenmede nelere dikkat etmeliyim?',
    ],
  },
  {
    triggers: ['ayak', 'ayak bakımı'],
    answer:
      'Diyabette ayak bakımı, ülser ve enfeksiyon riskini azaltmak için kritiktir.\n\nGünlük öneriler:\n• Ayakları her gün kontrol edin; kızarıklık, yara veya nasır varsa hekime bildirin.\n• Nemli ama ıslak olmayan cilt bakımı uygulayın.\n• Dar ayakkabı ve yalınayak dolaşmaktan kaçının.\n• Tırnak kesimini dikkatli yapın; gerekirse podiatri/hekim desteği alın.',
    triage_level: 'GREEN',
    sources: [
      {
        document: '1.hafta-Diyabet ve Prediyabet Hakkında Bilgi',
        section: 'Ayak Bakımı',
      },
    ],
    follow_ups: [
      'Egzersize nasıl başlamalıyım?',
      'Kan şekerimi nasıl takip etmeliyim?',
    ],
  },
]

function normalize(text: string): string {
  return text
    .toLocaleLowerCase('tr-TR')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
}

function detectTriage(message: string): TriageLevel | null {
  const n = normalize(message)
  if (EMERGENCY_KEYWORDS.some((k) => n.includes(normalize(k)))) {
    return 'EMERGENCY'
  }
  if (DOSE_KEYWORDS.some((k) => n.includes(normalize(k)))) {
    return 'RED'
  }
  if (
    n.includes('260') ||
    n.includes('uc gundur') ||
    n.includes('üç gündür') ||
    n.includes('cok yuksek') ||
    n.includes('çok yüksek')
  ) {
    return 'YELLOW'
  }
  return null
}

function buildMockResponse(message: string): ChatResponse {
  const triage = detectTriage(message)
  if (triage === 'EMERGENCY') {
    return {
      answer:
        'Belirttiğiniz belirtiler acil durum işaretleri olabilir.\n\nLütfen hemen 112 Acil Çağrı Merkezi’ni arayın veya en yakın acil servise başvurun. Bu asistan acil tıbbi müdahale sağlayamaz.',
      triage_level: 'EMERGENCY',
      sources: [],
      disclaimer: DEFAULT_DISCLAIMER,
      follow_ups: [],
    }
  }

  if (triage === 'RED') {
    return {
      answer:
        'İlaç dozu artırma/azaltma veya kişiye özel tedavi değişikliği konusunda öneri veremem.\n\nDoz ve tedavi kararları yalnızca hekiminiz tarafından verilmelidir. Görüşmenizde şu soruları sorabilirsiniz:\n• Mevcut dozum uygun mu?\n• Yan etki veya hipoglisemi riskim nedir?\n• Kontrol randevum ne zaman olmalı?',
      triage_level: 'RED',
      sources: [],
      disclaimer: DEFAULT_DISCLAIMER,
      follow_ups: ['Prediyabet nedir?', 'Kan şekerimi nasıl takip etmeliyim?'],
    }
  }

  if (triage === 'YELLOW') {
    return {
      answer:
        'Anlattığınız durum yakın zamanda hekim değerlendirmesi gerektirebilir.\n\nLütfen takip eden ilk fırsatta sağlık ekibinize başvurun. Şiddetli belirtiler (bilinç bulanıklığı, göğüs ağrısı, nefes darlığı) varsa gecikmeden 112’yi arayın.',
      triage_level: 'YELLOW',
      sources: [],
      disclaimer: DEFAULT_DISCLAIMER,
      follow_ups: ['Hipoglisemi belirtileri nelerdir?', 'Kan şekerimi nasıl takip etmeliyim?'],
    }
  }

  const n = normalize(message)
  const hit = MOCK_KB.find((entry) =>
    entry.triggers.some((t) => n.includes(normalize(t))),
  )

  if (hit) {
    return {
      answer: hit.answer,
      triage_level: hit.triage_level,
      sources: hit.sources,
      disclaimer: DEFAULT_DISCLAIMER,
      follow_ups: hit.follow_ups,
    }
  }

  return {
    answer:
      'Bu konuda doğrulanmış eğitim kaynağımda net bir eşleşme bulamadım. Tip 2 diyabet eğitimi kapsamında şunları sorabilirsiniz: prediyabet, beslenme, egzersiz, kan şekeri takibi veya ayak bakımı.\n\nKişisel tıbbi kararlar için lütfen hekiminize danışın.',
    triage_level: 'GREEN',
    sources: [],
    disclaimer: DEFAULT_DISCLAIMER,
    follow_ups: [
      'Prediyabet nedir?',
      'Egzersize nasıl başlamalıyım?',
      'Beslenmede nelere dikkat etmeliyim?',
    ],
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function callRealApi(request: ChatRequest, baseUrl: string, history: ChatHistoryItem[]): Promise<ChatResponse> {
  const response = await fetch(`${baseUrl.replace(/\/$/, '')}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...request, history }),
  })

  if (!response.ok) {
    throw new Error(`API hatası: ${response.status} ${response.statusText}`)
  }

  const data = (await response.json()) as ChatResponse
  return {
    answer: data.answer,
    triage_level: data.triage_level,
    sources: data.sources ?? [],
    disclaimer: data.disclaimer ?? DEFAULT_DISCLAIMER,
    follow_ups: data.follow_ups ?? [],
  }
}

export async function sendChatMessage(
  request: ChatRequest,
  history: ChatHistoryItem[] = [],
): Promise<SendMessageResult> {
  const apiUrl = import.meta.env.VITE_API_URL as string | undefined

  if (apiUrl && apiUrl.trim().length > 0) {
    const response = await callRealApi(request, apiUrl.trim(), history)
    return { response, usedMock: false }
  }

  await delay(700 + Math.random() * 500)
  return { response: buildMockResponse(request.message), usedMock: true }
}

export const QUICK_PROMPTS = [
  'Prediyabet nedir?',
  'Egzersize nasıl başlamalıyım?',
  'Beslenmede nelere dikkat etmeliyim?',
  'Kan şekerimi nasıl takip etmeliyim?',
  'Ayak bakımında nelere dikkat etmeliyim?',
] as const

export { DEFAULT_DISCLAIMER }
