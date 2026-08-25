# Diyabot

**Kaynak-temelli, hafızalı ve güvenlik öncelikli Türkçe Tip-2 diyabet hasta eğitim asistanı.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

---

## İçindekiler

1. [Ne İşe Yarıyor](#ne-i̇şe-yarıyor)
2. [Uçtan Uca Akış](#uçtan-uca-akış)
3. [Triage Katmanı](#triage-katmanı)
4. [Retrieval ve RAG](#retrieval-ve-rag)
5. [Hafıza (Memory) Sistemi](#hafıza-memory-sistemi)
6. [Değerlendirme (RAGAS + RAGChecker)](#değerlendirme-ragas--ragchecker)
7. [Veri ve Gold Set](#veri-ve-gold-set)
8. [Proje Yapısı](#proje-yapısı)
9. [Kurulum](#kurulum)
10. [Çalıştırma](#çalıştırma)
11. [API](#api)
12. [Konfigürasyon](#konfigürasyon)
13. [Testler](#testler)
14. [Sınırlamalar](#sınırlamalar)
15. [Lisans](#lisans)

---

## Ne İşe Yarıyor

Diyabot, Tip-2 diyabet hastalarına **Türkçe, kaynağa dayalı eğitim bilgisi** veren bir chatbot. Tasarımın merkezinde tek bir ayrım var:

> Bilgi vermek ile tıbbi karar vermek aynı şey değil.

Bu yüzden sistem hiçbir aşamada tanı koymaz, ilaç dozu önermez, tedavi kararı vermez — bunu prompt seviyesinde bir "nezaket" olarak değil, **kod seviyesinde bir mimari kısıt** olarak uygular: risk taşıyan mesajlar LLM'e hiç gitmez, retrieval'a hiç girmez.

Dört bileşen bir araya geliyor:

- **Triage** — mesaj LLM'e gitmeden önce güvenlik açısından sınıflandırılıyor.
- **RAG** — cevap, serbestçe üretilmiyor; doğrulanmış dokümanlardan getirilen içeriğe dayandırılıyor.
- **Hafıza** — konuşma tek seferlik değil; profil, özet, notlar ve çelişki takibiyle çok turlu bir yapı.
- **Değerlendirme** — model seçimi ve kalite iddiaları varsayımla değil, benchmark ve otomatik RAG değerlendirmesiyle destekleniyor.

---

## Uçtan Uca Akış

```
Kullanıcı mesajı
      │
      ▼
Hafıza yükle (profil, özet, notlar, son turlar)
      │
      ▼
┌─────────────────────────┐
│      TRIAGE             │
│  Hard Veto → Fusion →   │
│  Grey Zone               │
└─────────────┬───────────┘
              │
     ┌────────┴─────────┐
EMERGENCY/REFUSE    GREEN/YELLOW
     │                    │
     ▼                    ▼
Hazır güvenli yanıt   Retrieval (BGE-M3 dense)
(RAG'a hiç gidilmez)       │
                           ▼
                     Reranking (cross-encoder)
                           │
                           ▼
                Top context + hafıza → LLM (Nemotron)
                           │
                           ▼
              Cevap + kaynaklar + triage + disclaimer
                           │
                           ▼
              Hafızaya yaz + arka planda maintenance
```

Bu akışın orkestrasyonu `src/api/pipeline.py` içindeki `run_chat()` fonksiyonunda toplanıyor. Kritik nokta: **EMERGENCY ve REFUSE seviyesindeki mesajlar retrieval'a ve LLM'e hiç girmiyor** — güvenli, hazır bir yanıtla doğrudan sonlanıyor. Hafıza yazımı ve maintenance de kullanıcıya cevap dönen kritik yolun dışında, arka planda (fire-and-forget) çalışıyor; bu sayede memory işlemleri kullanıcı deneyimini yavaşlatmıyor.

---

## Triage Katmanı

`src/api/triage/` altında üç aşamalı, hibrit bir sınıflandırma var:

**1. Hard Veto** — açık acil durum belirtileri, doz değiştirme talepleri, tanı koyma istekleri gibi durumları kural tabanlı ve kesin olarak yakalıyor. Buradaki öncelik "en iyi cevabı bulmak" değil, riskli bir durumda hiç gecikmeden yönlendirme yapmak.

**2. Fusion** (`fusion.py`) — hard veto'nun kesin olarak sonlandırmadığı mesajlarda, kural tabanlı ve skor tabanlı sinyaller birleştiriliyor.

**3. Grey Zone** (`grey_zone.py`, `grey_model.py`) — net GREEN/YELLOW/RED ayrımının belirsiz kaldığı sınır durumlar için ayrı bir sınıflandırıcı katmanı. Bu katman `classification_model/` altında eğitilen bir "grey-band classifier" ile destekleniyor (`model_output_v10/grey_band_classifier.joblib`).

Ortam değişkenleriyle bu katmanın davranışı değiştirilebiliyor (grey model'i devre dışı bırakma, LLM tabanlı grey-zone A/B testi vb.) — detaylar [Konfigürasyon](#konfigürasyon) bölümünde.

Triage seviyeleri: `GREEN`, `YELLOW`, `RED`, `REFUSE`, `EMERGENCY`.

---

## Retrieval ve RAG

`src/retrieval/` üç adımlı bir pipeline:

- **`embed.py`** — sorgu ve dokümanlar `BAAI/bge-m3` ile vektörleştiriliyor, önceden hesaplanmış index `data/index/bge-m3/` altında (`embeddings.npy` + `meta.json`).
- **`retrieve.py`** — dense retrieval ile aday chunk'lar (top-k, varsayılan 10) çekiliyor.
- **`rerank.py`** — cross-encoder tabanlı bir reranker, adayları yeniden sıralayıp LLM'e gidecek nihai bağlamı (varsayılan top-3) belirliyor.

Kaynak dokümanlar `src/ingest/` altındaki script'lerle (PDF/DOCX çıkarma, temizleme, chunk'lama, etiketleme) işlenip `data/processed/` altına chunk JSONL'leri olarak yazılıyor — her chunk `section`/`chapter`/`section_path` gibi hiyerarşi bilgisi taşıyor, bu bilgi kaynak kartlarında ("Bölüm: X > Y") kullanıcıya insan-okunur şekilde gösteriliyor, ham chunk ID'si değil.

LLM'e giden prompt sadece "soru + kaynaklar" değil; profil, özet, son turlar ve bekleyen çelişkiler de aynı prompt'a ekleniyor (`build_user_prompt_with_memory`). Sistem prompt'u, kaynakta olmayan bilgiyi üretmemeyi ve kaynak dokümanlara gömülü olabilecek talimat-benzeri içerikleri kullanıcı komutu gibi kabul etmemeyi zorunlu kılıyor.

---

## Hafıza (Memory) Sistemi

Diyabot'taki hafıza, "son N mesajı tut" mantığından çok daha fazlası — `src/api/memory/` altında ayrı bir alt sistem olarak tasarlandı, konuşma başına ayrı dosyalarla (`data/memory/{conv_id}/`, `data/profiles/{conv_id}.json`).

| Modül | Görevi |
|---|---|
| `models.py` | Profil, not, tur, bekleyen çelişki gibi tüm veri şemaları (Pydantic) |
| `storage.py` | Atomik yazma (`os.replace`), dosya kilidi (`portalocker`), path traversal koruması, `asyncio.Lock` tabanlı conversation-level eşzamanlılık koruması |
| `memory_store.py` | Yüksek seviye okuma/yazma — `create_turn_atomic()`, not eviction (limit dolunca en değerli notları tutma), boş/varsayılan context oluşturma |
| `deterministic.py` | LLM'siz, kural tabanlı kontroller: grounding doğrulaması (bir not gerçekten kaynak mesajda var mı), çakışma kuralları, staleness (eskime) tespiti, Türkçe normalizasyon |
| `maintenance.py` | Tur başına 1-3 LLM çağrısıyla sınırlı bakım döngüsü: profil güncelleme tespiti, yüksek riskli alanlarda ek doğrulama, not çıkarma, özetleme — hepsi tek bir `asyncio.Lock` altında sıralı çalışıyor |
| `expiry.py` | Bekleyen çakışmaların (pending conflict) süre sonunda `expired_rejected` olarak işaretlenip kullanıcıya takip sorusu bırakılması |
| `encryption.py` | Opsiyonel at-rest şifreleme (Fernet), varsayılan kapalı, açıldığında disk üzerindeki tüm hasta verisini şifreliyor |
| `retention_cleanup.py` | Belirli bir süreden eski konuşma verisini (dry-run varsayılan) temizleyen script |
| `metrics.py`, `logger.py` | JSONL tabanlı yapılandırılmış loglama + temel sayaçlar (`/metrics` ile expose edilebilir) |

**Tasarım ilkesi:** LLM'e yalnızca gerçekten "anlama" gerektiren işler için soruluyor (bir mesajın profil güncellemesi içerip içermediği, bir cevabın neyi ima ettiği). "Eşleştirme" nitelikli işler (bir notun kaynak metinde geçip geçmediği, bir ilacın zaten profilde olup olmadığı) LLM olmadan, deterministik kod ile çözülüyor — bu hem maliyeti hem gecikmeyi düşürüyor, hem de bu kontrolleri test edilebilir kılıyor (`tests/test_deterministic.py`).

Eşzamanlılık, sistemin en çok test edilen yanı: aynı konuşmaya iki paralel istek geldiğinde bile `turn_id` çakışması olmaması `tests/test_memory_store.py` ve `tests/test_pipeline.py` içinde ayrıca doğrulanıyor.

---

## Değerlendirme (RAGAS + RAGChecker)

Sistem kalitesi tek bir metrikle değil, **iki bağımsız metodolojiyle** ölçülüyor — `src/eval/` altında:

**RAGAS** (`ragas_predict.py` + `ragas_score.py`) — gold sorular üzerinde uçtan uca cevap üretip (`predictions.jsonl`), ardından ayrı bir judge modeliyle (chat modelinden farklı bir sağlayıcı — self-judge önyargısını önlemek için) skorluyor. Kullanılan başlıca metrikler: `faithfulness`, `context_recall`, `context_precision`, `factual_correctness`, `context_entity_recall`, `noise_sensitivity`. Guardrail (EMERGENCY/REFUSE) satırları, hiç RAG kullanmadıkları için değerlendirmenin dışında tutuluyor.

**RAGChecker** (`ragchecker_run.py`) — cevabı ve referans cevabı atomik iddialara (claim) bölüp her birini context'e karşı doğrulayarak, hatanın **retrieval'dan mı yoksa generation'dan mı** kaynaklandığını ayırt eden daha ince taneli bir teşhis sağlıyor. Retriever metrikleri (`claim_recall`, `context_precision`) top-k context ile, generator metrikleri (`faithfulness`, `hallucination`, `noise_sensitivity`) ise LLM'in gerçekten gördüğü top-3 context ile ayrı ayrı hesaplanıyor. Bu script bilinçli olarak RAGAS'ın kod tabanından tamamen bağımsız — ayrı bir sanal ortamda (`.venv-ragchecker`), farklı bir judge modeliyle (bağımsız triangülasyon için) çalışacak şekilde tasarlandı.

Sonuçlar `eval_results/ragas_<tarih>/` altında `predictions.jsonl`, `scores.jsonl`, `summary.md` ve (RAGChecker koştuysa) `ragchecker_scores.json`/`ragchecker_summary.md` olarak birikiyor.

`src/eval/goldset/` altındaki `build_gold_set.py` ve `build_reference_answers.py`, ham vaka tanımlarından (`data/gold/authoring/cases.jsonl`) test edilebilir gold set'i üretiyor — her satırın `curator_verified`, `coverage_status`, `safety_critical` gibi kalite kontrol alanları var.

---

## Veri ve Gold Set

`data/` altında beş ayrı sorumluluk var, birbirine karıştırılmıyor:

- **`data/raw/`** — orijinal kaynak dokümanlar (DOCX/PDF, tıbbi dergiler, kılavuzlar).
- **`data/processed/`** — chunk'lanmış, etiketlenmiş, retrieval-hazır JSONL'ler.
- **`data/index/bge-m3/`** — önceden hesaplanmış embedding index'i.
- **`data/gold/`** — değerlendirme için gold set (`gold_set.jsonl`), triage test setleri (`triage_test_set.json`, `soft_label_set.json`) ve bunların üretildiği ham vaka tanımları (`authoring/cases.jsonl`).
- **`data/memory/`, `data/profiles/`** — runtime'da oluşan, konuşmaya özel hafıza verisi (bunlar kalıcı proje verisi değil, kullanıcı verisi — production'a taşınırken şifreleme katmanı açılmalı).

---

## Proje Yapısı

```
Diyabot/
├── classification_model/       # Grey-band triage sınıflandırıcısı (eğitim + model artefact)
│   ├── data/
│   ├── model_output_v10/
│   └── src/
│
├── data/
│   ├── raw/                    # Ham kaynak dokümanlar
│   ├── processed/              # Chunk'lanmış, etiketlenmiş veri
│   ├── index/bge-m3/           # Embedding index'i
│   ├── gold/                   # Gold set + authoring
│   ├── memory/                 # Runtime hafıza (konuşma başına)
│   └── profiles/               # Runtime hasta profilleri
│
├── docs/                       # Mimari tasarım dokümanları, karar kayıtları
│
├── eval_results/                # RAGAS / RAGChecker / benchmark çıktıları
│
├── frontend/                   # React + Vite + TypeScript arayüz
│   └── src/
│       ├── components/         # ChatMessage, Composer, Sidebar, SourceCard, ...
│       ├── services/           # chatService.ts (backend ile iletişim)
│       ├── lib/, types/, styles/
│       └── App.tsx
│
├── logs/memory/                 # Yapılandırılmış JSONL loglar (günlük rotasyon)
│
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI giriş noktası
│   │   ├── pipeline.py         # run_chat() — ana orkestrasyon
│   │   ├── llm.py              # LLM çağrı katmanı
│   │   ├── memory/             # Hafıza alt sistemi (yukarıda detaylı)
│   │   └── triage/             # Triage alt sistemi
│   │
│   ├── retrieval/               # embed.py, retrieve.py, rerank.py
│   ├── ingest/                  # Doküman işleme / chunk'lama script'leri
│   │
│   └── eval/
│       ├── ragas_predict.py, ragas_score.py, ragchecker_run.py
│       ├── goldset/             # Gold set üretimi
│       ├── benchmarks/          # Embedding/rerank/sparse benchmarkları
│       ├── checks/              # Triage doğrulama script'leri
│       └── core/                # Ortak eval yardımcıları (dump, metrics, stats)
│
├── tests/                       # pytest test paketi
├── requirements.txt
├── start.bat
└── LICENSE
```

---

## Kurulum

### 1. Depoyu klonla

```bash
git clone https://github.com/baranselyucedag/Diyabot.git
cd Diyabot
```

### 2. Python ortamı

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. RAGChecker için ayrı ortam (opsiyonel)

RAGChecker'ın bağımlılık zinciri (litellm, spacy, refchecker) ana ortamla çakışabildiği için ayrı bir venv'de tutuluyor:

```bash
python -m venv --system-site-packages .venv-ragchecker
.venv-ragchecker\Scripts\python.exe -m pip install ragchecker
.venv-ragchecker\Scripts\python.exe -m spacy download en_core_web_sm
```

### 4. Frontend

```bash
cd frontend
npm install
```

`frontend/.env` içine backend adresi ve LLM API bilgileri eklenmeli:

```env
VITE_API_URL=http://localhost:8000
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b
```

> API anahtarlarını asla Git'e commit etmeyin.

---

## Çalıştırma

**Backend:**

```bash
uvicorn src.api.app:app --reload
```

**Frontend:**

```bash
cd frontend
npm run dev
```

Windows'ta ikisini birlikte başlatmak için depo kökündeki `start.bat` kullanılabilir.

**Değerlendirme:**

```bash
# RAGAS
python -m src.eval.ragas_predict --limit 20
python -m src.eval.ragas_score

# RAGChecker (ayrı venv'den)
.venv-ragchecker\Scripts\python.exe src\eval\ragchecker_run.py --predictions eval_results\ragas_<tarih>\predictions.jsonl
```

---

## API

**`POST /chat`**

```json
{
  "message": "Prediyabet nedir?",
  "conversation_id": "example-conversation-id"
}
```

Örnek yanıt:

```json
{
  "answer": "Prediyabet hakkında kaynaklara dayalı eğitim bilgisi...",
  "triage_level": "GREEN",
  "sources": [
    {
      "document": "Diyabet ve Prediyabet Hakkında Bilgi",
      "section_label": "Bölüm: Diyabet Nedir? > Prediyabet Tanımı",
      "snippet": "..."
    }
  ],
  "disclaimer": "Bu bilgi genel eğitim amaçlıdır; tanı veya tedavi önerisi değildir.",
  "follow_ups": ["Kan şekerimi nasıl takip etmeliyim?", "Egzersize nasıl başlamalıyım?"]
}
```

**`GET /health`** — sistem ve hafıza katmanının sağlık durumu (`memory_ready` alanı disk yazılabilirliğini kontrol ediyor).

**`GET /metrics`** — Prometheus text formatında temel sayaçlar (LLM çağrı sayısı, maintenance task sayısı, staleness olayları).

---

## Konfigürasyon

**Retrieval:** embedding modeli `BAAI/bge-m3`, index `data/index/bge-m3/`, varsayılan aday havuzu 10, LLM'e giden bağlam sayısı 3.

**Triage:**

```env
TRIAGE_GREY_MODEL_PATH=...
TRIAGE_SKIP_GREY_MODEL=1     # grey model'i devre dışı bırak
TRIAGE_USE_LLM_GREY=1        # LLM tabanlı grey-zone A/B testi
TRIAGE_SKIP_LLM=1            # LLM grey-zone'u devre dışı bırak
```

**Memory:** `src/api/memory/config.py` içindeki `MEMORY_CONFIG` — güven eşikleri (`gate_approval_threshold`, `critical_verify_threshold`), staleness/expiry süreleri, not limiti, şifreleme bayrağı ve ortam bazlı override'lar (`APP_ENV=dev|staging|prod`) tek bir yerden yönetiliyor. Güvenlik eşikleri (`pending_conflict_expiry_days`, `critical_verify_threshold` vb.) ortam override'larına asla dahil edilmiyor — sadece operasyonel ayarlar (timeout, retry, log retention) ortama göre değişiyor.

---

## Testler

```bash
python -m pytest tests -q
```

Test paketi özellikle şunları kapsıyor: atomik yazma ve dosya kilidi (`test_storage.py`), deterministik grounding/staleness/conflict kuralları (`test_deterministic.py`), eşzamanlı `create_turn_atomic()` çağrılarında `turn_id` çakışmasının olmadığının doğrulanması (`test_memory_store.py`, `test_pipeline.py`), şifreleme roundtrip'i (`test_encryption.py`), retention temizliği (`test_retention_cleanup.py`) ve LLM istemcisinin retry/fallback davranışı (`test_llm_client.py`).

---

## Sınırlamalar

Diyabot, klinik kullanım için doğrulanmış bir tıbbi cihaz ya da bağımsız klinik karar destek sistemi **değil**. Somut olarak:

- Gold set boyutu (yüzlerce satır) klinik ölçekli bir validasyon için yeterli değil.
- RAGAS/RAGChecker skorları otomatik değerlendirmedir; klinisyen değerlendirmesinin yerini tutmaz. Ayrıca referans-bağımlı metrikler, gold set'teki bazı referans cevapların kısa özet formatında olmasından kaynaklanan bilinen bir sistematik önyargı taşıyabilir.
- Triage modeli bağımsız bir klinik güvenlik sertifikasyonundan geçmedi.
- Cevap kalitesi, RAG kaynaklarının kapsamıyla doğrudan sınırlı — kaynakta olmayan konularda sistem net şekilde "bulamadım" diyor, ama bu kaynak kapsamı genişletilmeden çözülemez.
- Şifreleme, retention cron ve izleme (health/metrics) katmanları varsayılan olarak **kapalı/opsiyonel** — gerçek hasta verisiyle production'a çıkmadan önce bilinçli olarak açılmalı.

Sistem bir **hasta eğitimi ve mühendislik prototipi**dir; tanı, tedavi veya ilaç yönetimi kararı için kullanılmamalıdır.

---

## Lisans

Apache License 2.0 — detaylar için [`LICENSE`](LICENSE) dosyasına bakın.

---

**Diyabot** — güvenlik önce, cevap kaynağa dayalı, hafıza doğrulanmadan güvenilmez.
