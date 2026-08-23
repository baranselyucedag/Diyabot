# Değerlendirme (Eval) Katmanı — 4 Bağımsız Faz
### RAGAS · RAGChecker · ARES · MIRAGE — Araştırmaya Dayalı, Öğretici + Üretime Hazır Plan (v2)

**Kapsam:** Bu dört bölüm birbirinden bağımsızdır — istediğin sırayla, istediğin kadarını uygulayabilirsin. Her bölüm önce **ne olduğunu** anlatıyor, sonra **gerçek paket/API** bilgisiyle faz faz pseudo-kod veriyor, sonunda **üretime hazır sürüm** için ne eklemen gerektiğini söylüyor.

**Ortam notu:** Projen Python 3.12.9 kullanıyor (pytest çıktısından teyitli). RAGChecker şu an **Python ≤3.12** gerektiriyor (3.13'te derleme hatası veriyor, GitHub issue'da doğrulanmış) — senin ortamın buna zaten uygun, ekstra bir şey yapmana gerek yok.

**v2'de ne değişti (kısa özet):**
- RAGAS pseudo-kodu **güncel, sınıf-tabanlı API'ye** çevrildi (eski fonksiyon-import tarzı RAGAS 0.1/0.2 dönemine aitti).
- `answer_relevancy` artık "ekle" tavsiyesi değil — senin bilinçli "dışarıda bıraktım" kararına saygılı, **opsiyonel** bir not olarak sunuluyor.
- RAGChecker bölümüne üç kritik düzeltme: **guardrail filtresi**, **referans alanının kısa özet olması uyarısı**, **maliyet kontrolü**.
- ARES bölümüne üç kritik düzeltme: **judge bağımsızlığı kuralı**, **Türkçe in-domain few-shot zorunluluğu**, **PPI'nın asıl değerinin güven aralığı olduğu** motivasyonu.
- Üç framework'ün çıktısını `gold_id` ile birleştiren **tek ana rapor** adımı eklendi.

---

## BÖLÜM 1 — RAGAS (mevcut kodun düzeltmesi + üretim sağlamlaştırma)

### Ne yapıyor, kısaca

Cevabı **atomik iddialara** böler, her iddianın verilen context'ten çıkarılıp çıkarılamayacağını bir LLM judge'a sorar. Referans (gold) cevaba ihtiyaç duymadan çalışabilir (`faithfulness`), ya da gold cevabınla karşılaştırarak da çalışabilir (`context_recall`, `factual_correctness`).

### Faz 1.1 — Mevcut bug'ları düzelt (bunu zaten konuşmuştuk, öncelik bu)

```
FAZ 1.1 — ragas_predict.py + ragas_score.py düzeltme
  a) predict_one() ölü kodunu sil; retrieve_step() + generate_step() olarak ikiye böl,
     hem tekil hem paralel akış AYNI fonksiyonları çağırsın.
  b) resolve_judge_config()'e base_url ekle (OPENAI_BASE_URL env, default
     "https://api.openai.com/v1" — proje genelinde NVIDIA_BASE_URL ile aynı desen).
  c) run_score() içindeki çift resolve_judge_config() çağrısını teke indir.
```

### Faz 1.2 — Gerçek RAGAS API (kurulum + minimal akış)

```bash
pip install ragas datasets
```

```python
# PSEUDO-KOD — güncel ragas API'sine göre (sınıf-tabanlı, 0.2+ dönemi)
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,                     # context'e sadakat (halüsinasyon yok mu)
    LLMContextPrecisionWithReference, # üstteki chunk'lar gerçekten alakalı mı
    LLMContextRecall,                 # gerekli TÜM bilgi retrieve edilmiş mi (gold'a göre)
    FactualCorrectness,               # cevap, gold cevaba ne kadar yakın
)

# 1. predictions.jsonl -> HuggingFace Dataset formatına çevir
rows = load_predictions("predictions.jsonl")
dataset = Dataset.from_dict({
    "question":          [r["question"] for r in rows],
    "answer":            [r["answer"] for r in rows],
    "contexts":          [[c["content"] for c in r["retrieved_contexts"]] for r in rows],
    "ground_truth":      [r["reference"] for r in rows],
})

# 2. Değerlendir — metrikler ARTIK SINIF: örnekleyerek (instance olarak) veriyorsun
result = evaluate(
    dataset,
    metrics=[Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall(), FactualCorrectness()],
    llm=judge_llm,   # senin resolve_judge_config()'ten kurduğun LLM wrapper
)

result.to_pandas().to_csv("ragas_scores.csv")
```

**İki dürüst not (v2 düzeltmeleri):**

1. **Eski plan burada hatalıydı.** `from ragas.metrics import faithfulness` gibi fonksiyon-import tarzı RAGAS'ın eski (0.1/0.2 başı) dönemine ait; internetteki 2026 tarihli "guide" yazıları bile hâlâ o eski tarzı gösterebiliyor. Güncel kullanım yukarıdaki gibi **sınıf-tabanlı**: metrikleri `Faithfulness()` olarak örneklendirip veriyorsun. Yine de kural geçerli: kurduğun sürümün kendi dokümanını aç, imzaları oradan teyit et.
2. **Kolon isimleri sürüme göre değişebiliyor.** Bazı sürümler `question/answer/contexts/ground_truth`, daha yenileri `user_input/response/retrieved_contexts/reference` bekliyor. Kod yazmadan önce `pip show ragas` ile sürümü öğren, o sürümün örnek veri şemasına bak — tahminle kolon adı yazma.

**`answer_relevancy` hakkında dürüst not:** Bu metrik, judge LLM'in yanında bir de **embedding modeli** istiyor. Sen bu projede embedding-bağımlı metrikleri **bilinçli olarak dışarıda bıraktın** (ikinci bir API bağımlılığı eklememek için) — bu geçerli bir karar ve plan buna saygı duyuyor; metrik bu planın varsayılan listesinde **yok**. İleride eklemek istersen en temiz yol: retrieval'da zaten kullandığın yerel **bge-m3** modelini RAGAS'ın custom embedding arayüzüne sarmak — maliyet sıfır, yeni API bağımlılığı yok:

```python
# PSEUDO-KOD — sadece ileride answer_relevancy istersen
from ragas.embeddings import BaseRagasEmbeddings

class LocalBGEEmbeddings(BaseRagasEmbeddings):
    # içeride senin mevcut bge-m3 modelini çağırır; API'ya gitmez
    ...
```

### Faz 1.3 — Üretime hazır sağlamlaştırma

- **CI-tarzı eşik kontrolü:** Her metriğin bir "taban" değeri olsun (`faithfulness >= 0.85` gibi — halüsinasyon en kritik hata türü olduğu için en sıkı eşik ona), script bu eşiğin altına düşünce **non-zero exit code** versin — ileride otomatik bir kalite kapısı olarak kullanılabilir.
- **50-200 soru** aralığı, RAGAS için pratikte "yeterli çeşitlilik" kabul edilen bir aralık — senin 178 satırlık gold set'in bu aralığa zaten uygun, büyütmene gerek yok.
- Sonuçları hem `scores.jsonl` (satır bazlı, debug için) hem `summary.md` (ortalama + dağılım, rapor için) olarak yaz — zaten senin mevcut tasarımında bu ayrım var, koru.

---

## BÖLÜM 2 — RAGChecker (yeni entegrasyon)

### Ne yapıyor, RAGAS'tan farkı

RAGAS'a benzer şekilde iddia-seviyesinde çalışıyor, ama **iki yönlü** parçalıyor: hem "cevaptaki iddialar context'te destekleniyor mu" (precision) hem "gold cevaptaki iddialar cevapta var mı" (recall) — bu ayrım, hatanın **retrieval'dan mı generation'dan mı** geldiğini net ayırt etmeni sağlıyor. Kendi yayınladıkları meta-değerlendirmede (insan tercihiyle korelasyon) RAGAS ve ARES'i geçtiğini iddia ediyorlar — kendi makaleleri olduğu için temkinli oku, ama metodolojik olarak gerçekten daha ince taneli.

### Faz 2.1 — Kurulum

```bash
pip install ragchecker
python -m spacy download en_core_web_sm
```

⚠️ **Türkçe uyarı (iki katmanlı):**
1. **spaCy katmanı:** `en_core_web_sm` İngilizce bir model — RAGChecker bunu cümle bölme gibi yardımcı işler için kullanıyor. Türkçe için `tr_core_news_sm` mevcut; kullanıp kullanamayacağını RAGChecker dokümanından doğrula.
2. **Asıl kritik katman — claim extraction:** RAGChecker'ın iddia çıkarma promptları kütüphanenin içinde gömülü ve **İngilizce** yazılmış. Judge LLM Türkçe metni anlayabilir ama prompt diliyle içerik dili farklı olunca çıkarım kalitesi düşebilir (claim'ler birleşir/atlanır). Tam çalıştırmadan önce **5-10 Türkçe örnekte** çıkan claim'leri elle kontrol et — claim'ler anlamlı ve eksiksizse devam et, değilse bunu raporda sınırlama olarak not et. Bu, "önce oku sonra yaz" kuralımızın tam uygulanacağı yer.

### Faz 2.2 — Girdi formatı ve çalıştırma

```python
# PSEUDO-KOD — gerçek RAGChecker API'sine göre
from ragchecker import RAGResults, RAGChecker

# 1. predictions.jsonl -> RAGChecker'ın beklediği JSON şemasına çevir
#    ÖNCE İKİ FİLTRE (aşağıdaki Faz 2.3 notlarına bak):
#    a) guardrail satırlarını (EMERGENCY/REFUSE, retrieved_contexts == []) ÇIKAR
#    b) referansı olmayan satırları ÇIKAR
rows = filter_for_ragchecker(load_predictions("predictions.jsonl"))

checking_input = {
    "results": [
        {
            "query_id": row["gold_id"],
            "query": row["question"],
            "gt_answer": row["reference"],          # ⚠️ aşağıdaki uyarıyı oku!
            "response": row["answer"],
            "retrieved_context": [
                {"doc_id": c["chunk_id"], "text": c["content"]}
                for c in row["retrieved_contexts"]
            ],
        }
        for row in rows
    ]
}
write_json(checking_input, "ragchecker_input.json")

# 2. Değerlendir — model adı litellm formatında: "openai/<model>"
evaluator = RAGChecker(
    extractor_name="openai/gpt-4o-mini",   # iddia çıkarma için LLM
    checker_name="openai/gpt-4o-mini",     # entailment kontrolü için LLM
    batch_size_extractor=8,
    batch_size_checker=8,
)
evaluator.evaluate(rag_results, metrics=SECTIGIN_METRIKLER)  # all_metrics şart değil, aşağıya bak
print(rag_results)
```

⚠️ **`gt_answer` alan uyarısı (doğrulanmış gerçek sorun):** Senin `gold_set.jsonl`'indeki referans alanının adı `expected_answer_summary` ve içeriği **tam bir cevap değil, ~15 kelimelik bir özet** (örnek: *"Kronik ama %5-10 kilo kaybı + yaşam tarzı ile remisyon olabilir; karar hekimin."*). RAGChecker gibi claim-parçalama ile çalışan bir framework'te bu sistematik yanlılık yaratır:
- Özet kısa → az claim çıkar → claim recall **yapay olarak yüksek** görünür.
- Senin chatbot'unun doğru cevabındaki claim'ler özette geçmiyor → "desteklenmiyor" sayılır → precision/hallucination tarafı **yapay olarak düşük** çıkar.

**İki seçeneğin var:** (a) gold set'teki her satıra tam bir referans cevap alanı ekle (en temiz çözüm ama el emeği ister), ya da (b) özeti olduğu gibi kullan ve RAGChecker sonuçlarını raporda bu yanlılığı açıkça yazarak sun. Hangisini seçersen seç, kararı **kod yazmadan önce** ver.

### Faz 2.3 — Üretime hazır sağlamlaştırma

- **Guardrail filtresi (RAGAS'taki mantığın aynısı):** EMERGENCY/REFUSE gibi hazır cevap satırlarında `retrieved_contexts` boş — bunları RAGChecker'a geçme, yoksa context-metrikleri dejenere olur. RAGAS tarafında kullandığın `split_rag_vs_guardrail` mantığını buraya da taşı. Kaç satır girdi, kaç satır neden atlandı — açıkça logla.
- **Maliyet kontrolü (atlanmaması gereken nokta):** RAGChecker soru başına claim çıkarma + her claim × her chunk kontrolü yaptığı için **30-150 judge çağrısı** tüketir — üç framework içinde açık ara en pahalısı. 178 sorunun tamamında çalıştırma; gold set'inde zaten var olan `safety_critical` flag'ini kullan: **safety-critical satırlar + rastgele %20'lik bir alt küme** yeterli. `all_metrics` yerine ihtiyacın olan metrikleri seç (hallucination, noise sensitivity, faithfulness, claim recall gibi).
- **Judge faturası:** RAGChecker litellm altyapısı kullanıyor — model adını `"openai/<model>"` formatında veriyorsun. İstersen OpenAI yerine elindeki NVIDIA endpoint'inden (OpenAI-uyumlu, `integrate.api.nvidia.com`) koşturmayı dene; base_url ayarını litellm'in nasıl aldığını önce küçük bir testle doğrula — ikinci bir faturalama hesabı açmaktan kurtarır.
- **Raporlama:** `retriever metrikleri` ile `generator metrikleri` ayrı raporlanıyor — bunları RAGAS'ın context_precision/faithfulness ayrımıyla yan yana bir tabloda göster, bu tam senin istediğin "üçlü metodoloji karşılaştırması" hikayesine hizmet eder.

---

## BÖLÜM 3 — ARES (iki yol: hızlı UES/IDP + tam PPI)

### Ne yapıyor, ve senin bugünkü zaman kısıtın için kritik bir bulgu

ARES'in tek çalışma modu, makaledeki ağır yöntem (judge fine-tuning + kalibrasyon) değil. Kütüphanenin (`ares-ai`) kendisi **iki ayrı mod** sunuyor:

1. **UES/IDP** (Unsupervised Evaluation / In-Domain Prompting) — **fine-tuning YOK**. Bir LLM'i doğrudan few-shot prompt ile "hakem" olarak kullanıyor. Kavramsal olarak RAGAS/RAGChecker'a çok yakın (LLM-judge), sadece ARES'in kendi skorlama formatını kullanıyor. **Bugün yapılabilir.**
2. **PPI (tam ARES)** — makaledeki asıl yöntem: küçük bir judge modeli fine-tune + sentetik veri + insan etiketli kalibrasyon seti. GPU + zaman gerektiriyor (donanım tarafını sen ayarlıyorsun, plan buna karışmıyor).

**Önerim:** Bugün **UES/IDP** ile başla (hızlı, çalışır, ARES'in resmi bir modu — "ARES kullanmadım" değil, "ARES'in hafif modunu kullandım" diyebilirsin, bu dürüst ve savunulabilir). PPI'yı, zamanın kalırsa ya da tezin ileri bir revizyonu için **Faz 3.3**'e bıraktım.

### Faz 3.1 — Kurulum

```bash
pip install ares-ai
```

### Faz 3.2 — UES/IDP (bugün yapılabilir, fine-tuning yok)

```python
# PSEUDO-KOD — gerçek ares-ai API'sine göre
from ares import ARES

ues_idp_config = {
    "in_domain_prompts_dataset": "diabet_few_shot_prompt_for_judge_scoring.tsv",
    "unlabeled_evaluation_set": "diabet_unlabeled_output.tsv",   # senin predictions.jsonl -> TSV
    "model_choice": "meta/llama-3.1-70b-instruct",   # ⚠️ aşağıdaki judge kuralını oku!
    "vllm": True,
    "host_url": "https://integrate.api.nvidia.com/v1",  # NVIDIA endpoint'in OpenAI-uyumlu
}
# NOT: host_url'e NVIDIA endpoint'ini verince API anahtarının nasıl geçtiğini
# (env var mı, header mı) küçük bir testle doğrula — dokümanına bak.

ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()
print(results)
# -> {'Context Relevance Scores': [...], 'Answer Faithfulness Scores': [...], 'Answer Relevance Scores': [...]}
```

**İki kritik kural (v2'de eklendi):**

1. **Judge bağımsızlığı kuralı:** RAGAS tarafında judge'ın GPT-4o-mini (OpenAI). ARES UES judge'ını **da** aynı model yaparsan, iki framework aynı "gözle" bakmış olur — bu, "üçlü metodoloji karşılaştırması"nın temelini çürütür (aynı model ailesi iki kere aynı şeyi ölçer, bağımsız triangülasyon olmaz). Bu yüzden yukarıda bilerek **farklı bir model** yazdım: NVIDIA endpoint'indeki Llama-3.1-70B. Elindeki `NVIDIA_API_KEY` zaten var, ekstra fatura yok. Dağılım: **RAGAS = GPT judge, ARES = Llama judge, RAGChecker = ikisinden biri** — gerçek üçlü karşılaştırma ancak böyle olur.
2. **"In-domain" adının içini doldur:** IDP'nin IDP'si = *in-domain prompting*, yani few-shot örneklerinin **senin alanından** olması. `diabet_few_shot_prompt_for_judge_scoring.tsv` dosyasını **Türkçe, diyabet/tıbbi eğitim alanından 3-5 örnekle** kendin yazmalısın. Resmi örnek dosyadaki İngilizce NQ (Natural Questions) örneklerini olduğu gibi bırakırsan "in-domain" boş bir etiket kalır — İngilizce generic judge'dan farksız çalışır.

**Şema notu (önemli, kod yazmadan önce doğrulanmalı):** ARES'in beklediği TSV kolonları resmi örnek dosyalarda (`nq_few_shot_prompt_for_judge_scoring.tsv`, `nq_unlabeled_output.tsv`) tanımlı. Kodlayan yapay zeka bu dosyaları **indirip açmalı**, kolon isimlerini (`Query`, `Document`, `Answer` gibi olması muhtemel ama teyit edilmeli) görüp, senin `predictions.jsonl`'ini **o şemaya uyacak şekilde** dönüştürmeli — tahmin ederek TSV yazmasın.

### Faz 3.3 — PPI (tam ARES, üretim/tez-derinliği için, zaman varsa)

**Neden uğraşmaya değer — asıl motivasyon bu:** PPI'nın çıktısı nokta tahmin değil, **güven aralığıdır**. "Faithfulness = 0.87" yerine **"Faithfulness = 0.87 ± 0.03"** yazabilmek demek. Bu cümle tezde/jüride ciddi akademik ağırlık taşır — judge'ın ne kadar güvenilir olduğunu istatistiksel olarak kanıtlamış olursun. Üç framework içinde bunu verebilen tek yöntem PPI.

```python
# PSEUDO-KOD — sadece zamanın varsa
ppi_config = {
    "evaluation_datasets": ["diabet_unlabeled_output.tsv"],
    "few_shot_examples_filepath": "diabet_few_shot_prompt_for_judge_scoring.tsv",
    "checkpoints": ["<fine-tune_ettigin_judge_modelinin_yolu>"],  # ⚠️ "llm_judge" değil!
    "rag_type": "question_answering",
    "labels": ["Context_Relevance_Label", "Answer_Faithfulness_Label", "Answer_Relevance_Label"],
    "gold_label_path": "diabet_labeled_output.tsv",   # kalibrasyon seti — aşağıdaki kuralı oku
}
ares = ARES(ppi=ppi_config)
results = ares.evaluate_RAG()
print(results)  # istatistiksel güven aralıklı, kalibre edilmiş skorlar
```

**Config uyarısı:** PPI config anahtarları UES config'inden farklı — gerçek pakette fine-tune'lu judge `checkpoints` anahtarıyla veriliyor (önceki taslaktaki `llm_judge` yazımı doğru değildi). Faz 3.3'e geçince resmi PPI örneğini açıp anahtarları oradan teyit et.

**Kalibrasyon seti kuralı (v2'de düzeltildi):**
- Kalibrasyon etiketlerin, değerlendirdiğin veriyle **aynı dağılımdan** ama **ayrık** bir alt kümeden gelmeli.
- Aynı 178 satırın tamamını hem eval'de hem kalibrasyonda kullanma — döngüsellik olur, güven aralığı sahte bir özgüvenle daralır.
- En doğrusu: 178'i katmanlı olarak ikiye böl (ör. **130 eval + 48 kalibrasyon** — `safety_critical` ve kategori dağılımı iki tarafta da korunarak), ya da kalibrasyon için aynı üslupta/dağılımda birkaç düzine **yeni soru** üret.

**Etiketleme akışı:**
1. Kalibrasyon alt kümeni belirle (yukarıdaki kurala göre).
2. Bu satırları güçlü bir modele (GPT-4 sınıfı) `Context_Relevance_Label`, `Answer_Faithfulness_Label`, `Answer_Relevance_Label` için **taslak** etiket ürettir.
3. **Sen** bu taslakları hızlıca gözden geçirip düzelt (ARES'in kendi makalesinde GPT-4 etiketi ile insan etiketini karşılaştırdıkları bölüm — Table 4 — tam bu yöntemi meşrulaştırıyor, referans göster).
4. `gold_label_path`'e bu düzeltilmiş etiketleri yaz.

---

## BÖLÜM 4 — MIRAGE (dış geçerlilik / bağlamsallaştırma bölümü)

### Neden ve nasıl konumlandırılmalı (tekrar hatırlatma)

MIRAGE, İngilizce çoktan seçmeli tıp sınavı sorularından oluşuyor (MedQA-US, MMLU-Med, MedMCQA, PubMedQA*, BioASQ-Y/N — 7.663 soru toplam), kendi korpusu (MedCorp: PubMed + StatPearls + ders kitapları + Wikipedia) senin Türkçe diyabet dokümanlarınla **ilgisiz**. Bunu ana kalite metriğin gibi sunma — raporunda **ayrı bir "dış geçerlilik" bölümü** olarak kullan.

### Faz 4.1 — Küçük alt küme + published baseline karşılaştırması

```bash
# GitHub deposundan hazır veriyi çek
git clone https://github.com/Teddy-XiongGZ/MIRAGE
# benchmark.json içinde 5 görev, her biri için sorular + çoktan seçmeli şıklar var
```

```python
# PSEUDO-KOD
import json, random

benchmark = json.load(open("MIRAGE/benchmark.json"))
# 150-300 soruluk stratifiye bir örneklem al (5 görevden orantılı)
sample = stratified_sample(benchmark, n=200, per_task_proportional=True)

results = []
for item in sample:
    question = item["question"]
    options = item["options"]  # {"A": "...", "B": "...", ...}
    gold_answer = item["answer"]

    # NOT: MIRAGE "question-only retrieval" bekliyor — senin retriever'ını
    # senin index'ine karşı DEĞİL, MedCorp'a (ya da hazır retrieved_snippets_10k.zip'e) karşı çalıştırman gerekir
    prompt = format_mcq_prompt(question, options)
    model_answer = call_your_llm(prompt)  # sadece LLM, senin RAG pipeline'ın değil (ya da MedCorp-retrieval ile)

    results.append({"correct": model_answer == gold_answer, "task": item["task"]})

accuracy_by_task = aggregate_accuracy(results)
```

```python
# Published baseline tablosuna yerleştirme (makale Table 5-7'den)
baseline_table = {
    "GPT-3.5 (CoT)":        {"MMLU-Med": 72.9, "MedQA-US": 65.0, ...},
    "GPT-4 (CoT)":          {"MMLU-Med": 89.4, "MedQA-US": 83.9, ...},
    "Senin modelin (CoT)":  accuracy_by_task,   # kendi ölçtüğün
}
# Tabloyu yan yana raporda sun
```

### Faz 4.2 — Üretime hazır / rapor-hazır not

- Tam retrieval (MedCorp üzerinde) kurmak zaman alır — hazır `retrieved_snippets_10k.zip`'i kullanmak (MIRAGE deposunda mevcut) zamandan büyük tasarruf sağlar, kendi retrieval'ını çalıştırmana gerek kalmaz.
- Raporda mutlaka şu cümleyi ekle: *"MIRAGE değerlendirmesi, sistemin altındaki dil modelinin genel tıbbi muhakeme kapasitesini, kendi Türkçe korpusumdan bağımsız olarak ölçmek amacıyla yapılmıştır; asıl sistem kalitesi değerlendirmesi Bölüm X'teki (gold_set) sonuçlara dayanmaktadır."* — bu cümle, jüri/danışman karşısında metodolojik karışıklığı önler.

---

## Genel Uygulama Sırası (bugün için önerilen)

```
1. RAGAS bug düzeltmesi (Faz 1.1)                    — ~30 dk, zorunlu, zaten konuşulmuştu
2. ARES UES/IDP (Faz 3.2)                             — ~1-2 saat, fine-tuning yok, hızlı
   (unutma: judge = NVIDIA Llama-3.1-70B, few-shot TSV = Türkçe/tıbbi)
3. RAGChecker entegrasyonu (Faz 2.1-2.2)              — ~1-2 saat, ama ÖNCE:
   guardrail filtresi + gt_answer kararı + 5-10 Türkçe claim doğrulaması
4. MIRAGE alt küme + baseline (Faz 4.1)               — ~1-2 saat, hazır snippet'lerle hızlanır
5. (Zaman kalırsa) ARES PPI (Faz 3.3)                 — saatler sürer, etiketleme + GPU gerekir
6. TÜM sonuçları gold_id ile tek tabloda birleştir    — aşağıya bak
```

**Son adım — tek ana rapor:** Üç framework ayrı ayrı dosya yazacak; bunları raporda yan yana koyabilmek için hepsini **`gold_id` üzerinden birleştir**: her framework'ün satır-bazlı çıktısını okuyup tek bir `scores_master.jsonl` (ya da CSV) üret — kolonlar: `gold_id`, `ragas_*`, `ragchecker_*`, `ares_*`. "Üçlü metodoloji karşılaştırması" tablon bu dosyadan tek komutla çıkar. Judge maliyet dengesi için hatırlatma: RAGAS tüm sette (ucuz), ARES UES tüm sette (ucuz), RAGChecker sadece safety-critical + %20 alt kümede (pahalı).

**Kritik uyarı — her bölüm için geçerli:** Hiçbir pseudo-kodu olduğu gibi kopyala-yapıştır etme. Her kütüphanenin (RAGAS, RAGChecker, ares-ai) **gerçek, güncel API'si** burada yazdığımdan küçük farklar taşıyabilir (kütüphane sürümleri hızlı değişiyor — bu planın v1'inde RAGAS için tam bunu yaşadık). Kodlayan yapay zekaya her bölüme başlamadan önce, o kütüphanenin **kendi resmi örnek dosyalarını/dokümantasyonunu** açıp gerçek şemayı doğrulamasını, sonra kod yazmasını söyle — tıpkı memory sisteminde yaptığımız "ADIM 0: önce oku" disiplini gibi.
