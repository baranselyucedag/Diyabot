# -*- coding: utf-8 -*-
"""
batch_label.py — Toplu LLM etiketleme (bulk + verification), checkpoint'li.

Amaç: grey-band sınıflandırıcı için GREEN/YELLOW eğitim verisi üretmek.

Akış:
  1) 3 kaynaktan GREEN adaylarını oku (mucize forum + pool_green + pool_belirsiz)
  2) mucize tarafında bariz tıbbi/acil mesajları kural filtresiyle ele (API tasarrufu)
  3) metin kopyalarını ve --exclude-file ile verilen setlerdeki cümleleri düş
  4) hızlı modelle BATCH halinde etiketle (GREEN / YELLOW / SKIP)
  5) GREEN çıkanları büyük modelle doğrula ("gizli aciliyet var mı?")
  6) final_label üretip CSV'ye yaz

Neden hızlı: 15 mesaj = 1 API çağrısı + 4 worker paralel + bulk'ta hızlı model.
Checkpoint: yarıda kesilirse aynı komutla kaldığı yerden devam eder.

Kullanım (repo kökünden):
    $env:NVIDIA_API_KEY = "nvapi-..."
    python classification_model/src/PreProcessing/batch_label.py --limit 45
    python classification_model/src/PreProcessing/batch_label.py \
        --mucize-n 2000 --green-n 0 --belirsiz-n 2500 \
        --exclude-file classification_model/data/augmented/test_v2.csv
"""

import argparse, json, os, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ─── Yollar ──────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]                       # classification_model/
OUT_DIR_DEFAULT = ROOT / "data" / "labeled" / "batch"
POOLS_DEFAULT = Path(os.environ.get("TEMP", ".")) / "triage_pools"

BASE_URL = "https://integrate.api.nvidia.com/v1"

# Kaynak dosyalar: (dosya adı, metin kolonu, mucize ön-filtresi uygulanacak mı)
SOURCES = {
    "mucize":   {"file": "mucize_genel.csv",      "text_col": "metin", "prefilter": True},
    "green":    {"file": "pool_green_v3.csv",     "text_col": "q",     "prefilter": False},
    "belirsiz": {"file": "pool_belirsiz_v3.csv",  "text_col": "q",     "prefilter": False},
}

MAX_TEXT = 600   # prompt'a giren metin üst sınırı (uzunları kesiyoruz)

# ─── Bulk etiketleme promptu ─────────────────────────────────────────────────
# GREEN tanımı netleştirilen haliyle: kişisel aciliyet/belirti/değer yoksa
# bilgi, ürün, maliyet, motivasyon, yaşam tarzı = GREEN.
BULK_SYSTEM = """Sen Türkçe hasta mesajlarını triage eden bir uzmansın. Tip-2 diyabet
chatbot'unun güvenlik katmanı için eğitim verisi etiketliyorsun.

ETİKETLER:
- GREEN: Genel bilgi sorusu, ürün/fiyat/maliyet, yaşam tarzı, motivasyon, mecaz/deyim,
  günlük sohbet. Kişisel aciliyet, belirti, ölçüm değeri, kötüleşme YOK.
  Örnek GREEN: "Metformin fiyatı nedir?", "HbA1c nedir?", "Motivasyonum çok yüksek",
  "Beyindeki lipom nedir?", "Hangi egzersiz önerilir?", "Pahalıymış bea"
- YELLOW: Kişisel durum + klinik temkin: belirti, ölçüm değeri ("şekerim 320"),
  kötüleşme, hipo/hiper şüphesi, ilaç bırakma/doz değiştirme, sağlık yakınmasıyla
  birlikte "ne yapmalıyım".
  Örnek YELLOW: "Şekerim 320 çıktı ne yapmalıyım?", "Üç gündür çok halsizim",
  "Bebeğim 3 aylık öksürük var kusacak gibi oluyor ne yapmalıyım"
- SKIP: Bu eğitim setine girmemesi gerekenler: doz/tanı talebi, cinsel içerik/şiddet,
  anlamsız veya bozuk metin.

KURALLAR:
- Sadece tıbbi terim geçiyor diye YELLOW yapma; kişisel aciliyet ara.
- Mecaz/deyim ("fiyatı çok yüksek", "üç gündür tatil") → GREEN.
- Kararsızsan GREEN deme; YELLOW de.
- Gerekçeyi (reason) en fazla 5 kelime yaz, yanıtı asla kesme.
- SADECE JSON array döndür, başka hiçbir şey yazma:
  [{"id":1,"label":"GREEN","conf":0.9,"reason":"kısa gerekçe"}]"""

# ─── Verification promptu (sadece GREEN'lere, pahalı model) ──────────────────
VERIFY_SYSTEM = """Aşağıdaki mesajlar GREEN (zararsız) etiketlendi. Sen denetçisin; görevin
yalnızca GERÇEK aciliyeti yakalamak. Bu chatbot Tip-2 diyabet eğitim/bilgilendirme
yapar; genel tıbbi bilgi sorusu, yaşam tarzı, ürün/fiyat, randevu sohbetleri zararsızdır.

RISKY = kişisel kan şekeri/ölçüm bildirimi, hipo/hiper belirtisi, kötüleşme/ani
durum, ilaç/doz değiştirme-bırakma, "ne yapmalıyım" + aktif belirti.

SAFE = genel bilgi sorusu (tıbbi terim içerse bile: "lipom nedir?", "gabapentin
nedir?"), kilo/diyet/egzersiz yaşam tarzı, ürün/fiyat/maliyet, randevu/lojistik,
deneyim paylaşımı, bebek/günlük sohbet.

Önemli: RISKY sadece aciliyet AÇIK olduğunda de; şüphede SAFE bırak.
SADECE JSON array döndür: [{"id":1,"verdict":"SAFE"}]"""

# ─── mucize ön-filtresi (bariz bebek-sağlığı/acil mesajlarını API'ye sokma) ───
PREFILTER_TIBBI = re.compile(
    r"\b(doktor|hastane|ilaç|ilac|tahlil|ateş|ates|kusma|kusuyor|kanama|"
    r"antibiyotik|tansiyon|alerji|muayene|reçete|recete|serum|iğne|igne|"
    r"aşı|asi|öksürük|oksuruk|ishal|kabız|kabiz|pişik|pisik|döküntü)\b", re.I)
PREFILTER_ACIL = re.compile(
    r"\b(acil|ne yapmalıyım|ne yapmaliyim|ne yapsam|korkuyorum|panik|"
    r"dayanamıyorum|dayanamiyorum|bayıldı|bayildi|nefes alamıyor|112|"
    r"hastaneye götürdük|götürdük|goturduk)\b", re.I)


def log(msg: str):
    print(msg, flush=True)


def norm_key(text: str) -> str:
    """Kopya/dışlama eşleşmesi için normalize anahtar."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ─── 1) Veri yükleme ─────────────────────────────────────────────────────────
def load_candidates(args) -> list[dict]:
    """3 kaynağı birleştirir: [{'uid','source','text'}]"""
    items, seen = [], set()

    # Dışlanacak setler (örn. sınıflandırıcının kendi test seti)
    excluded = set()
    for path in (args.exclude_file or []):
        p = Path(path)
        if not p.exists():
            log(f"[UYARI] exclude dosyası yok, atlanıyor: {p}")
            continue
        if p.suffix == ".json":
            data = json.load(open(p, encoding="utf-8"))
            for it in data:
                t = it.get("question") or it.get("sentence") or it.get("text")
                if t:
                    excluded.add(norm_key(t))
        else:
            df = pd.read_csv(p, sep=None, engine="python",
                             encoding="utf-8-sig", dtype=str)
            col = next((c for c in ("sentence", "text", "q", "metin")
                        if c in df.columns), None)
            if col:
                excluded.update(df[col].dropna().map(norm_key))
        log(f"[exclude] {p.name}: toplam {len(excluded)} metin dışlanacak")

    wanted = {"mucize": args.mucize_n, "green": args.green_n,
              "belirsiz": args.belirsiz_n}
    for src, cfg in SOURCES.items():
        df = pd.read_csv(args.pools_dir / cfg["file"], sep=";",
                         encoding="utf-8-sig", dtype=str).fillna("")
        texts = df[cfg["text_col"]].map(lambda t: str(t).strip())
        texts = texts[texts.str.len() >= 15]          # çok kısa = bilgi yok
        if cfg["prefilter"]:
            before = len(texts)
            mask = texts.map(lambda t: bool(PREFILTER_TIBBI.search(t)
                                            or PREFILTER_ACIL.search(t)))
            texts = texts[~mask]
            log(f"[prefilter] {src}: {before} -> {len(texts)} "
                f"({int(mask.sum())} tıbbi/acil elendi)")

        # kopya + dışlama
        uniq = []
        for t in texts:
            k = norm_key(t)
            if k in seen or k in excluded:
                continue
            seen.add(k)
            uniq.append(t)
        # örnekleme (0 = hepsi)
        n_wanted = wanted[src]
        if n_wanted and len(uniq) > n_wanted:
            uniq = random.Random(42).sample(uniq, n_wanted)
        log(f"[kaynak] {src}: {len(uniq)} aday alındı")
        items += [{"uid": f"{src}::{i}", "source": src, "text": t[:MAX_TEXT]}
                  for i, t in enumerate(uniq)]

    if args.limit:
        items = items[:args.limit]
    log(f"[toplam] {len(items)} aday etiketlenecek")
    return items


# ─── 2) API çağrısı (retry + 429 backoff) ────────────────────────────────────
def call_model(client, model, system_prompt, batch, max_retries=6):
    """Bir batch'i modele sorar. Dönüş: {id: dict} veya {} (başarısız)."""
    user = "\n".join(f"{i + 1}) {t}" for i, t in enumerate(batch))
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=6000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = (r.choices[0].message.content or "").strip()
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if not m:
                raise ValueError("JSON array bulunamadı")
            arr = json.loads(m.group(0))
            return {int(o["id"]): o for o in arr
                    if isinstance(o, dict) and "id" in o}
        except Exception as e:
            wait = 20 * attempt if "429" in str(e) else 3 * attempt
            log(f"    [retry {attempt}/{max_retries}] {e} -> {wait}s")
            time.sleep(wait)
    return {}


# ─── 3) Bir aşamayı paralel çalıştır (bulk veya verify) ──────────────────────
def run_stage(client, model, system_prompt, items, key_field, done_ids,
              ckpt_fh, batch_size, workers, stage_name):
    """items: [{'uid','text',...}] -> her item'a key_field ekler."""
    todo = [it for it in items if it["uid"] not in done_ids]
    log(f"[{stage_name}] {len(todo)} yeni mesaj (model: {model})")
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_model, client, model, system_prompt,
                          [it["text"] for it in b]): b for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            results = fut.result()
            for local_id, it in enumerate(batch, start=1):
                obj = results.get(local_id)
                it[key_field] = obj if obj else {"error": "parse/id eksik"}
                ckpt_fh.write(json.dumps(
                    {"uid": it["uid"], "stage": stage_name,
                     key_field: it[key_field]},
                    ensure_ascii=False) + "\n")
            ckpt_fh.flush()
            done_count += len(batch)
            if done_count % 300 < batch_size:
                log(f"    [{stage_name}] {done_count}/{len(todo)}")


def load_checkpoint(path: Path):
    """uid -> {'bulk': {...sonuç...}, 'verify': {...sonuç...}} (düz sonuçlar)."""
    done = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
                stage = o.get("stage")
                if stage in ("bulk", "verify") and isinstance(o.get(stage), dict):
                    done.setdefault(o["uid"], {})[stage] = o[stage]
            except Exception:
                continue
    return done


# ─── 4) Ana akış ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Toplu LLM etiketleme (bulk+verify)")
    ap.add_argument("--mucize-n",   type=int, default=2000,
                    help="forum örneklemi (0=hepsi)")
    ap.add_argument("--green-n",    type=int, default=0,
                    help="pool_green örneklemi (0=hepsi, 3.615)")
    ap.add_argument("--belirsiz-n", type=int, default=2500,
                    help="pool_belirsiz örneklemi")
    ap.add_argument("--limit",      type=int, default=0,
                    help="global üst sınır (duman testi)")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--model-bulk",   default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--model-verify", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--verify-only", action="store_true",
                    help="bulk'u atla; sadece verify çalıştır "
                         "(RISKY/hatalı GREEN'leri yeni prompt ile yeniden denetler)")
    ap.add_argument("--pools-dir",  type=Path, default=POOLS_DEFAULT,
                    help="pool CSV'lerinin dizini")
    ap.add_argument("--exclude-file", nargs="*", default=None,
                    help="Eğitimden düşülecek setler (csv/json, birden fazla olabilir)")
    ap.add_argument("--out",        default=str(OUT_DIR_DEFAULT / "batch_labeled.csv"))
    ap.add_argument("--checkpoint", default=str(OUT_DIR_DEFAULT / "batch_checkpoint.jsonl"))
    args = ap.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit('[HATA] NVIDIA_API_KEY tanımlı değil. '
                 'Önce: $env:NVIDIA_API_KEY="nvapi-..."')
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    items = load_candidates(args)
    ckpt = load_checkpoint(Path(args.checkpoint))

    # Önceki koşudan kalan sonuçları belleğe geri yükle (resume desteği)
    for it in items:
        prev = ckpt.get(it["uid"], {})
        for stage in ("bulk", "verify"):
            if isinstance(prev.get(stage), dict):
                it[stage] = prev[stage]

    bulk_done = {u for u, s in ckpt.items()
                 if isinstance(s.get("bulk"), dict)
                 and s["bulk"].get("label") in ("GREEN", "YELLOW", "SKIP")}
    if args.verify_only:
        # Sadece SAFE kesinleşmişleri atla; RISKY/hatalıları yeniden denetle
        verify_done = {u for u, s in ckpt.items()
                       if isinstance(s.get("verify"), dict)
                       and s["verify"].get("verdict") == "SAFE"}
    else:
        verify_done = {u for u, s in ckpt.items()
                       if isinstance(s.get("verify"), dict) and "verdict" in s["verify"]}
    log(f"[checkpoint] bulk done={len(bulk_done)} verify done={len(verify_done)}")

    ckpt_fh = open(args.checkpoint, "a", encoding="utf-8")

    # Aşama 1: bulk (--verify-only ise atla)
    if not args.verify_only:
        run_stage(client, args.model_bulk, BULK_SYSTEM, items, "bulk",
                  bulk_done, ckpt_fh, args.batch_size, args.workers, "bulk")

    # Aşama 2: verify — sadece bulk'ta GREEN denenler
    greens = [it for it in items
              if str((it.get("bulk") or {}).get("label", "")).upper() == "GREEN"]
    log(f"[verify] {len(greens)} GREEN doğrulanacak")
    run_stage(client, args.model_verify, VERIFY_SYSTEM, greens, "verify",
              verify_done, ckpt_fh, args.batch_size, args.workers, "verify")
    ckpt_fh.close()

    # final_label mantığı
    rows = []
    for it in items:
        b = it.get("bulk") or {}
        label = str(b.get("label", "ERROR")).upper()
        verdict = str((it.get("verify") or {}).get("verdict", "")).upper()
        if label == "GREEN":
            final = {"SAFE": "GREEN", "RISKY": "YELLOW"}.get(verdict, "REVIEW")
        elif label in ("YELLOW", "SKIP"):
            final = label
        else:
            final = "REVIEW"
        rows.append({"uid": it["uid"], "source": it["source"], "text": it["text"],
                     "bulk_label": label, "bulk_conf": b.get("conf", ""),
                     "bulk_reason": b.get("reason", ""), "verify": verdict,
                     "final_label": final})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    log(f"[bitti] {args.out} yazıldı ({len(out)} satır)")
    log("[dağılım]\n" + out["final_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
