# -*- coding: utf-8 -*-
"""
audit_green.py — Domain GREEN'lerin son denetimi.

Neden var: batch_label.py'nin verify aşaması diyabet-merkezli yazıldığı için
green/belirsiz kaynaklı GREEN'lerin içine kişisel semptom ("parmağım uyuşuyor"),
kişisel tahlil/MR/ultrason sonucu ve cinsel/mahrem içerik sızdı. Bunlar grey-band
sınıflandırıcısının eğitiminde zehirli yanlış-GREEN olur.

Ne yapar:
  1) batch_labeled.csv içinden source ∈ {green, belirsiz} ve
     final_label ∈ {GREEN, REVIEW} olan satırları alır (mucize temiz, dokunulmaz)
  2) Ultra modelle dengeli denetim: SAFE / RISKY / SKIP
  3) Sonucu batch_labeled_audited.csv'ye yazar (final_label güncellenir)

Kullanım (repo kökünden):
    $env:NVIDIA_API_KEY = "nvapi-..."
    python classification_model/src/preprocessing/audit_green.py
"""

import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL    = "nvidia/nemotron-3-ultra-550b-a55b"

SCRIPT_PATH = Path(__file__).resolve()
BATCH_DIR   = SCRIPT_PATH.parents[2] / "data" / "labeled" / "batch"
IN_CSV      = BATCH_DIR / "batch_labeled.csv"
OUT_CSV     = BATCH_DIR / "batch_labeled_audited.csv"
CKPT        = BATCH_DIR / "audit_checkpoint.jsonl"

BATCH_SIZE, WORKERS = 15, 4

AUDIT_SYSTEM = """Aşağıdaki mesajlar bir diyabet chatbot'unun eğitim verisi için GREEN
(zararsız) etiketlendi. Sen denetçisin; her mesajı objektif değerlendir.

RISKY = kişisel sağlık yakınması var: belirti (ağrı, uyuşma, yanma, akıntı...),
kişisel ölçüm/tahlil/MR/ultrason sonucu veya değer bildirimi ("akş 108", "4mm
ekojen görünüm"), kötüleşme, "ne yapmalıyım" + aktif yakınma, tedavi kararı sorusu.

SKIP = cinsel/mahrem içerik, doz/tanı talebi, anlamsız veya bozuk metin.

SAFE = genel bilgi sorusu, yaşam tarzı (boy/kilo söylese bile diyet-egzersiz-kilo
sorusu SAFE'dir), ürün/fiyat/maliyet, randevu/lojistik, deneyim paylaşımı, günlük
sohbet, tıbbi terim içeren ama kişisel olmayan bilgi soruları ("lipom nedir?").

Karar kuralı: "kilo vermek istiyorum, 73 kiloyum" = SAFE; "tahlilimde X çıktı,
ağrım var" = RISKY. Ne RISKY'yi abart ne SAFE'i gevşet; objektif ol.
SADECE JSON array döndür: [{"id":1,"verdict":"SAFE"}]"""


def log(msg: str):
    print(msg, flush=True)


def call_model(client, batch, max_retries=6):
    """Bir batch'i denetler. Dönüş: {yerel_id: {'verdict': ...}}"""
    user = "\n".join(f"{i + 1}) {t}" for i, t in enumerate(batch))
    for attempt in range(1, max_retries + 1):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": AUDIT_SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=3000,
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


def main():
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit('[HATA] Önce: $env:NVIDIA_API_KEY="nvapi-..."')
    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    mask = df["source"].isin(["green", "belirsiz"]) & \
           df["final_label"].isin(["GREEN", "REVIEW"])
    targets = df[mask].copy()
    log(f"[hedef] {len(targets)} satır denetlenecek "
        f"(green+belirsiz, GREEN/REVIEW)")

    # checkpoint resume
    done = {}
    if CKPT.exists():
        for line in open(CKPT, encoding="utf-8"):
            try:
                o = json.loads(line)
                if isinstance(o.get("verdict"), str):
                    done[o["uid"]] = o["verdict"]
            except Exception:
                continue
    log(f"[checkpoint] {len(done)} satır zaten denetlenmiş")

    todo = targets[~targets["uid"].isin(done)]
    batches = [todo.iloc[i:i + BATCH_SIZE]
               for i in range(0, len(todo), BATCH_SIZE)]
    log(f"[denetim] {len(todo)} yeni satır, {len(batches)} batch")

    ckpt_fh = open(CKPT, "a", encoding="utf-8")
    processed = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(call_model, client, b["text"].tolist()): b
                for b in batches}
        for fut in as_completed(futs):
            batch = futs[fut]
            results = fut.result()
            for local_id, (_, row) in enumerate(batch.iterrows(), start=1):
                obj = results.get(local_id) or {}
                verdict = str(obj.get("verdict", "")).upper()
                if verdict in ("SAFE", "RISKY", "SKIP"):
                    done[row["uid"]] = verdict
                    ckpt_fh.write(json.dumps(
                        {"uid": row["uid"], "verdict": verdict},
                        ensure_ascii=False) + "\n")
            ckpt_fh.flush()
            processed += len(batch)
            if processed % 300 < BATCH_SIZE:
                log(f"    {processed}/{len(todo)}")
    ckpt_fh.close()

    # Birleştir: verdict -> yeni final_label
    verdict_map = {"SAFE": "GREEN", "RISKY": "YELLOW", "SKIP": "SKIP"}
    df["audit_verdict"] = df["uid"].map(done).fillna("")
    update = df["audit_verdict"].map(verdict_map)
    df.loc[update.notna(), "final_label"] = update[update.notna()]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log(f"[bitti] {OUT_CSV} yazıldı")
    log("[son dağılım]\n" + df["final_label"].value_counts().to_string())
    audited = df[df["audit_verdict"] != ""]
    log("[audit sonucu]\n" + audited["audit_verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
