#!/usr/bin/env python
"""Referans cevap üretici — gold_set.jsonl'deki her küratör onaylı soruya
`expected_answer` (tam, kaynağa dayalı referans cevap) ekler.

Akış:
  1. gold_set.jsonl → `curator_verified=true` satırlarını al.
  2. Her satırın `expected_chunk_ids`'indeki chunk içeriklerini
     `build_content_map(CHUNKS_DIR)` ile yükle.
  3. NVIDIA NIM LLM'ine ver: "Bu kaynaklara dayanarak, soruya tam ve eksiksiz
     bir referans cevap yaz. Kaynakta olmayan hiçbir bilgi ekleme."
  4. `expected_answer` olarak gold_set.jsonl'e yaz (in-place, ayrı dosya YOK).

`expected_answer_summary` (kısa özet) SİLİNMEZ — rapor tablosu/hızlı okuma için
kalır; `expected_answer` onun YANINA yeni bir alan olarak eklenir.

Idempotent: satırda zaten `expected_answer` varsa `--force` olmadan atlanır;
uzun koşu yarıda kesilirse kaldığı yerden devam edilir.

LLM = NVIDIA NIM (OpenAI-uyumlu), `REFERENCE_MODEL` env'i ile seçilir:
  model    : nvidia/nemotron-3-ultra-550b-a55b (varsayılan; free tier çalışır)
  base_url : https://integrate.api.nvidia.com/v1
  api_key  : NVIDIA_API_KEY (chat Nemotron ile aynı key; ayrı key gerekmez)
Not: `minimaxai/minimax-m3` free tier'da rate limit'e takılır (429); istersen
`REFERENCE_MODEL=minimaxai/minimax-m3` ile deneyebilirsin.

Örnek:
  python -m src.eval.goldset.build_reference_answers --limit 3
  python -m src.eval.goldset.build_reference_answers --all
  python -m src.eval.goldset.build_reference_answers --all --write-cases
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.env import load_project_env  # noqa: E402
from src.retrieval.embed import CHUNKS_DIR  # noqa: E402
from src.retrieval.retrieve import build_content_map  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "gold_set.jsonl"
CASES_PATH = ROOT / "data" / "gold" / "authoring" / "cases.jsonl"

MAX_CHUNK_CHARS = 3000  # chunk başına LLM'e gidecek üst karakter sınırı


def load_gold_rows(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    """gold_set.jsonl'i okur; satır sırası korunur (in-place yazım için)."""
    if not path.exists():
        raise FileNotFoundError(f"Gold set yok: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_gold_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Satırları JSONL olarak yazar (UTF-8, satır başına bir kayıt)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_reference_config() -> dict[str, str]:
    """Referans cevap LLM ayarını çözer (NVIDIA NIM, OpenAI-uyumlu).

    Varsayılan Nemotron 550b (free tier'da sorunsuz çalışır). MiniMax M3 free
    tier rate limit'ine takıldığı için `REFERENCE_MODEL` env'i ile model
    değiştirilebilir. Key NVIDIA_API_KEY'den alınır (yeni key gerekmez).
    """
    load_project_env()
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY yok. frontend/.env içine NVIDIA_API_KEY=... yazın "
            "(referans üretimi NVIDIA NIM üzerinden)."
        )
    base_url = (os.getenv("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1").rstrip("/")
    model = os.getenv("REFERENCE_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
    return {"api_key": api_key, "base_url": base_url, "model": model}


def build_reference_prompt(question: str, chunks: list[str]) -> str:
    """Soru + kaynak chunk'larını tek MiniMax user mesajına çevirir.

    Kullanıcı talimatı birebir: "Kaynakta olmayan hiçbir bilgi ekleme."
    """
    blocks: list[str] = []
    for i, text in enumerate(chunks, start=1):
        body = (text or "").strip()[:MAX_CHUNK_CHARS]
        blocks.append(f"[KAYNAK {i}]\n{body}")
    joined = "\n\n".join(blocks) if blocks else "(kaynak yok)"
    return (
        "Görev: Verilen kaynaklara dayanarak, soruya tam ve eksiksiz bir "
        "referans cevap yaz. Kaynakta olmayan hiçbir bilgi ekleme. "
        "Sadece referans cevabı yaz; başlık, numara veya ek açıklama ekleme.\n\n"
        f"SORU:\n{(question or '').strip()}\n\n"
        f"KAYNAKLAR:\n{joined}"
    )


def generate_reference(
    client: Any,
    question: str,
    chunks: list[str],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """NVIDIA NIM LLM'ine referans cevap ürettirir (OpenAI-uyumlu).

    Nemotron'da thinking kapatılır (reasoning çıktısı cevaba karışmasın).
    429/rate/timeout gibi geçici hatalarda backoff ile tekrar dener.
    """
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "user", "content": build_reference_prompt(question, chunks)}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if "nemotron" in model:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                        "force_nonempty_content": True,
                    }
                }
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            if not content or not str(content).strip():
                raise RuntimeError("LLM boş cevap döndü.")
            return str(content).strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc)
            transient = (
                "429" in msg
                or "Too Many Requests" in msg
                or "rate" in msg.lower()
                or "timeout" in msg.lower()
                or "503" in msg
                or "overloaded" in msg.lower()
                or "service unavailable" in msg.lower()
            )
            if transient and attempt < 3:
                time.sleep(6.0 * (attempt + 1))
                continue
            break
    raise last_exc


def write_cases_answers(gold_rows: list[dict[str, Any]]) -> int:
    """gold_set'teki `expected_answer` değerlerini cases.jsonl'e geri yazar.

    cases.jsonl canonical kaynaktır (build_gold_set.py onu gold_set.jsonl'e
    çevirir). `question` (gold) ↔ `q` (cases) birebir eşleşir. Bu adım sayesinde
    referans cevaplar build_gold_set yeniden koşulduğunda kaybolmaz.
    """
    if not CASES_PATH.exists():
        return 0
    cases = load_gold_rows(CASES_PATH)  # aynı JSONL okuyucu (satır bazlı)
    by_q = {row.get("q", "").strip(): row for row in cases}
    updated = 0
    for g in gold_rows:
        ans = g.get("expected_answer")
        if not ans:
            continue
        target = by_q.get((g.get("question") or "").strip())
        if target is not None and target.get("expected_answer") != ans:
            target["expected_answer"] = ans
            updated += 1
    if updated:
        write_gold_rows(CASES_PATH, cases)
    return updated


def select_targets(
    gold: list[dict[str, Any]],
    *,
    limit: int | None,
    force: bool,
) -> list[dict[str, Any]]:
    """Üretilecek (henüz expected_answer'ı olmayan) küratör onaylı satırlar.

    RED_REFUSE/gap satırlarında chunk olmadığından referans üretilmez; doğal
    olarak atlanır (expected_chunk_ids boş).
    """
    targets: list[dict[str, Any]] = []
    for r in gold:
        if not r.get("curator_verified"):
            continue
        if not (r.get("expected_chunk_ids") or []):
            continue
        if r.get("expected_answer") and not force:
            continue
        targets.append(r)
    if limit is not None:
        targets = targets[: max(0, limit)]
    return targets


def run(
    *,
    limit: int | None = None,
    force: bool = False,
    workers: int = 1,
    out_path: Path | None = None,
    write_cases: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> Path:
    """Referans cevapları üretir ve gold_set.jsonl'i günceller; çıktı yolunu döner."""
    cfg = resolve_reference_config()
    gold = load_gold_rows(GOLD_PATH)
    targets = select_targets(gold, limit=limit, force=force)

    if not targets:
        print("Üretilecek satır yok (hepsi dolu ya da limit 0). --force ile yeniden üret.")
        return out_path or GOLD_PATH

    content_map = build_content_map(CHUNKS_DIR)

    from openai import OpenAI

    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
    print("Referans cevap üretici (NVIDIA NIM)")
    print(f"  model    : {cfg['model']}")
    print(f"  base_url : {cfg['base_url']}")
    print(f"  satır    : {len(targets)} (toplam gold {len(gold)})")
    print(f"  workers  : {workers}, max_tokens={max_tokens}, temp={temperature}")

    # chunk yoksa (map'te eksik) üretilemez → hemen boş bırak, hata sayma
    missing_chunks = 0

    def work(row: dict[str, Any]) -> tuple[str, str | None, str | None]:
        qid = str(row.get("id"))
        cids = list(row.get("expected_chunk_ids") or [])
        chunks = [content_map.get(cid, "") for cid in cids]
        chunks = [c for c in chunks if (c or "").strip()]
        if not chunks:
            return qid, None, "chunk yok"
        try:
            ans = generate_reference(
                client, row.get("question") or "", chunks,
                model=cfg["model"], max_tokens=max_tokens, temperature=temperature,
            )
            return qid, ans, None
        except Exception as exc:  # noqa: BLE001
            return qid, None, str(exc)

    results: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, r): r.get("id") for r in targets}
        done = 0
        for fut in as_completed(futures):
            done += 1
            qid, ans, err = fut.result()
            if ans is not None:
                results[qid] = ans
                print(f"  [{done}/{len(targets)}] OK {qid}")
            else:
                errors[qid] = err or "?"
                print(f"  [{done}/{len(targets)}] HATA {qid}: {err}", file=sys.stderr)

    if not results and errors:
        print("\nHiçbir satır üretilemedi — MiniMax yapılandırmasını kontrol edin.", file=sys.stderr)
        return out_path or GOLD_PATH

    # gold satırlarına sonuçları işle (sıra korunur)
    for r in gold:
        qid = str(r.get("id"))
        if qid in results:
            r["expected_answer"] = results[qid]

    dest = out_path or GOLD_PATH
    write_gold_rows(dest, gold)
    print(f"\nYazıldı: {dest} ({len(results)} yeni, {len(errors)} hata)")

    if write_cases:
        n = write_cases_answers(gold)
        print(f"cases.jsonl güncellendi: {n} satır")

    if missing_chunks:
        print(f"Uyarı: {missing_chunks} satırda chunk içeriği bulunamadı (atlandı).")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="gold_set.jsonl'e NVIDIA NIM LLM ile expected_answer (tam referans) ekler."
    )
    parser.add_argument("--limit", type=int, default=None, help="Kaç satır işlensin (test).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tüm eksik satırları işle (limit yok).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Var olan expected_answer'ları da yeniden üret.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Paralel MiniMax M3 çağrı sayısı (free tier 429'a takılır; varsayılan 1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Çıktı dosyası (verilmezse gold_set.jsonl in-place güncellenir).",
    )
    parser.add_argument(
        "--write-cases",
        action="store_true",
        help="Referans cevapları cases.jsonl'e de geri yaz (canonical koruma).",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    run(
        limit=None if args.all else args.limit,
        force=args.force,
        workers=args.workers,
        out_path=args.out,
        write_cases=args.write_cases,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
