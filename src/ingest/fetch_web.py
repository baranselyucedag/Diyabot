# -*- coding: utf-8 -*-
"""Web sayfasını çekip chunk_jsonl uyumlu JSONL'ye çevirir.

turkdiab.org gibi hasta eğitimi sayfaları için: <h1> başlığından sonraki
içerik bloklarını toplar, nav/footer gürültüsünü atlar.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# İçerik bittiğinde karşılaşılacak nav/footer işaretleri.
STOP_MARKERS = (
    "English", "Randevu Alın", "İletişim", "Haberler", "Linkler",
    "Bize Ulaşın", "Copyright", "IBAN:", "Çerez", "Bizi Takip Edin",
)


def slugify(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9çğıöşü\-]+", "-", t, flags=re.IGNORECASE)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or "document"


def fetch_html(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=30)
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def extract_blocks(html: str) -> tuple[str | None, list[str]]:
    """<h1> başlığı + sonraki içerik bloklarını (nav/footer hariç) döndürür.

    turkdiab içeriği <p>/<li> etiketinde değil, <h1>'in parent div'inde düz
    metin olarak durur; bu yüzden div metnini satır satır toplarız.
    """
    soup = BeautifulSoup(html, "lxml")
    h1 = soup.find("h1")
    if not h1:
        return None, []
    title = h1.get_text(" ", strip=True)
    container = h1.find_parent("div")

    blocks: list[str] = []
    for ln in container.get_text("\n", strip=True).split("\n"):
        ln = ln.strip()
        if not ln or ln == title:
            continue
        if any(m in ln for m in STOP_MARKERS):
            break
        if len(ln) < 4 or ln in blocks:
            continue
        blocks.append(ln)
    return title, blocks


def build_records(title: str, blocks: list[str], source: str) -> list[dict]:
    doc_id = slugify(title)
    records: list[dict] = []
    for i, b in enumerate(blocks):
        text = b
        # Soru formatındaki kısa satırlar (ör. "Tip 2 Diyabet Nedir?") bu sitede
        # bölüm başlığıdır; chunk_jsonl bunları section olarak işler.
        is_heading = (
            len(b) <= 100
            and b.rstrip().endswith("?")
            and not b.startswith(("SORU:", "S:", "YANIT:", "C:"))
        )
        records.append(
            {
                "document_id": doc_id,
                "source_file": source,
                "paragraph_index": i,
                "text": text,
                "block_type": "heading" if is_heading else "paragraph",
            }
        )
    return records


def save_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    html = fetch_html(args.url)
    title, blocks = extract_blocks(html)
    if not title:
        raise SystemExit("H1 bulunamadı, içerik çekilemedi.")
    records = build_records(title, blocks, args.url)
    output = args.output or Path("data/processed") / f"{slugify(title)}.raw.jsonl"
    save_jsonl(records, output)
    print(f"Baslik: {title}")
    print(f"Blok: {len(blocks)}")
    print(f"Kaydedildi: {output}")


if __name__ == "__main__":
    main()
