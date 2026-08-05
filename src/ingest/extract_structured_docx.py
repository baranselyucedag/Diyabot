from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from read_docx import slugify, table_to_text
from text_clean import clean_source_text

# Düz "Normal" metinde numaralı alt başlık: "4.1. | GLİSEMİK HEDEFLER",
# "6.1.1. | ...", "12.1.4. | TEDAVİ" gibi. Bazı kılavuzlar başlıklara
# Word Heading stili vermez; bunları da alt başlık olarak yakalarız.
NUMBERED_SUBHEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s*\|\s*\S")

# "A. RİSKLİ BİREYLERİN BELİRLENMESİ" gibi harf-maddeli büyük başlıklar.
LETTER_SUBHEADING = re.compile(r"^[A-ZÇĞİÖŞÜ]\.\s+[A-ZÇĞİÖŞÜ]")


def style_name(paragraph: Paragraph) -> str:
    """Paragrafın stil adını güvenli biçimde döndürür."""
    if paragraph.style and paragraph.style.name:
        return paragraph.style.name
    return ""


def heading_level(paragraph: Paragraph) -> int | None:
    """
    Paragrafın başlık seviyesini döndürür (1/2/3) veya None.

    Word Heading stili yoksa numaralı/harf desenli düz metni de değerlendirir.
    """
    style = style_name(paragraph)
    text = clean_source_text((paragraph.text or "").strip())

    if not text:
        return None

    if style == "Heading 1":
        return 1
    if style in {"Heading 2", "Heading 4"}:
        # H4: bazı kılavuzlarda (TEMD teknoloji) Kaynaklar / ara başlık.
        return 2
    if style in {"Heading 3", "Heading 5", "Heading 6"}:
        return 3

    # Stilsiz (Normal) ama numaralı/harf desenli kısa başlık satırları.
    if len(text) <= 90 and (
        NUMBERED_SUBHEADING.match(text) or LETTER_SUBHEADING.match(text)
    ):
        # Nokta sayısına göre kaba seviye: "4.1" -> 2, "6.1.1" -> 3
        dot_count = len(re.findall(r"\d+", text.split("|", 1)[0]))
        return 3 if dot_count >= 3 else 2

    return None


def extract_structured(docx_path: Path) -> list[dict]:
    """
    DOCX'i belge sırasıyla gezer; her paragraf/tabloya bölüm bağlamı ekler.

    Her kayıt: document_id, source_file, paragraph_index, text, block_type,
    style, heading_level, chapter, subsection.
    """
    if not docx_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {docx_path}")
    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"DOCX bekleniyor, gelen: {docx_path.suffix}")

    document = Document(docx_path)
    document_id = slugify(docx_path.name)
    records: list[dict] = []
    index = 0

    chapter = ""
    subsection = ""

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = clean_source_text((paragraph.text or "").strip())

            if not text:
                continue

            level = heading_level(paragraph)

            if level == 1:
                chapter = text
                subsection = ""
            elif level == 2:
                subsection = text
            elif level == 3:
                subsection = text

            records.append(
                {
                    "document_id": document_id,
                    "source_file": docx_path.name,
                    "paragraph_index": index,
                    "text": text,
                    "block_type": "paragraph",
                    "style": style_name(paragraph),
                    "heading_level": level,
                    "chapter": chapter,
                    "subsection": subsection,
                }
            )
            index += 1

        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            text = clean_source_text(table_to_text(table))

            if not text:
                continue

            records.append(
                {
                    "document_id": document_id,
                    "source_file": docx_path.name,
                    "paragraph_index": index,
                    "text": text,
                    "block_type": "table",
                    "style": "",
                    "heading_level": None,
                    "chapter": chapter,
                    "subsection": subsection,
                }
            )
            index += 1

    return records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Kayıtları JSONL olarak kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: list[dict]) -> None:
    """Bölüm bağlamıyla kısa özet yazdırır."""
    from collections import Counter

    tables = sum(1 for record in records if record["block_type"] == "table")
    chapters = Counter(record["chapter"] for record in records if record["chapter"])

    print(f"Toplam kayıt : {len(records)}")
    print(f"Tablo bloğu  : {tables}")
    print(f"Bölüm sayısı : {len(chapters)}")
    for chapter, count in list(chapters.items())[:30]:
        print(f"  [{count:4}] {chapter[:70]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOCX'i başlık/bölüm bağlamı koruyarak yapısal JSONL'ye çevirir."
    )
    parser.add_argument("--input", required=True, type=Path, help="DOCX yolu.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı JSONL. Verilmezse data/processed/<slug>.raw.jsonl.",
    )
    args = parser.parse_args()

    docx_path: Path = args.input
    records = extract_structured(docx_path)
    output_path = args.output or Path("data/processed") / f"{slugify(docx_path.name)}.raw.jsonl"
    save_jsonl(records, output_path)
    print_summary(records)
    print(f"\nKaydedildi: {output_path}")


if __name__ == "__main__":
    main()
