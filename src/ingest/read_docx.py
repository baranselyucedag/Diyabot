from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from text_clean import clean_source_text


def slugify(text: str) -> str:
    """Dosya adından basit document_id üretir."""
    text = text.lower().strip()
    text = re.sub(r"\.docx$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9çğıöşü\-]+", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "document"


def iter_block_items(parent: DocumentObject):
    """DOCX gövdesindeki paragraf ve tabloları belge sırasıyla üretir."""
    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif child.tag == qn("w:tbl"):
            yield Table(child, parent)


def table_to_text(table: Table) -> str:
    """Tabloyu satır satır, hücreleri | ile birleştirilmiş metne çevirir."""
    rows: list[str] = []

    for row in table.rows:
        cells = [
            clean_source_text(re.sub(r"\s+", " ", (cell.text or "")).strip())
            for cell in row.cells
        ]
        # Birleşik hücre tekrarlarını sadeleştir.
        deduped: list[str] = []
        for cell in cells:
            if not cell:
                continue
            if not deduped or deduped[-1] != cell:
                deduped.append(cell)

        if deduped:
            rows.append(" | ".join(deduped))

    return "\n".join(rows)


def extract_paragraphs(docx_path: Path) -> list[dict]:
    """
    DOCX'ten paragrafları ve tabloları belge sırasıyla çıkarır.

    Her kayıt: document_id, source_file, paragraph_index, text, block_type
    """
    if not docx_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {docx_path}")
    if docx_path.suffix.lower() != ".docx":
        raise ValueError(f"DOCX bekleniyor, gelen: {docx_path.suffix}")

    document = Document(docx_path)
    document_id = slugify(docx_path.name)
    records: list[dict] = []
    index = 0

    for block in iter_block_items(document):
        if isinstance(block, Paragraph):
            text = clean_source_text((block.text or "").strip())
            block_type = "paragraph"
        else:
            text = clean_source_text(table_to_text(block))
            block_type = "table"

        if not text:
            continue

        records.append(
            {
                "document_id": document_id,
                "source_file": docx_path.name,
                "paragraph_index": index,
                "text": text,
                "block_type": block_type,
            }
        )
        index += 1

    return records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Kayıtları JSONL olarak kaydeder (satır başına bir JSON)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_first_n(records: list[dict], n: int = 5) -> None:
    print(f"Toplam kayıt: {len(records)}")
    tables = sum(1 for record in records if record.get("block_type") == "table")
    print(f"Tablo bloğu: {tables}")
    print(f"İlk {min(n, len(records))} kayıt:\n")
    for i, record in enumerate(records[:n], start=1):
        preview = record["text"][:160].replace("\n", " / ")
        print(
            f"{i}. [index={record['paragraph_index']}] "
            f"({record.get('block_type', 'paragraph')}) {preview}"
        )
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DOCX dosyasından paragraf ve tablo metinlerini JSONL'ye çevirir."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="DOCX dosyasının yolu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı JSONL yolu. Verilmezse data/processed/<slug>.jsonl kullanılır.",
    )
    args = parser.parse_args()

    docx_path: Path = args.input
    records = extract_paragraphs(docx_path)
    output_path = args.output or Path("data/processed") / f"{slugify(docx_path.name)}.jsonl"
    save_jsonl(records, output_path)
    print_first_n(records, n=5)
    print(f"\nKaydedildi: {output_path}")


if __name__ == "__main__":
    main()
