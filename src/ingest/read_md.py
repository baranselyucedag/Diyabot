from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from text_clean import clean_source_text


SECTION_HEADING = re.compile(r"^(\d+)\.\s+(.+)$")
FAQ_INLINE = re.compile(
    r"^\*{0,2}S:\*{0,2}\s*(.+?)\s*\*{0,2}C:\*{0,2}\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
HORIZONTAL_RULE = re.compile(r"^-{3,}$")
MARKDOWN_BOLD = re.compile(r"\*\*(.+?)\*\*")
MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MISSING_MARKER = "BU KONU MEVCUT KAYNAKLARDA"


def slugify(text: str) -> str:
    """Dosya adından basit document_id üretir."""
    text = text.lower().strip()
    text = re.sub(r"\.(md|markdown)$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9çğıöşü\-]+", "-", text, flags=re.IGNORECASE)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "document"


def strip_markdown(text: str) -> str:
    """Hasta SSS metnindeki Markdown işaretlerini sadeleştirir."""
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = MARKDOWN_BOLD.sub(r"\1", text)
    text = text.replace("**", "")
    text = re.sub(r"[ \t]+", " ", text)
    return clean_source_text(text.strip())


def looks_like_section_heading(line: str) -> bool:
    """Numaralı konu başlığını ayırt eder (liste maddesi değil)."""
    match = SECTION_HEADING.match(line)
    if not match:
        return False

    title = match.group(2).strip()
    # SSS, uzun paragraf veya adım listesi başlık değildir.
    if "S:" in title or "C:" in title or len(title) > 90:
        return False
    # "1. Ellerinizi yıkayın." gibi yönerge maddelerini ele.
    if title.endswith("."):
        return False
    # Çok kısa / anlamsız maddeler konu başlığı değildir.
    # "9. Aşılar" gibi tek kelimelik gerçek bölüm başlıklarına izin ver.
    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+", title)
    if not words:
        return False
    if len(words) == 1 and len(title) < 4:
        return False

    return True


def chapter_heading(line: str) -> str:
    """chunk_jsonl'in chapter olarak tanıdığı BÖLÜM N: ... formatına çevirir."""
    match = SECTION_HEADING.match(line)
    assert match is not None
    number, title = match.group(1), strip_markdown(match.group(2))
    return f"Bölüm {number}: {title}"


def parse_faq_line(line: str) -> tuple[str, str] | None:
    """**S:** ... **C:** ... satırını (soru, cevap) olarak ayırır."""
    cleaned = line.strip()
    # Bold işaretlerini geçici olarak kaldırıp yakala.
    probe = cleaned.replace("**", "")
    match = FAQ_INLINE.match(probe)
    if not match:
        return None

    question = strip_markdown(match.group(1))
    answer = strip_markdown(match.group(2))
    return question, answer


def rebuild_target_table(lines: list[str], start: int) -> tuple[str | None, int]:
    """
    Bölüm 1'deki bozuk tabloyu tek tablo bloğuna çevirir.

    Dönüş: (table_text, next_index). Tablo değilse (None, start).
    """
    if start >= len(lines) or strip_markdown(lines[start]) != "Durum":
        return None, start

    expected = [
        "Durum",
        "Açlık / Öğün Öncesi",
        "Tokluk (2. Saat)",
        "HbA1c Hedefi",
        "Genel Yetişkin",
        "80 - 130 mg/dL",
        "< 160 mg/dL",
        "≤ %7",
        "Sağlıklı Yaşlı",
        "80 - 130 mg/dL",
        "80 - 180 mg/dL",
        "< %7 - 7.5",
        "Kırılgan Yaşlı",
        "100 - 180 mg/dL",
        "110 - 200 mg/dL",
        "< %8 - 8.5",
        "Gebe (İnsülin kullanmayan)",
        "< 95 mg/dL",
        "< 120 mg/dL",
        "< %6 - 6.5",
    ]

    collected: list[str] = []
    index = start
    for expected_value in expected:
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            return None, start
        value = strip_markdown(lines[index])
        if value != expected_value:
            return None, start
        collected.append(value)
        index += 1

    while index < len(lines) and not lines[index].strip():
        index += 1

    rows = [
        " | ".join(collected[0:4]),
        " | ".join(collected[4:8]),
        " | ".join(collected[8:12]),
        " | ".join(collected[12:16]),
        " | ".join(collected[16:20]),
    ]
    return "\n".join(rows), index


def extract_records(md_path: Path) -> list[dict]:
    """Markdown SSS dosyasını chunk_jsonl uyumlu kayıtlara çevirir."""
    if not md_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {md_path}")

    raw = md_path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in raw.splitlines()]

    document_id = slugify(md_path.name)
    source_file = md_path.name
    records: list[dict] = []
    index = 0
    # None = cevap beklenmiyor; list = çok satırlı C: toplanıyor.
    pending_answer_parts: list[str] | None = None

    def flush_pending_answer() -> None:
        nonlocal pending_answer_parts, index
        if pending_answer_parts is None:
            return

        answer = strip_markdown("\n".join(pending_answer_parts))
        pending_answer_parts = None

        if not answer or MISSING_MARKER in answer.upper():
            # Eksik/boş cevap → eşlik eden S: kaydını da çıkar.
            if records and str(records[-1]["text"]).startswith("S:"):
                records.pop()
                index = records[-1]["paragraph_index"] + 1 if records else 0
            return

        records.append(
            {
                "document_id": document_id,
                "source_file": source_file,
                "paragraph_index": index,
                "text": f"C: {answer}",
                "block_type": "paragraph",
            }
        )
        index += 1

    def append_record(text: str, block_type: str = "paragraph") -> None:
        nonlocal index
        text = strip_markdown(text) if block_type != "table" else text
        if not text:
            return
        records.append(
            {
                "document_id": document_id,
                "source_file": source_file,
                "paragraph_index": index,
                "text": text,
                "block_type": block_type,
            }
        )
        index += 1

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or HORIZONTAL_RULE.match(line):
            i += 1
            continue

        table_text, next_i = rebuild_target_table(lines, i)
        if table_text is not None:
            flush_pending_answer()
            append_record(table_text, block_type="table")
            i = next_i
            continue

        if looks_like_section_heading(line):
            flush_pending_answer()
            append_record(chapter_heading(line))
            i += 1
            continue

        faq = parse_faq_line(line)
        if faq is not None:
            flush_pending_answer()
            question, answer = faq

            if MISSING_MARKER in answer.upper():
                i += 1
                continue

            append_record(f"S: {question}")
            if answer:
                append_record(f"C: {answer}")
            else:
                pending_answer_parts = []
            i += 1
            continue

        if pending_answer_parts is not None:
            pending_answer_parts.append(line)
            i += 1
            continue

        append_record(line)
        i += 1

    flush_pending_answer()
    return records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Kayıtları JSONL olarak kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_preview(records: list[dict], n: int = 8) -> None:
    print(f"Toplam kayıt: {len(records)}")
    tables = sum(1 for record in records if record.get("block_type") == "table")
    faqs = sum(1 for record in records if str(record.get("text", "")).startswith("S:"))
    print(f"Tablo bloğu: {tables}")
    print(f"SSS soru (S:): {faqs}")
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
        description="Markdown SSS dosyasını chunk_jsonl uyumlu JSONL'ye çevirir."
    )
    parser.add_argument("--input", required=True, type=Path, help="Markdown dosya yolu.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı JSONL yolu. Verilmezse data/processed/<slug>.jsonl kullanılır.",
    )
    args = parser.parse_args()

    md_path: Path = args.input
    records = extract_records(md_path)
    output_path = args.output or Path("data/processed") / f"{slugify(md_path.name)}.jsonl"
    save_jsonl(records, output_path)
    print_preview(records)
    print(f"\nKaydedildi: {output_path}")


if __name__ == "__main__":
    main()
