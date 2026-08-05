from __future__ import annotations

"""
Beslenme ve Egzersiz (Metabolizma Derneği) kitabı için hasta-eğitimi filtresi.

Whitelist:
  4  — Makrobesinler
  6  — Beslenme programları (Akdeniz, DASH, düşük KH, aralıklı oruç)
  9  — Güvenli gıda / etiket (hasta etiket okuma)
  12 — Sağlıklı bireylerde egzersiz
  13 — Diyabet ve egzersiz (doz tabloları hariç)

Dışarıda: klinik değerlendirme, mikrobesin dozları (IU), genetik,
sporcu ürünleri, alerji, kaynakça, insülin bolus ayarı.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CHAPTER_WHITELIST = {"4", "6", "9", "12", "13"}
CHAPTER_NUMBER = re.compile(r"Bölüm\s+(\d+)\b", re.IGNORECASE)

DROP_SUBSECTION_RE = re.compile(
    r"kaynaklar|genetik|gdo|mikrobiyota|alerji|sporcu|"
    r"insülin\s+pl[aâ]n|bolus|doz\s+ayar",
    re.IGNORECASE,
)

DROP_CONTENT_PATTERNS = [
    # Doz / klinik tedavi
    r"\bünite\b",
    r"\biu\b",
    r"\bdozaj",
    r"\btitrasyon",
    r"\bbolus\b",
    r"insülin\s+doz",
    r"dozundan\s+%\d+",
    r"%\d+\s*[-–]\s*%?\d*\s*oranında\s+azalt",
    r"\d+\s?mg\s?/\s?(gün|kg)",
    r"\d+\s?mg(?!\s?/\s?d[lL])",
    r"yükleme\s+dozu|idame\s+dozu",
    # Ağır araştırma / klinik jargon
    r"randomize",
    r"meta-?analiz",
    r"\bet al",
    r"plasebo",
    r"mortalite|insidans|prevalans",
    r"malabsorbsiyon|steroid\s+tedav",
    r"serum\s+25",
    # Kapsam dışı konular (bölüm 9 içinde bile)
    r"biyogüvenlik|cartegena|good\s+manufacturing",
    r"dezenfeksiyon|gıda\s+işletme",
]
DROP_CONTENT_RE = re.compile("|".join(DROP_CONTENT_PATTERNS), re.IGNORECASE)


def tr_lower(text: str) -> str:
    return (
        text.replace("İ", "i")
        .replace("I", "ı")
        .replace("Ş", "ş")
        .replace("Ğ", "ğ")
        .replace("Ü", "ü")
        .replace("Ö", "ö")
        .replace("Ç", "ç")
        .lower()
    )


def chapter_number(chapter_title: str) -> str | None:
    match = CHAPTER_NUMBER.search(chapter_title or "")
    return match.group(1) if match else None


def is_references_marker(text: str) -> bool:
    return tr_lower(text or "").strip() == "kaynaklar"


def is_inline_reference(text: str) -> bool:
    return bool(re.match(r"\s*kaynak(lar)?\s*:", tr_lower(text or "")))


def evaluate(record: dict[str, Any]) -> tuple[bool, str]:
    number = chapter_number(record.get("chapter", ""))
    if number not in CHAPTER_WHITELIST:
        return False, f"chapter_out_of_scope({number})"

    subsection = record.get("subsection", "") or ""
    if DROP_SUBSECTION_RE.search(tr_lower(subsection)):
        return False, "drop_subsection"

    if is_inline_reference(record["text"]):
        return False, "references_inline"

    if record.get("heading_level") == 1:
        return True, "ok"

    if record.get("heading_level") in (2, 3):
        lowered = tr_lower(record["text"])
        if DROP_SUBSECTION_RE.search(lowered) or DROP_CONTENT_RE.search(lowered):
            return False, "drop_heading"
        return True, "ok"

    text = record.get("text", "")
    if DROP_CONTENT_RE.search(tr_lower(text)):
        return False, "drop_content"

    # Sayfa numarası / çok kısa gürültü
    if len(text.strip()) < 40 and record.get("block_type") == "paragraph":
        if text.strip().isdigit():
            return False, "page_number"
        return False, "too_short"

    return True, "ok"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beslenme/egzersiz kitabını hasta-eğitimi için etiketler."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--labeled-output", type=Path, default=None)
    parser.add_argument("--valid-output", type=Path, default=None)
    parser.add_argument("--audience", default="patient")
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {input_path}")

    stem = input_path.name
    for suffix in (".raw.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    labeled_path = args.labeled_output or input_path.with_name(f"{stem}.labeled.jsonl")
    valid_path = args.valid_output or input_path.with_name(f"{stem}.valid.jsonl")

    records = load_jsonl(input_path)
    labeled: list[dict[str, Any]] = []
    valid_only: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    kept_by_chapter: Counter[str] = Counter()
    in_references = False

    for record in records:
        if record.get("heading_level") in (1, 2, 3):
            in_references = False

        if is_references_marker(record["text"]):
            in_references = True
            valid, reason = False, "references"
        elif in_references:
            valid, reason = False, "references"
        else:
            valid, reason = evaluate(record)

        reasons[reason] += 1
        enriched = {
            **record,
            "audience": args.audience,
            "valid": valid,
            "reason": reason,
        }
        labeled.append(enriched)
        if valid:
            kept_by_chapter[record.get("chapter", "")[:70]] += 1
            valid_only.append(enriched)

    save_jsonl(labeled, labeled_path)
    save_jsonl(valid_only, valid_path)

    print("--- BESLENME/EGZERSİZ ETİKETLEME ---")
    print(f"Toplam kayıt : {len(records)}")
    print(f"Geçerli      : {len(valid_only)}")
    print(f"Elenen       : {len(records) - len(valid_only)}")
    print(f"Gerekçeler   : {dict(reasons)}")
    print("Tutulan bölümler:")
    for chapter, count in kept_by_chapter.items():
        print(f"  [{count:4}] {chapter}")
    print(f"\nDenetim  : {labeled_path}")
    print(f"Chunk'a hazır: {valid_path}")


if __name__ == "__main__":
    main()
