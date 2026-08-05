from __future__ import annotations

"""
TEMD Diyabet Teknolojileri Kılavuzu için hasta-eğitimi etiketleyici.

Kapsam (dar, zero-leak):
  Bölüm 1 — Kapiller glukoz ölçüm / akıllı kalem (genel)
  Bölüm 2 — SGM / CGM (TIR, AGP, kullanım farkındalığı)
  Bölüm 5 — Yazılım/uygulama (takip; doz hesaplayıcı hariç)

Dışarıda: pompa doz ayarı (Bölüm 3), profesyonel cihaz kataloğu (4),
araştırma sonuçları (6), gelecek öngörüleri (7), kaynakçalar, ünite/formül.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

CHAPTER_WHITELIST = {"1", "2", "5"}
CHAPTER_NUMBER = re.compile(r"Bölüm\s+(\d+)\b", re.IGNORECASE)

# Alt başlık / metin: doz, formül, endikasyon, ağır klinik.
DROP_SUBSECTION_RE = re.compile(
    r"doz|bazal\s+oran|bolus|kh\s*/\s*i|idf|endikasyon|kontrendike|"
    r"mard|tarihçe|çalışma\s+prensip|sistemle\s+etkileşen|"
    r"karbonhidrat\s+sayım|insülin\s+doz\s+hesap",
    re.IGNORECASE,
)

DROP_CONTENT_PATTERNS = [
    # Doz / formül / ünite
    r"\bünite\b",
    r"\bdoz\s*hesap",
    r"\bdozaj",
    r"kh\s*/\s*i",
    r"\bidf\b",
    r"1700\s*/",
    r"500\s*/",
    r"\btitrasyon",
    r"\bbolus\b",
    r"bazal\s+(oran|hız|doz)",
    r"\d+\s?mg(?!\s?/\s?d[lL])",  # 500 mg tablet; mg/dL korunur
    # Pompa / AID klinik ayar
    r"otomatik\s+mod",
    r"manuel\s+mod",
    r"kapalı\s+loop",
    r"\baid\b",
    r"sc[iİiı]{2}",
    r"pompa\s+entegrasyon",
    # Profesyonel metrik / marka katalog
    r"\bmard\b",
    r"mean\s+absolute\s+relative",
    r"eversense|dexcom|guardian\s+connect|libre\s+[23]|sibionics|medtronic|"
    r"abbott\s+lingo|libre\s+rio|touch\s+care",
    # Araştırma / atıf dili
    r"\bçalışma\b",
    r"randomize",
    r"meta-?analiz",
    r"\bet al",
    r"kohort",
    r"prevalans|insidans|mortalite",
    # Moleküler / laboratuvar ağır jargon (hasta için gereksiz)
    r"glukoz\s+oksidaz",
    r"dehidrogenaz",
    r"biosensör",
    r"hidrojen\s+peroksid",
    r"ikodekstrin",
    r"hematokrit",
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

    # Bölüm başlık satırlarını (H1) her zaman tut — bağlam için.
    if record.get("heading_level") == 1:
        return True, "ok"

    # Alt başlık satırları: drop listesindeyse ele, değilse tut.
    if record.get("heading_level") in (2, 3):
        lowered_heading = tr_lower(record["text"])
        if DROP_SUBSECTION_RE.search(lowered_heading):
            return False, "drop_heading"
        if DROP_CONTENT_RE.search(lowered_heading):
            return False, "drop_heading_content"
        return True, "ok"

    text = record.get("text", "")
    if DROP_CONTENT_RE.search(tr_lower(text)):
        return False, "drop_content"

    # Çok kısa gürültü / tablo yıldız notları
    if len(text.strip()) < 40 and record.get("block_type") == "paragraph":
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
        description="TEMD diyabet teknolojileri kılavuzunu hasta-eğitimi için etiketler."
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
        heading_level = record.get("heading_level")
        if heading_level in (1, 2, 3):
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

    print("--- TEKNOLOJİ ETİKETLEME RAPORU ---")
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
