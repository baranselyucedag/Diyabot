from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Yapılandırma — hangi bölümler hasta eğitimi için geçerli ("valid") sayılır.
# TEMD kılavuzunda H1 başlıkları sonda iki haneli bölüm no taşır:
#   "... GLİSEMİK KONTROL HEDEFLERİ04" -> 04
# ---------------------------------------------------------------------------
# Seçenek 3 (dar, güvenli alt küme): yalnızca hasta eğitimine en yakın 4 bölüm.
# Amaç: hasta chatbotuna klinik/araştırma/doz/tanı sızıntısı OLMAMASI.
CHAPTER_WHITELIST = {
    "05",  # Tıbbi Beslenme Tedavisi (genel ilkeler)
    "06",  # Egzersiz ve Fizik Aktivite (genel ilkeler)
    "12",  # Akut Komplikasyonlar (hipo/hiper belirti farkındalığı)
    "14",  # Diyabetik Ayak (önleme)
}

CHAPTER_NUMBER = re.compile(r"(\d{2})\s*$")

# ---------------------------------------------------------------------------
# Katı içerik filtresi: hekim/araştırma/doz/tanı diline dair HERHANGİ bir
# sinyal taşıyan kayıt elenir. Zero-leak hedefi için geniş tutulmuştur
# (recall'dan çok precision önceliklidir).
# ---------------------------------------------------------------------------
CLINICAL_DROP_PATTERNS = [
    # Araştırma / kanıt / atıf
    r"\bçalışma", r"\baraştırma", r"\bfaz\s*\d", r"randomize", r"plasebo",
    r"meta-?analiz", r"\bet al", r"(19|20)\d{2}\s*;\s*\d", r"kohort",
    r"monoklonal|antikor|otoantikor|sekretogog|immün|monoterapi",
    # İlaç / molekül adları
    r"rituximab|barisitinib|metformin|sglt-?2|glp-?1|dpp-?4|pioglitazon|"
    r"sulfonil[üu]re|glinid|akarboz|alpelisib|inavolisib|"
    r"insülin\s+(glarjin|detemir|aspart|lispro|glulisin|degludek|regüler)|"
    r"ace-?i\b|arb\b|statin|antihipertansif|diüretik",
    # Doz / uygulama
    r"\bmg\s*/\s*(gün|kg)\b", r"\bünite\b", r"\b\d+\s*ü\b", r"titrasyon",
    r"\bdoz", r"\bbolus", r"enjeksiyon|enteral|infüzyon|ampul|flakon",
    # Tanı / sınıflama / evreleme
    r"tanı\s+kriter|tanısı\s+kon|klinik\s+bir\s+tanı|sınıflama|sınıfland[ıi]r|"
    r"\bevre\s*\d|wagner|megitt|prognoz",
    # Kanıt düzeyi / hekim-audience
    r"temd\s+öneriler|kanıt\s+düzey|\(\s*[abc]\s*\)|interdisipliner|reçete|"
    r"konsültasyon|endike|kontrendike|hastane|bariyatrik|cerrah|ameliyat|"
    r"amputasyon|revaskül|debridman|biyopsi|mortalite|morbidite|prevalans|insidans",
    # Klinik lab jargonu
    r"egfr|mmol\s*/\s*l|meq\s*/\s*l|anyon\s+açı|kreatinin|album[iü]n[üu]ri|"
    r"c-?peptid|ivgtt|serum\s+\w",
    # Patoloji / görüntüleme / tanısal muayene
    r"osteomiyelit|selülit|gangren|nekroz|\biskemi|nöropati|periferik\s+arter|"
    r"arteriyel|sintigrafi|nükleer\s+tıp|radyograf|manyetik\s+rezonans|"
    r"görüntüleme|doppler|monofilament|sensitif|spesifik|palpasyon|muayene\s+form",
]
CLINICAL_DROP_RE = re.compile("|".join(CLINICAL_DROP_PATTERNS), re.IGNORECASE)


def tr_lower(text: str) -> str:
    """Türkçe güvenli küçük harf."""
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
    """H1 başlığının sonundaki iki haneli bölüm numarasını döndürür."""
    match = CHAPTER_NUMBER.search(chapter_title or "")
    return match.group(1) if match else None


def is_treatment_subsection(subsection_title: str) -> bool:
    """
    Alt başlık 'tedavi/ilaç' odaklı mı? (Kapsam dışı: doz/protokol/ilaç.)

    Dikkat: '... tedavisinde beslenme' gibi başlıkları YANLIŞLIKLA elememek
    için kelime sınırı/çapa kullanılır ('tedavisinde' eşleşmez).
    """
    title = tr_lower(subsection_title or "")

    if not title:
        return False

    rules = [
        r"\|\s*tedav[iı]\b",          # "12.1.4. | TEDAVİ"
        r"\btedavisi(\s+ve\s+yönetimi)?\s*$",  # "... tedavisi" / "... tedavisi ve yönetimi"
        r"\byönetimi\s*$",            # "... riskinin ... yönetimi"
        r"antibiyotik",
        r"revaskül",
        r"anti[-\s]?hiperglisemik\s+ila",  # "ANTİ-HİPERGLİSEMİK İLAÇLAR"
        r"insülin\s+tedavis",
    ]
    return any(re.search(rule, title) for rule in rules)


# Katı mod: alt başlığın KENDİSİ klinik/araştırma odaklıysa tüm alt bölüm
# (başlık + gövde tabloları dahil) elenir. 14.3 SINIFLAMA gibi evreleme/tanı
# tablolarının sızmasını kökten engeller.
CLINICAL_SUBSECTION_RE = re.compile(
    r"sınıflama|sınıfland|değerlendir|evrele|\btanı|\bklinik|enfeksiyon|"
    r"patogenez|patofizyoloji|epidemiyoloji|prognoz|\bülser|amputasyon|"
    r"yönetim|tedavi|tarama|prevalans|insidans|osteomiyelit|selülit|gangren|"
    r"iskemi|nöropati|arter|insülin|duyarlılık\s+faktör|düzeltme\s+bolus",
    re.IGNORECASE,
)


def is_clinical_subsection(subsection_title: str) -> bool:
    title = tr_lower(subsection_title or "")
    return bool(title) and bool(CLINICAL_SUBSECTION_RE.search(title))


# Gerçek ilaç dozu sinyalleri. mg/dL, mmol/L gibi LAB değerleri KORUNUR.
DRUG_DOSE_PATTERNS = [
    re.compile(r"\d+\s?mg\s?/\s?(gün|kg)", re.IGNORECASE),
    re.compile(r"\d+\s?(ünite|iu)\b", re.IGNORECASE),
    re.compile(r"\btitrasyon", re.IGNORECASE),
    re.compile(r"\d+\s?mg(?!\s?/\s?d)", re.IGNORECASE),  # "500 mg" ama "mg/dL" değil
]


def has_drug_dose(text: str) -> bool:
    """Paragrafta açık ilaç dozu var mı? (Lab birimleri hariç.)"""
    lowered = tr_lower(text)

    # Lab bağlamı baskınsa (mg/dl, mmol) doz sayma.
    if re.search(r"mg\s?/\s?dl|mmol", lowered):
        # Yine de mg/gün gibi net doz varsa yakala.
        if not re.search(r"mg\s?/\s?(gün|kg)|\d+\s?(ünite|iu)\b|titrasyon", lowered):
            return False

    return any(pattern.search(text) for pattern in DRUG_DOSE_PATTERNS)


def is_references_marker(text: str) -> bool:
    """Bölüm sonu 'KAYNAKLAR' başlığını tanır (bölge başlatır)."""
    return tr_lower(text or "").strip() == "kaynaklar"


def is_inline_reference(text: str) -> bool:
    """'Kaynaklar: Hoelzel W, ...' gibi tek satırlık atıf notunu tanır."""
    return bool(re.match(r"\s*kaynak(lar)?\s*:", tr_lower(text or "")))


def clinical_content_hit(text: str) -> str | None:
    """Metinde hekim/araştırma/doz/tanı sinyali varsa eşleşen ifadeyi döndürür."""
    match = CLINICAL_DROP_RE.search(tr_lower(text or ""))
    return match.group(0) if match else None


def evaluate(record: dict[str, Any], drug_filter: bool, strict: bool) -> tuple[bool, str]:
    """Kaydın geçerliliğini ve gerekçesini döndürür (durumsuz kurallar)."""
    number = chapter_number(record.get("chapter", ""))

    if number not in CHAPTER_WHITELIST:
        return False, f"chapter_out_of_scope({number})"

    if is_treatment_subsection(record.get("subsection", "")):
        return False, "treatment_subsection"

    if strict and is_clinical_subsection(record.get("subsection", "")):
        return False, "clinical_subsection"

    if is_inline_reference(record["text"]):
        return False, "references_inline"

    if drug_filter and record["block_type"] == "paragraph" and has_drug_dose(record["text"]):
        return False, "drug_dose"

    # Katı mod: bölüm başlığı satırları hariç, klinik sinyalli her kayıt elenir.
    if strict and record.get("heading_level") is None:
        hit = clinical_content_hit(record["text"])
        if hit:
            return False, "clinical_content"

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
        description=(
            "Yapısal JSONL kayıtlarını hasta-eğitimi kapsamına göre 'valid' "
            "etiketler; geçerli olanları chunk'a hazır ayrı dosyaya yazar."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="*.raw.jsonl yolu.")
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=None,
        help="Tüm kayıtlar + valid/etiket (denetim için). Varsayılan: <stem>.labeled.jsonl",
    )
    parser.add_argument(
        "--valid-output",
        type=Path,
        default=None,
        help="Sadece valid kayıtlar (chunk'a hazır). Varsayılan: <stem>.valid.jsonl",
    )
    parser.add_argument(
        "--audience",
        default="clinical_guideline",
        help="Kayıtlara eklenecek audience etiketi.",
    )
    parser.add_argument(
        "--no-drug-filter",
        action="store_true",
        help="Paragraf içi ilaç-dozu filtresini kapatır (yalnız bölüm/alt başlık).",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Katı klinik-içerik filtresini kapatır (varsayılan: açık, zero-leak).",
    )
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
    drug_filter = not args.no_drug_filter
    strict = not args.no_strict

    labeled: list[dict[str, Any]] = []
    valid_only: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    kept_by_chapter: Counter[str] = Counter()

    # Bölüm sonu KAYNAKLAR listesi: başlıktan sonraki atıf satırları bir sonraki
    # gerçek başlığa kadar kaynakça bölgesidir; RAG için tümü elenir.
    in_references = False

    for record in records:
        heading_level = record.get("heading_level")

        # Gerçek bir başlık görülünce kaynakça bölgesi biter.
        if heading_level in (1, 2, 3):
            in_references = False

        if is_references_marker(record["text"]):
            in_references = True
            valid, reason = False, "references"
        elif in_references:
            valid, reason = False, "references"
        else:
            valid, reason = evaluate(record, drug_filter, strict)

        reasons[reason] += 1

        enriched = {**record, "audience": args.audience, "valid": valid, "reason": reason}
        labeled.append(enriched)

        if valid:
            kept_by_chapter[record.get("chapter", "")[:50]] += 1
            # chunk_jsonl.py şemasıyla uyumlu, ama ek alanları da taşır.
            valid_only.append(enriched)

    save_jsonl(labeled, labeled_path)
    save_jsonl(valid_only, valid_path)

    print("--- ETİKETLEME RAPORU ---")
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
