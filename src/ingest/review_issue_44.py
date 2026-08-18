"""Sayı 44 için manuel kalite düzeltmeleri.

PDF düzenindeki iki bozuk kolon, dergi tanıtım parçaları ve güvenlik açısından
yanlış triage etiketleri extraction sonrasında burada normalize edilir.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_CHUNKS = {
    "00-diyabet-yasam-sayi-44_ch_010",
    "00-diyabet-yasam-sayi-44_ch_037",
}

CHUNK_31_CONTENT = """### AĞIZDAKİ BUMERANG ETKİSİ

Sindirim ağızda başlar. Diyabetli bireyde kanda glukoz düzeyi yüksekse; idrar miktarındaki artışla vücutta sıvı kaybı gelişir, buna bağlı tükürük miktarı da azalır, yapışkanlığı artar ve ağız kuruluğu olur. Besinlerin parçalanması ve yutulması zorlaşır. Kanda glukoz yüksekse tükürükte de yüksektir. Bu durum diş ve diş etlerinde, dilde dental plak (yiyecek artığı ve bakteri) birikimini artırır. Ek olarak tükürüğün miktarının azalması ve yoğunluğunun artması dental plak birikimini hızlandırır.

Dental plak diş çürüklerinin ve diş eti hastalıklarının başlıca nedenidir.

Tükürük az ise ağzın fizyolojik temizliği zorlaşır. Bu duruma kişinin ağız hijyenine önem vermemesi de eklenirse biriken dental plak diş çürüklerine ve diş eti hastalıklarına yol açar.

Dental plak ağız kokusuna da neden olur.

Çoğunlukla diyabetle yaşamda ağız sağlığı diğer sorunların yanında pek gündeme gelmez. Oysa diş çürükleri ve diş eti hastalıkları oluşabilir; bu nedenle düzenli diş hekimi kontrolü ve ağız hijyeni önemlidir."""

CHUNK_32_CONTENT = """### AĞIZDAKİ BUMERANG ETKİSİ

Tokluk hissi yemek süresi 20 dakikayı geçtikten sonra gelişeceğinden daha çok yemek istenebilir.

Gerekenden fazla alınan besin kan glukoz dengesini bozar ve kilo artışına yol açabilir.

Ağrısız, acısız ve uzun süre çiğneyerek beslenebilen kişide planlanan öğünün uygulanması kolaylaşır; kilo artışı daha kolay önlenebilir ve diyabetin kontrolü kolaylaşır.

Diyabetin ağızdaki olumsuz yansımalarına rağmen ağız sağlığını koruyan diyabetli bireylerin diyabet yönetimi daha başarılı olur."""


def clean_institutional_terms(text: str) -> str:
    """RAG için gereksiz kurum/şehir adlarını çıkarır."""
    text = re.sub(r"\bTürkiye Diyabet Vakfı\b", "", text)
    text = re.sub(r"\bDiyabet Vakfı\b", "", text)
    text = re.sub(r"\bİstanbul\b", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sayı 44 chunk kalite düzeltmeleri")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reviewed: list[dict] = []
    for row in rows:
        if row["chunk_id"] in BAD_CHUNKS:
            continue

        row["content"] = clean_institutional_terms(row["content"])
        if row["paragraph_index_start"] == 375:
            row["content"] = CHUNK_31_CONTENT
        elif row["paragraph_index_start"] == 410:
            row["content"] = CHUNK_32_CONTENT

        if row["paragraph_index_start"] == 117:
            row["triage_level"] = "YELLOW"
        elif row["paragraph_index_start"] == 435:
            row["triage_level"] = "YELLOW"
            row["content"] = row["content"].replace(
                "Tip 1 diyabet hastaları, egzersiz öncesi insülin dozajını azaltabilirler.",
                "İnsülin dozunda değişiklik yalnızca hekim önerisiyle yapılmalıdır.",
            )
        elif row["paragraph_index_start"] == 535:
            row["triage_level"] = "EMERGENCY"

        reviewed.append(row)

    for number, row in enumerate(reviewed, start=1):
        row["chunk_id"] = f"{row['document_id']}_ch_{number:03d}"

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed),
        encoding="utf-8",
    )
    print(f"Düzeltilen chunk sayısı: {len(reviewed)}")


if __name__ == "__main__":
    main()
