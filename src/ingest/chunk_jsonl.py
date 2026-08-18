from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from text_clean import clean_source_text


# Embedding modelleri için güvenli, hedef chunk boyutu.
MAX_CHARS = 1600

# Bunun altındaki içerikler genelde kapak/yalnız başlıktır; chunk oluşturulmaz.
MIN_CONTENT_CHARS = 80

IMPORTANT_MARKERS = {
    "ÖNEMLİ NOT!",
    "DİKKAT!",
    "UNUTMAYIN!",
    "ÖNERİ",
    "BİLGİ KUTUSU",
    "UYARI!",
    "ACİL!",
}

TOC_MARKER = "İÇİNDEKİLER"
TOC_END_MARKER = "KAYNAKLAR"

# RAG cevabına değer katmayan karşılama/motivasyon metinleri.
SKIP_PATTERNS = [
    r"^değerli katılımcı,?$",
    r"^öncelikle .* teşekkür ederiz\.?$",
    r"^lütfen kendinize inanın.*$",
]

# "BÖLÜM 1: ...", "1. BÖLÜM: ...", "BÖLÜM: ..." türü üst başlıklar.
CHAPTER_PATTERN = re.compile(
    r"^(?:(?:\d+\.)?\s*)?bölüm(?:\s+\d+)?\s*:\s*.+$",
    re.IGNORECASE,
)

# TEMD / Metabolizma kitapları: "Bölüm 13 DİYABETES MELLİTUS VE EGZERSİZ"
# (iki nokta üst üste yok).
BOLUM_CHAPTER_PATTERN = re.compile(
    r"^Bölüm\s+\d+\b.+",
    re.IGNORECASE,
)

# "3. PREDİYABET VE ...", "4. PREDİYABET VE ..." — BÖLÜM kelimesi olmadan
# tek seviyeli major bölümler (alt başlık 3.1 ile karışmaz).
MAJOR_NUMBERED_CHAPTER_PATTERN = re.compile(
    r"^\d+\.\s+(?!.*\d+\.\d+)[A-ZÇĞİÖŞÜ].{20,}$"
)

# FITT çerçevesi satırları; bölünmemeli.
FITT_LINE_PATTERN = re.compile(
    r"^[FITT]\s*\(\s*(?:Frequency|Intensity|Time|Type)\b",
    re.IGNORECASE,
)

# "1.1 Başlık", "1.1. Başlık", "1.6.2.1 Başlık" türü alt başlıklar.
# En az bir nokta zorunlu: "1. Kilo yönetimi" gibi liste maddelerini dışarıda bırakır.
NUMBERED_HEADING_PATTERN = re.compile(
    r"^\d+(?:\.\d+)+\.?\s+\S.+$"
)

# Numarasız ama belgelerde bölüm olarak kullanılan başlıklar.
NAMED_HEADING_PATTERN = re.compile(
    r"^(ÖNSÖZ|GİRİŞ(?:\s+VE\s+AMAÇ)?|REHBERİN\s+AMACI|HEDEF\s+KİTLE|"
    r"KAYNAKLAR|EKLER?)$",
    re.IGNORECASE,
)

# Sadece büyük harf olan kısa başlıklar; uzun metinleri yanlışlıkla başlık saymamak için sınırlı.
UPPERCASE_HEADING_PATTERN = re.compile(
    r"^[A-ZÇĞİÖŞÜ0-9][A-ZÇĞİÖŞÜ0-9\s:,\-–—()/.!?]{4,100}$"
)

# Yalnızca gerçek acil yönlendirme ifadeleri.
# "acil müdahale" tek başına eğitim metninde sık geçer; EMERGENCY sayılmaz.
EMERGENCY_ACTION_TERMS = [
    "112",
    "acil servis",
    "acil servise sevk",
    "acile başvur",
    "derhal acil",
    "tıbbi yardım alın",
]

YELLOW_TERMS = [
    "doktorunuza danış",
    "doktorunuza başvur",
    "hekiminize danış",
    "hekiminize başvur",
    "doktor kontrolü",
    "doktor kontrolünden",
    "sağlık profesyoneline başvur",
    "tıbbi yardım alın",
    "acile başvur",
    "hemen durun",
]

# Gerçek ara başlıklarda sık geçen kökler (Türkçe ekler için \w* kullanılır).
INLINE_HEADING_CUE = re.compile(
    r"(?i)("
    r"tanım\w*|önemi|kriter\w*|tedavi\w*|belirti\w*|"
    r"risk\s+faktör\w*|epidemiyoloji\w*|patofizyoloji\w*|amaç\w*|"
    r"kapsam\w*|hedef\s+kitle|sınıflandır\w*|önleme|tarama|"
    r"komplikasyon\w*|karşılaştır\w*|prensip\w*|"
    r"hipoglisemi\w*|hiperglisemi\w*|ketoasidoz\w*|retinopati\w*|nöropati\w*|nefropati\w*|"
    r"diyabetik\s+ayak|insülin\s+tedavi\w*|oral\s+antidiyabetik\w*|"
    r"değiştirile(?:mez|bilir)|farmakolojik\w*|antropometrik\w*|"
    r"dikkat\s+edilecek|sık\s+sorulan|fiziksel\s+aktivite|"
    r"beslenme\s+prensip\w*|kendi\s+kendine\s+izlem|"
    r"stres\s+yönetimi|uyku\s+düzeni|boy\s+ölçümü|kilo\s+ölçümü|"
    r"bel\s+çevresi|vücut\s+kitle|ölçüm\s+hataları|"
    r"uygulamanın|motivasyonel|davranış\s+değişikliği|"
    r"görev\s+tanımı|dokümantasyon|iletişimin\s+önemi|"
    r"psikolojik\s+destek|yolculuk|seyahat|sosyal\s+hak\w*|cinsel\s+yaşam|"
    r"aşılama|dini\s+uygulama|sigara|alkol\s+kullan|ayak\s+bakım\w*|ayakkabı\w*|"
    r"tırnak\w*|cilt\s+bakım\w*|ağız\w*|diş\s+sağlığı|sinir\s+hasar\w*|damar\s+hasar\w*|"
    r"ayak\s+sorun\w*|iş\s+hayatı|çalışma\s+yaşamı|"
    r"obezite\s+cerrahi|obezite\w*|cerrahi\s+aday|kilo\s+kayb\w*|kilo\s+verme|kilo\s+kontrol"
    r")"
)

# Liste maddesi / amaç cümlesi gibi görünen satırları başlık sayma.
LIST_ITEM_PATTERN = re.compile(
    r"(?i)("
    r"olanlarda|olan\s+kadınlarda|olan\s+bireylerde|bireylerde$|"
    r"varlığında$|olanlarda$|"
    r"için\s+\w{3,}$|"
    r"yetkinleştirmek$|kazandırmak$|güçlendirmek$|sağlamak$|"
    r"sunmak$|geliştirmek$|yapılmalıdır$|edilmelidir$|"
    r"önerilmektedir$|bulunmalıdır$"
    r")"
)

# Tek başına başlık olmayan kısa egzersiz / madde isimleri.
NON_HEADING_SHORT_PHRASES = {
    "yoga",
    "tai chi",
    "pilates",
    "jogging",
    "tempolu yürüyüş",
    "bisiklete binme",
    "yüzme",
    "dans etme",
    "merdiven çıkma",
    "sağlıklı beslenme",
    "düzenli fiziksel aktivite",
    "kilo kontrolü",
    "düzenli sağlık kontrolleri",
    "sigara ve alkolden uzak durma",
}


def normalize_text(text: str) -> str:
    """Kaynak artefaktlarını temizler; tablo satır sonlarını korur."""
    text = clean_source_text(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def tr_lower(text: str) -> str:
    """Türkçe İ/I harflerini güvenli biçimde küçük harfe çevirir."""
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


def is_references_heading(text: str) -> bool:
    """Kaynaklar bölümü başlığını tanır; RAG için atlanır."""
    normalized = tr_lower(normalize_text(text))
    if normalized == "kaynaklar":
        return True
    return bool(re.search(r"bölüm\s*:?\s*kaynaklar\s*$", normalized))


def should_skip(text: str) -> bool:
    """RAG için değersiz karşılama ve motivasyon metinlerini belirler."""
    normalized = normalize_text(text)
    return any(
        re.match(pattern, normalized, flags=re.IGNORECASE)
        for pattern in SKIP_PATTERNS
    )


def is_important_marker(text: str) -> bool:
    """Uyarı veya bilgi kutusu işaretçisini tanır."""
    return normalize_text(text).upper() in IMPORTANT_MARKERS


def is_chapter_heading(text: str) -> bool:
    """Üst seviye bölüm başlığı formatlarını tanır."""
    normalized = normalize_text(text)

    if CHAPTER_PATTERN.match(normalized):
        return True

    if BOLUM_CHAPTER_PATTERN.match(normalized):
        return True

    # Hiyerarşik alt başlıkları (3.1, 1.6.2) chapter sayma.
    if NUMBERED_HEADING_PATTERN.match(normalized):
        return False

    if not MAJOR_NUMBERED_CHAPTER_PATTERN.match(normalized):
        return False

    # Büyük harf oranı yüksek olan major başlıklar (kaynak yazım hatalı olsa bile).
    letters = [char for char in normalized if char.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio >= 0.55


def is_fitt_line(text: str) -> bool:
    """FITT (Frequency/Intensity/Time/Type) satırını tanır."""
    first_line = normalize_text(text).split("\n", 1)[0]
    return bool(FITT_LINE_PATTERN.match(first_line))


def merge_fitt_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ardışık FITT satırlarını tek kayıtta birleştirir; bölünmeyi engeller."""
    if not records:
        return []

    merged: list[dict[str, Any]] = []
    index = 0

    while index < len(records):
        record = records[index]

        if not is_fitt_line(record["text"]):
            merged.append(record)
            index += 1
            continue

        group = [record]
        index += 1

        while index < len(records) and is_fitt_line(records[index]["text"]):
            group.append(records[index])
            index += 1

        if len(group) == 1:
            merged.append(group[0])
            continue

        merged.append(
            {
                **group[0],
                "text": "\n\n".join(item["text"] for item in group),
                "paragraph_index": group[0]["paragraph_index"],
            }
        )

    return merged


def is_numbered_heading(text: str) -> bool:
    """Hiyerarşik numaralı başlıkları liste maddelerinden ayırır."""
    normalized = normalize_text(text)

    if len(normalized) > 160:
        return False

    return bool(NUMBERED_HEADING_PATTERN.match(normalized))


def is_named_heading(text: str) -> bool:
    """Önsöz, giriş ve kaynaklar gibi bilinen başlıkları tanır."""
    return bool(NAMED_HEADING_PATTERN.match(normalize_text(text)))


def is_uppercase_heading(text: str) -> bool:
    """Kısa, tamamı büyük harfli gerçek bölüm başlıklarını tanır."""
    normalized = normalize_text(text)

    if is_important_marker(normalized):
        return False

    if len(normalized) > 100 or any(char.islower() for char in normalized):
        return False

    forbidden = {
        "EĞİTİM VE UYGULAMA REHBERİ",
        "2026",
        TOC_MARKER,
        TOC_END_MARKER,
    }

    if normalized.upper() in forbidden:
        return False

    return bool(UPPERCASE_HEADING_PATTERN.match(normalized))


def is_inline_heading(text: str) -> bool:
    """
    Word'de Heading stili verilmemiş kısa ara başlıkları tanır.

    Liste maddelerini (Tai Chi, PKOS olan kadınlarda, amaç cümleleri)
    başlık saymamak için hem biçim hem içerik ipucu ister.
    """
    normalized = normalize_text(text)

    if (
        not normalized
        or is_important_marker(normalized)
        or len(normalized) < 8
        or len(normalized) > 80
        or normalized[-1] in ".?!:;,"
        or normalized.startswith(("*", "-", "•", "S:", "C:"))
        or re.match(r"^\d+[\.\)]\s+", normalized)
        or LIST_ITEM_PATTERN.search(normalized)
        or normalized.lower() in NON_HEADING_SHORT_PHRASES
    ):
        return False

    words = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", normalized)

    if not 2 <= len(words) <= 8:
        return False

    # En az iki kelime büyük harfle başlamalı (Title Case benzeri).
    uppercase_word_count = sum(word[0].isupper() for word in words)
    if uppercase_word_count < 2:
        return False

    # Bilinen başlık ipucu yoksa liste maddesi kabul edilir.
    return bool(INLINE_HEADING_CUE.search(normalized))


def get_heading_type(text: str) -> str | None:
    """Metnin başlık türünü döndürür; başlık değilse None döndürür."""
    if is_important_marker(text):
        return None

    if is_chapter_heading(text):
        return "chapter"

    if is_numbered_heading(text):
        return "section"

    if is_named_heading(text):
        return "section"

    if is_uppercase_heading(text):
        return "section"

    if is_inline_heading(text):
        return "section"

    return None


def format_piece(text: str) -> str:
    """Özel işaretçileri Markdown vurgusuyla biçimlendirir."""
    text = normalize_text(text)

    if is_important_marker(text) or text.startswith("S:"):
        return f"**{text}**"

    return text


def split_text_at_sentence_boundaries(text: str, limit: int) -> list[str]:
    """Aşırı uzun tek paragrafı mümkünse cümle sınırlarından böler."""
    text = normalize_text(text)

    if len(text) <= limit:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            parts.append(current)
            current = ""

        # Tek bir cümle bile limitten uzunsa, kelime sınırında kes.
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)

            if cut <= 0:
                cut = limit

            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()

        current = sentence

    if current:
        parts.append(current)

    return parts


def split_paragraph_record(record: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    """Uzun paragrafı aynı kaynak indeksiyle birden fazla kayda ayırır."""
    text = normalize_text(record["text"])
    parts = split_text_at_sentence_boundaries(text, limit)

    return [{**record, "text": part} for part in parts]


def choose_topic(section: str, content: str) -> str:
    """
    Konuyu öncelikle section başlığından belirler.
    Aciliyet topic değildir; triage_level ile temsil edilir.
    """
    title = tr_lower(section)

    # Farmakolojik önleme / tedavi ilaç konusudur (önleme kelimesi prediyabet'e kaçmasın).
    if "farmakolojik" in title:
        return "ilaç_bilgisi"

    # Tip sınıflandırması / risk faktörleri ilaç konusu değildir.
    if re.search(r"tip\s*[12]|gestasyonel\s+diyabet|risk\s+faktör", title):
        return "genel_bilgi"

    # Tanım / belirti / önem başlıkları (özel konu yoksa) genel bilgi sayılır.
    if re.search(r"tanım|belirti|önemi", title):
        if "prediyabet" in title:
            return "prediyabet"
        if not re.search(
            r"beslenme|egzersiz|fiziksel aktivite|ilaç|insülin tedavi|komplikasyon",
            title,
        ):
            return "genel_bilgi"

    # Başlık kuralları: spesifikten genele (ilaç, komplikasyondan önce).
    title_rules: list[tuple[str, list[str]]] = [
        (
            "prediyabet",
            ["prediyabet", "gizli şeker", "önleme"],
        ),
        (
            "beslenme",
            ["beslenme", "diyet", "öğün", "porsiyon", "tıbbi beslenme"],
        ),
        (
            "egzersiz",
            [
                "egzersiz",
                "fiziksel aktivite",
                "aerobik",
                "direnç",
                "yürüyüş",
                "esneklik",
            ],
        ),
        (
            "ilaç_bilgisi",
            [
                "ilaç",
                "antidiyabetik",
                "metformin",
                "insülin tedavi",
                "insülin kullanımı",
                "oral antidiyabetik",
                "farmakolojik",
                "farmakolojik önleme",
                "medikal tedavi",
                "tedavi yaklaşım",
            ],
        ),
        (
            "komplikasyon",
            [
                "komplikasyon",
                "retinopati",
                "nöropati",
                "nefropati",
                "ketoasidoz",
                "diyabetik ayak",
                "ülser",
                "hipoglisemi",
                "hiperglisemi",
            ],
        ),
        (
            "glukoz_takibi",
            [
                "tanı kriter",
                "hba1c",
                "ogtt",
                "kan şekeri takip",
                "glukoz takip",
                "ölçüm",
                "izlem",
                "bireysel izlem",
            ],
        ),
    ]

    for topic, keywords in title_rules:
        if any(keyword in title for keyword in keywords):
            return topic

    # Başlık nötrse içerikten daha temkinli skorla.
    content_keywords = {
        "beslenme": ["beslenme tedavisi", "diyet", "öğün", "porsiyon"],
        "egzersiz": ["fiziksel aktivite", "egzersiz", "aerobik", "direnç egzersiz"],
        "ilaç_bilgisi": [
            "oral antidiyabetik",
            "insülin tedavisi",
            "metformin",
            "farmakolojik",
        ],
        "komplikasyon": [
            "komplikasyon",
            "retinopati",
            "nöropati",
            "nefropati",
            "ketoasidoz",
        ],
        "glukoz_takibi": ["hba1c", "ogtt", "açlık kan şekeri", "kan şekeri ölç"],
        "prediyabet": ["prediyabet", "gizli şeker"],
    }

    body = tr_lower(content)
    scores = {
        topic: sum(body.count(keyword) for keyword in keywords)
        for topic, keywords in content_keywords.items()
    }
    best_topic, best_score = max(scores.items(), key=lambda item: item[1])

    return best_topic if best_score > 0 else "genel_bilgi"


def choose_triage_level(content: str) -> str:
    """
    Triyaj seviyesini üretir.

    Eğitim metninde semptom adı geçmesi EMERGENCY sayılmaz.
    Yalnızca eylem / yönlendirme dili varsa yükseltilir.
    """
    text = tr_lower(content)

    if any(term in text for term in EMERGENCY_ACTION_TERMS):
        return "EMERGENCY"

    if any(term in text for term in YELLOW_TERMS):
        return "YELLOW"

    return "GREEN"


def build_content(section_path: str, records: list[dict[str, Any]]) -> str:
    """Bölüm bağlamını koruyarak chunk'ın Markdown içeriğini oluşturur."""
    lines = [f"### {section_path}", ""]

    for record in records:
        lines.extend([format_piece(record["text"]), ""])

    return "\n".join(lines).strip()


def split_section_into_chunks(
    section_path: str,
    records: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Bir bölümün paragraflarını MAX_CHARS sınırını aşmadan gruplar."""
    if not records:
        return []

    records = merge_fitt_records(records)

    header_size = len(f"### {section_path}\n\n")
    usable_limit = max(MAX_CHARS - header_size, 300)

    expanded_records: list[dict[str, Any]] = []
    for record in records:
        # FITT bloğu atomik kalır; cümle bölünmesine uğramaz.
        if is_fitt_line(record["text"]):
            expanded_records.append(record)
        else:
            expanded_records.extend(split_paragraph_record(record, usable_limit))

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = header_size

    for record in expanded_records:
        piece = format_piece(record["text"]) + "\n\n"

        if current and current_size + len(piece) > MAX_CHARS:
            groups.append(current)
            current = []
            current_size = header_size

        current.append(record)
        current_size += len(piece)

    if current:
        groups.append(current)

    return groups


def load_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Ham JSONL kayıtlarını doğrulayarak belleğe yükler."""
    required_fields = {
        "document_id",
        "source_file",
        "paragraph_index",
        "text",
    }
    records: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            missing = required_fields - record.keys()

            if missing:
                raise ValueError(
                    f"{input_path.name}, satır {line_number}: "
                    f"eksik alanlar: {sorted(missing)}"
                )

            record["text"] = normalize_text(str(record["text"]))

            if record["text"]:
                records.append(record)

    return records


def remove_table_of_contents(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    İÇİNDEKİLER ile onu bitiren KAYNAKLAR arasındaki kayıtları çıkarır.

    Hemşire rehberinde İçindekiler gerçek başlıklar içerdiği için,
    bu bölüm işlenirse aktif chapter yanlışlıkla son listedeki bölüm olur.
    """
    toc_start = next(
        (
            index
            for index, record in enumerate(records)
            if normalize_text(record["text"]).upper() == TOC_MARKER
        ),
        None,
    )

    if toc_start is None:
        return records

    toc_end = next(
        (
            index
            for index in range(toc_start + 1, len(records))
            if normalize_text(records[index]["text"]).upper() == TOC_END_MARKER
        ),
        None,
    )

    # İçindekiler sonu güvenle bulunamadıysa veri kaybetmemek için dokunma.
    if toc_end is None:
        return records

    return records[toc_end + 1 :]


def make_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ham paragrafları başlık bağlamlı, güvenli RAG chunk'larına dönüştürür."""
    if not records:
        return []

    records = remove_table_of_contents(records)

    if not records:
        return []

    document_id = records[0]["document_id"]
    source_file = records[0]["source_file"]

    chunks: list[dict[str, Any]] = []
    chapter = ""
    section = "Giriş"
    buffer: list[dict[str, Any]] = []
    chunk_number = 1

    def section_path() -> str:
        """Aktif üst bölüm ve alt bölümden okunabilir yol üretir."""
        if chapter and section and section != chapter:
            return f"{chapter} > {section}"

        return section or chapter or "Giriş"

    def flush_buffer() -> None:
        """Aktif bölümün yeterli içerik taşıyan chunk'larını üretir."""
        nonlocal buffer, chunk_number

        meaningful = [
            record for record in buffer
            if len(normalize_text(record["text"])) > 0
        ]

        raw_content_size = sum(len(record["text"]) for record in meaningful)

        # Başlıksız kapak, yalnız başlık veya çok kısa parçaları at.
        if raw_content_size < MIN_CONTENT_CHARS:
            buffer = []
            return

        active_path = section_path()

        for group in split_section_into_chunks(active_path, meaningful):
            content = build_content(active_path, group)

            if len(content) < MIN_CONTENT_CHARS:
                continue

            chunks.append(
                {
                    "chunk_id": f"{document_id}_ch_{chunk_number:03d}",
                    "source": source_file,
                    "document_id": document_id,
                    "section": section,
                    "chapter": chapter or None,
                    "section_path": active_path,
                    "content": content,
                    "topic": choose_topic(section, content),
                    "triage_level": choose_triage_level(content),
                    "paragraph_index_start": group[0]["paragraph_index"],
                    "paragraph_index_end": group[-1]["paragraph_index"],
                }
            )
            chunk_number += 1

        buffer = []

    for record in records:
        text = record["text"]

        if should_skip(text):
            continue

        # Kaynakça RAG için indekslenmez.
        if is_references_heading(text):
            flush_buffer()
            break

        heading_type = get_heading_type(text)

        if heading_type == "chapter":
            flush_buffer()
            chapter = text
            section = text
            continue

        if heading_type == "section":
            flush_buffer()
            section = text
            continue

        # SSS: her S: yeni grup başlatır; sonraki C: aynı buffer'da kalır.
        if text.startswith("S:"):
            flush_buffer()
            buffer.append(record)
            continue

        buffer.append(record)

    flush_buffer()
    return chunks


def validate_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    """Üretilen chunk'larda kritik yapısal hataları raporlar."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        content_length = len(chunk["content"])

        if chunk_id in seen_ids:
            errors.append(f"Tekrarlanan chunk_id: {chunk_id}")
        seen_ids.add(chunk_id)

        if not chunk["content"].startswith("### "):
            errors.append(f"Başlık eksik: {chunk_id}")

        if not chunk["content"].strip():
            errors.append(f"Boş content: {chunk_id}")

        if content_length > MAX_CHARS + 100:
            errors.append(
                f"Beklenenden uzun chunk ({content_length} karakter): {chunk_id}"
            )

        if chunk["paragraph_index_start"] > chunk["paragraph_index_end"]:
            errors.append(f"Geçersiz paragraf aralığı: {chunk_id}")

        if chunk["triage_level"] not in {"GREEN", "YELLOW", "EMERGENCY"}:
            errors.append(f"Geçersiz triage değeri: {chunk_id}")

    return errors


def print_quality_report(
    input_records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """Chunk sonuçları için terminalde özet kalite raporu gösterir."""
    errors = validate_chunks(chunks)
    lengths = [len(chunk["content"]) for chunk in chunks]
    topic_counts = Counter(chunk["topic"] for chunk in chunks)
    triage_counts = Counter(chunk["triage_level"] for chunk in chunks)

    print("\n--- KALİTE RAPORU ---")
    print(f"Ham paragraf sayısı : {len(input_records)}")
    print(f"Chunk sayısı        : {len(chunks)}")

    if lengths:
        print(f"En kısa chunk       : {min(lengths)} karakter")
        print(f"Ortalama chunk      : {sum(lengths) // len(lengths)} karakter")
        print(f"En uzun chunk       : {max(lengths)} karakter")

    print(f"Topic dağılımı      : {dict(topic_counts)}")
    print(f"Triyaj dağılımı     : {dict(triage_counts)}")

    if errors:
        print("\nHATALAR:")
        for error in errors:
            print(f"- {error}")
    else:
        print("\nYapısal doğrulama: BAŞARILI")


def save_jsonl(chunks: list[dict[str, Any]], output_path: Path) -> None:
    """Chunk kayıtlarını JSONL olarak diske kaydeder."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    """Komut satırından ham JSONL'yi chunk JSONL'ye dönüştürür."""
    parser = argparse.ArgumentParser(
        description="Ham JSONL dosyasını başlık bağlamlı RAG chunk'larına dönüştürür."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Ham paragraf JSONL dosyasının yolu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Çıktı JSONL yolu. Verilmezse .chunks.jsonl oluşturulur.",
    )

    args = parser.parse_args()
    input_path: Path = args.input

    if not input_path.exists():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {input_path}")

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}.chunks.jsonl"
    )

    records = load_jsonl(input_path)
    chunks = make_chunks(records)
    save_jsonl(chunks, output_path)
    print_quality_report(records, chunks)
    print(f"\nÇıktı kaydedildi: {output_path}")


if __name__ == "__main__":
    main()