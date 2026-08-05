from __future__ import annotations

import re

# Word / kopyala-yapıştır kaynaklı bilinen bozukluklar.
KNOWN_TYPOS = {
    "DAVLARINIŞLARI": "DAVRANIŞLARI",
    "Davlarinişlari": "Davranışları",
    "davlarinişlari": "davranışları",
    "DAVLARINISLARI": "DAVRANIŞLARI",
}

# Rumence/şapka s-cedilla → Türkçe ş
CHAR_FIXES = str.maketrans(
    {
        "\u0219": "ş",  # ș
        "\u0218": "Ş",  # Ș
        "\u021b": "t",  # ț (Rumence) → t
        "\u021a": "T",
    }
)

# Görsel gürültü (emoji, dingbat); metin anlamını taşımaz.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"  # Misc symbols & pictographs, emoticons, etc.
    "\U00002700-\U000027BF"  # Dingbats
    "\U0001FA00-\U0001FAFF"  # Extended-A
    "\U00002600-\U000026FF"  # Misc symbols
    "]+",
    flags=re.UNICODE,
)

# Satır başında kapanış tırnağı var, açılış yok: Prediyabet” → “Prediyabet”
ORPHAN_CLOSE_QUOTE = re.compile(
    r'(^|(?<=\n))([\wÇĞİÖŞÜçğıöşü][\wÇĞİÖŞÜçğıöşü\-]*)([”"])'
)


def clean_source_text(text: str) -> str:
    """
    DOCX kaynaklı yazım, karakter ve gürültü artefaktlarını temizler.

    Tıbbi içeriği değiştirmez; yalnızca bilinen bozuklukları düzeltir.
    """
    if not text:
        return ""

    text = text.translate(CHAR_FIXES)

    for wrong, right in KNOWN_TYPOS.items():
        text = text.replace(wrong, right)

    text = EMOJI_PATTERN.sub("", text)

    def _balance_quote(match: re.Match[str]) -> str:
        prefix, word, quote = match.group(1), match.group(2), match.group(3)
        opener = "“" if quote == "”" else '"'
        return f"{prefix}{opener}{word}{quote}"

    text = ORPHAN_CLOSE_QUOTE.sub(_balance_quote, text)

    # Emoji silindikten sonra çift boşlukları toparla (satır içi).
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)

    return text.strip()
