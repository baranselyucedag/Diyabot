"""Adım 2 — Regex / morfoloji bayrakları.

Sayıdan bağımsız deterministik dil katmanı:
  - EMERGENCY: bilinç/bayılma/göğüs/nefes/112 (112 canned)
  - REFUSE: doz / tanı / reçete talebi (112 yok; ayrı canned)
  - JAILBREAK: prompt injection / rol yapma → seviye REFUSE, ayrı canned
  - YELLOW: yumuşak uyarı kalıpları (RAG devam; soft sinyal)

Glukoz sayısı eşikleri burada YOK (Adım 1 numeric).
FT encoder / LLM / score fusion burada YOK.

Pattern'ler `norm` sonrası ASCII metinde; negatif lookahead ile FP azaltılır.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from src.api.triage.text_utils import norm

RegexLevel = Literal["EMERGENCY", "REFUSE", "YELLOW"]

# Jailbreak flag etiketleri — canned_response güvenlik metni için
JAILBREAK_FLAG_LABELS = frozenset(
    {"sistem_prompt", "prompt_yok_say", "doktor_gibi"}
)

# ---------------------------------------------------------------------------
# Hard EMERGENCY — sayı şart değil (acil dil / bilinç / kardiyopulmoner)
# ---------------------------------------------------------------------------
_EMERGENCY: list[tuple[str, str]] = [
    (r"(?<![0-9])112(?![0-9])", "112"),
    (r"(?<![a-z])bayil", "bayilma"),
    (r"(?<![a-z])baygin", "bayginlik"),
    (r"(?<![a-z])bilinc", "bilinc"),
    (r"cevap\s*vermiyor", "cevap_vermiyor"),
    # konusamiyorum / konusamiyor / konusamaz…
    (r"konus(amiyor|amaz)", "konusamama"),
    (r"konus(masi)?\s*bozul", "konusma_bozulmasi"),
    (r"kendinde\s*degil", "kendinde_degil"),  # #3 boundary bilinçli atlandı
    # gogus / gogsum / gogusum… + agr*
    (r"gog\w*\s*agr", "gogus_agrisi"),
    (r"nefes\s*alamiyorum", "nefes_alamama"),
    (r"nefes\s*darligi", "nefes_darligi"),
]

# ---------------------------------------------------------------------------
# REFUSE — doz / tanı / reçete (üçlü). Eğitim soruları ("doz nedir?") kaçınır.
# ---------------------------------------------------------------------------
_REFUSE_DOSE: list[tuple[str, str]] = [
    (r"kac\s*unite", "kac_unite"),
    (
        r"doz(um|unu|u|unu)?\s*(hesapla|ayarla|artir|azalt|onayla|yaz|soyle)",
        "doz_talep",
    ),
    (r"dozum.*(iki\s*)?kat", "doz_iki_kat"),
    (
        r"(insulin|metformin|ozempic|glukagon|sulfonilure|ilac).{0,48}"
        r"(kac\s*unite|doz|artir|azalt|onayla|hesapla|ayarla|iki\s*kat)",
        "ilac_doz_talep",
    ),
    (
        r"(kac\s*unite|doz|artir|azalt|onayla|hesapla|ayarla|iki\s*kat).{0,48}"
        r"(insulin|metformin|ozempic|glukagon|sulfonilure)",
        "ilac_doz_talep_ters",
    ),
    # "2000 mg'a çık" tipi doz artırma (sayı zorunlu)
    (r"\d+\s*mg'?a\s*cik", "mg_cikarma"),
    (r"recetesiz", "recetesiz"),
    (r"doz\s*semasi", "doz_semasi"),
]

_REFUSE_DIAGNOSIS: list[tuple[str, str]] = [
    (r"tani\s*koy", "tani_koy"),
    (r"tanisi\s*koy", "tanisi_koy"),
    (r"yetmezligi\s*var\s*mi", "yetmezlik_tani"),
    (r"evrem\s*nedir", "evre_tani"),
    # tıbbi bağlam zorunlu; "bende bu telefon var mı" FP olmasın
    (
        r"bende\s+.{0,40}?(yetmezlik|diyabet|hastalik|bobrek|kanser).{0,20}?var\s*mi",
        "bende_var_mi_tani",
    ),
]

_REFUSE_PRESCRIBE: list[tuple[str, str]] = [
    (r"ilac\s*(kombinasyon(u)?)?\s*yaz", "ilac_yaz"),
    (r"kombinasyon(u)?\s*yaz", "kombinasyon_yaz"),
    (r"recete\s*yaz", "recete_yaz"),
]

# Birleşik REFUSE (tek geçiş)
_REFUSE: list[tuple[str, str]] = (
    _REFUSE_DOSE + _REFUSE_DIAGNOSIS + _REFUSE_PRESCRIBE
)

# ---------------------------------------------------------------------------
# JAILBREAK / güvenlik — seviye REFUSE, canned ayrı
# ---------------------------------------------------------------------------
_JAILBREAK: list[tuple[str, str]] = [
    (r"sistem\s*prompt", "sistem_prompt"),
    (r"prompt(un|u)?\s*yok\s*say", "prompt_yok_say"),
    # iltifat ("doktor gibi konuşuyorsun") tetiklemesin
    (r"doktor\s*gibi\s*(davran|rol\s*yap|yaz)", "doktor_gibi"),
]

# ---------------------------------------------------------------------------
# Soft YELLOW — belirsiz / süregelen şikayet (hard değil)
# Ağırlıklar fusion.py'de uygulanır; burada sadece tespit + seed dict.
# ---------------------------------------------------------------------------
_SOFT_YELLOW: list[tuple[str, str]] = [
    (r"(?<![a-z])uc\s*gundur", "uc_gundur"),
    (r"(?<![a-z])cok\s*yuksek", "cok_yuksek"),
    (r"(?<![a-z])cok\s*dusuk", "cok_dusuk"),
    (r"hizla\s*(yukseliyor|dusuyor)", "hizli_degisim"),
]

# Seed ağırlıklar (ciddiyet sıralı). Fusion clamp [0,1] kullanır.
# Precision ölçümüyle ileride güncellenir; EMERGENCY satırları precision'a dahil değil.
SOFT_YELLOW_WEIGHTS: dict[str, float] = {
    "cok_yuksek": 0.70,
    "cok_dusuk": 0.70,
    "hizli_degisim": 0.55,
    "uc_gundur": 0.30,
}


@dataclass
class RegexTriageResult:
    """Regex katmanı çıktısı. level=None → bu katman karar vermedi."""

    level: RegexLevel | None
    reason: str
    flags: list[str] = field(default_factory=list)
    all_flags: list[str] = field(default_factory=list)
    suppressed: list[tuple[str, list[str]]] = field(default_factory=list)


def _match_flags(norm_text: str, rules: list[tuple[str, str]]) -> list[str]:
    """Eşleşen kural etiketlerini döner (sıra korunur, tekil)."""
    hit: list[str] = []
    seen: set[str] = set()
    for pat, label in rules:
        if re.search(pat, norm_text) and label not in seen:
            hit.append(label)
            seen.add(label)
    return hit


def _dedupe_preserve(items: list[str]) -> list[str]:
    """Sıra koruyarak tekilleştir."""
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def flag_emergency(norm_text: str) -> list[str]:
    """Hard EMERGENCY bayrakları."""
    return _match_flags(norm_text, _EMERGENCY)


def flag_refuse(norm_text: str) -> list[str]:
    """REFUSE bayrakları (doz + tanı + reçete) — tek geçiş."""
    return _match_flags(norm_text, _REFUSE)


def flag_jailbreak(norm_text: str) -> list[str]:
    """Jailbreak / prompt-injection bayrakları."""
    return _match_flags(norm_text, _JAILBREAK)


def flag_soft_yellow(norm_text: str) -> list[str]:
    """Yumuşak YELLOW uyarı bayrakları."""
    return _match_flags(norm_text, _SOFT_YELLOW)


def evaluate_regex_flags(message: str) -> RegexTriageResult | None:
    """Regex/morfoloji triage. Eşleşme yoksa None.

    Öncelik (katman içi): EMERGENCY > REFUSE > JAILBREAK.
    Soft YELLOW tek başına level üretmez (Adım 3 fusion karar verir):
    level=None, flags=soft, reason=soft_flags_only_for_fusion.
    JAILBREAK seviye olarak REFUSE döner.
    """
    text = message or ""
    if not text.strip():
        return None

    n = norm(text)
    emerg = flag_emergency(n)
    refuse = flag_refuse(n)
    jail = flag_jailbreak(n)
    soft = flag_soft_yellow(n)

    all_flags = _dedupe_preserve(emerg + refuse + jail + soft)
    if not all_flags:
        return None

    suppressed: list[tuple[str, list[str]]] = []

    if emerg:
        if refuse:
            suppressed.append(("REFUSE", refuse))
        if jail:
            suppressed.append(("JAILBREAK", jail))
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="EMERGENCY",
            reason=f"regex EMERGENCY: {', '.join(emerg)}",
            flags=emerg,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    if refuse:
        if jail:
            suppressed.append(("JAILBREAK", jail))
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="REFUSE",
            reason=f"regex REFUSE: {', '.join(refuse)}",
            flags=refuse,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    if jail:
        if soft:
            suppressed.append(("YELLOW", soft))
        return RegexTriageResult(
            level="REFUSE",
            reason=f"regex JAILBREAK->REFUSE: {', '.join(jail)}",
            flags=jail,
            all_flags=all_flags,
            suppressed=suppressed,
        )

    # Soft only → fusion için flag; seviye None (Adım 3 migration)
    return RegexTriageResult(
        level=None,
        reason="soft_flags_only_for_fusion",
        flags=soft,
        all_flags=all_flags,
        suppressed=suppressed,
    )
