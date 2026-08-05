"""Adım 1 — Sayısal glukoz triage motoru.

Yalnızca: sayı çıkarma + glukoz bağlamı + semptom bayrakları + eşikler.

Eşik kaynağı: TEMD/ADA/Endokrin-Aciller
Ramazan: bilinçli over-triage (under-triage riskini azaltmak için).
Birim: varsayılan mg/dL; mmol geçiyorsa convert yok → fail-safe YELLOW.

Sınırlar bilinçli olarak >= kullanır (250, 600 yuvarlak glukometre okumaları).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Literal

from src.api.triage.text_utils import norm

NumericLevel = Literal["EMERGENCY", "YELLOW"]

# Geriye uyumluluk (eski _norm importları)
_norm = norm

# Sayı civarında glukoz bağlamı arama penceresi (karakter)
_CONTEXT_WINDOW = 48

# Yanlış pozitif: para / kalori / sıcaklık
_FALSE_CONTEXT = re.compile(
    r"\b(lira|tl|try|kalori|kcal|derece|celsius|cm|kg|adet)\b",
)

# Glukoz bağlamı (normalize metinde). mmol dahil: pratikte kan şekeri birimi.
_GLUCOSE_CONTEXT = re.compile(
    r"(seker|glukoz|glikoz|kan\s*seker|mg\s*/?\s*dl|mmol|"
    r"hipoglisemi|hiperglisemi|aclik\s*seker|tokluk\s*seker|"
    r"parmak\s*ucu|olcum)",
)

# HbA1c yakını sayıyı glukoz sanma
_HBA1C_NEAR = re.compile(r"hba1c|a1c|glikozile")

_RAMADAN = re.compile(r"(ramazan|oruc|oruclu|iftar|sahur|orucumu|orucunu)")

_MMOL = re.compile(r"\bmmol\b")

# Sayı: 45, 70, 250, 3.5, 5,2
# Not: hem '.' hem ',' ondalık gibi işlenir (TR binlik/ondalık karışıklığı;
# glukoz için >1000 zaten elenir, pratik risk düşük).
_NUMBER = re.compile(r"(?<![A-Za-z0-9])(\d{1,4}(?:[.,]\d+)?)(?![A-Za-z0-9])")

# Semptomlar — metin _norm sonrası ASCII; negatif lookahead ile yanlış pozitif azaltılır.
# (?![a-z0-9]) : ekleme/çekim dışı kelime devamını keser (kusursuz, kusur…).
_DKA_SYMPTOMS = [
    r"(?<![a-z])kus(uyorum|uyoruz|uyor|muk|ma|unca|uk|tu|ar)(?![a-z0-9])",
    r"(?<![a-z])bulant",
    r"mide\s*bulan",
    r"(?<![a-z])karin\s*agr",
    r"nefes\s*(alam|dar)",
    r"asiri\s*susa",
    r"cok\s*susa",
    r"(?<![a-z])susuzluk",
    r"sik\s*idrar",
    r"idrar.*sik",
    r"(?<![a-z])poliuri",
    r"(?<![a-z])polidipsi",
    r"meyve.*nefes|(?<![a-z])aseton|nefes.*koku",
    r"(?<![a-z])halsiz",
    r"bulanik\s*gorm",
    r"(?<![a-z])bas\s*don(me|uyor|du|uk)?(?![a-z0-9])",
]

_CONSCIOUSNESS = [
    r"(?<![a-z])bilinc",
    r"(?<![a-z])baygin",
    r"(?<![a-z])bayil",
    r"cevap\s*vermiyor",
    r"konus(masi)?\s*bozul",
    r"kendinde\s*degil",
]

_DEHYDRATION = [
    r"asiri\s*susa",
    r"cok\s*susa",
    r"cok\s*su\s*ic",
    r"agzim?.*kur|(?<![a-z])agiz.*kur|(?<![a-z])kurudu",
    r"sik\s*idrar",
    r"(?<![a-z])dehidrat",
    r"sivi\s*kayb",
]


@dataclass
class GlucoseCandidate:
    """Metinden çıkarılmış tek glukoz adayı (indeksler normalize metinde)."""

    value: float
    start: int
    end: int
    has_context: bool


@dataclass
class NumericTriageResult:
    """Sayısal motor çıktısı. level=None → bu katman karar vermedi."""

    level: NumericLevel | None
    reason: str
    glucose_mgdl: float | None
    flags: list[str] = field(default_factory=list)
    ramadan: bool = False
    unit_unknown: bool = False


def has_glucose_context(norm_text: str, start: int, end: int) -> bool:
    """Sayının penceresinde glukoz bağlamı var mı; HbA1c-only ve para/kalori elenir.

    '300 lira' / '600 kalori' tetiklenmesin diye false-context kontrolü yapılır.
    """
    lo = max(0, start - _CONTEXT_WINDOW)
    hi = min(len(norm_text), end + _CONTEXT_WINDOW)
    window = norm_text[lo:hi]
    if _FALSE_CONTEXT.search(window):
        return False
    if _HBA1C_NEAR.search(window) and not _GLUCOSE_CONTEXT.search(window):
        return False
    return bool(_GLUCOSE_CONTEXT.search(window))


def has_ramadan_context(norm_text: str) -> bool:
    """Ramazan / oruç bağlamı var mı (tüm mesaj)."""
    return bool(_RAMADAN.search(norm_text))


def _any_flag(norm_text: str, patterns: list[str]) -> bool:
    return any(re.search(p, norm_text) for p in patterns)


def flag_dka_symptoms(norm_text: str) -> bool:
    """DKA / erken atipik belirtiler (kusma, susama, halsizlik, aseton nefesi…)."""
    return _any_flag(norm_text, _DKA_SYMPTOMS)


def flag_consciousness(norm_text: str) -> bool:
    """Bilinç bulanıklığı / bayılma / cevap vermeme."""
    return _any_flag(norm_text, _CONSCIOUSNESS)


def flag_dehydration_group(norm_text: str) -> bool:
    """Polidipsi / poliüri / ağız kuruluğu grubu (hasta dili)."""
    return _any_flag(norm_text, _DEHYDRATION)


def extract_glucose_candidates(message: str) -> list[GlucoseCandidate]:
    """Mesajdaki sayıları normalize metinde tarar; glukoz bağlamı olanları döner.

    Sayı araması norm çıktısı üzerinde yapılır → İ/casefold kaynaklı
    indeks kayması olmaz. Varsayılan birim mg/dL.
    """
    n = norm(message or "")
    out: list[GlucoseCandidate] = []
    for m in _NUMBER.finditer(n):
        token = m.group(1).replace(",", ".")
        try:
            value = float(token)
        except ValueError:
            continue
        start, end = m.start(), m.end()
        if not has_glucose_context(n, start, end):
            continue
        # Aşırı değer: glukoz bağlamı var ama >1000 → aday; karar ağacı YELLOW fail-safe
        out.append(
            GlucoseCandidate(
                value=value,
                start=start,
                end=end,
                has_context=True,
            )
        )
    return out


def _severity_rank(level: NumericLevel | None) -> int:
    if level == "EMERGENCY":
        return 2
    if level == "YELLOW":
        return 1
    return 0


def _decide_for_value(
    value: float,
    *,
    ramadan: bool,
    dka: bool,
    consciousness: bool,
    dehydration: bool,
) -> tuple[NumericLevel | None, str, list[str]]:
    """Tek glukoz değeri için eşik karar ağacı (hard → yellow → none).

    Sınırlar >= : yuvarlak glukometre okumaları (250, 600) sessizce kaçmasın.
    """
    flags: list[str] = []
    if ramadan:
        flags.append("ramadan")
    if dka:
        flags.append("dka_symptoms")
    if consciousness:
        flags.append("consciousness")
    if dehydration:
        flags.append("dehydration")

    # Aşırı / anlamsız mg/dL okuması
    if value > 1000:
        warnings.warn(
            f"numeric_triage: asiri glukoz adayi value={value} (insan gozden gecirmeli)",
            stacklevel=2,
        )
        return (
            "YELLOW",
            f"glukoz={value}>1000 asiri deger (fail-safe YELLOW)",
            flags + ["extreme_value"],
        )

    # HARD: ciddi hipoglisemi (ADA Level 2)
    if value < 54:
        return "EMERGENCY", f"glukoz={value}<54 (Level 2 hipoglisemi)", flags

    # HARD: Ramazan + hipo <70 (bilinçli over-triage)
    if ramadan and value < 70:
        return (
            "EMERGENCY",
            f"glukoz={value}<70 + Ramazan (oruç boz + acil yönlendirme)",
            flags,
        )

    # Boşluk: 54–69 → YELLOW (klinik <70 uyarı)
    if 54 <= value < 70:
        return "YELLOW", f"glukoz={value} in [54,70) hipoglisemi Seviye 1 bandı", flags

    # HARD: >=250 + DKA belirtileri
    if value >= 250 and dka:
        return "EMERGENCY", f"glukoz={value}>=250 + DKA belirtisi", flags

    # HARD: >=250 + bilinç (orta-yüksek + bilinç; Adım1 içinde garanti)
    if value >= 250 and consciousness:
        return "EMERGENCY", f"glukoz={value}>=250 + bilinc degisikligi", flags

    # HARD: Ramazan + >=300
    if ramadan and value >= 300:
        return (
            "EMERGENCY",
            f"glukoz={value}>=300 + Ramazan (oruç boz + acil yönlendirme)",
            flags,
        )

    # HARD: >=600 + bilinç veya dehidratasyon grubu
    if value >= 600 and (consciousness or dehydration):
        return (
            "EMERGENCY",
            f"glukoz={value}>=600 + bilinc/dehidratasyon (HHS suphesi)",
            flags,
        )

    # Boşluk: >=600 semptomsuz → YELLOW
    if value >= 600:
        return "YELLOW", f"glukoz={value}>=600 semptom bayragi yok", flags

    # Boşluk: >=250 DKA/bilinç yok → YELLOW
    if value >= 250:
        return "YELLOW", f"glukoz={value}>=250 DKA/bilinc yok", flags

    # 70–249.99: bu katman karar vermez
    return None, f"glukoz={value} aralik disi [70,250)", flags


def evaluate_numeric_triage(message: str) -> NumericTriageResult | None:
    """Sayısal glukoz triage. Aday yoksa None.

    Hard eşleşme → EMERGENCY; boşluk aralıkları → YELLOW.
    mmol birimi → convert yok, fail-safe YELLOW (hard yok).
    Birden fazla adayda en yüksek severity kazanır.
    """
    text = message or ""
    if not text.strip():
        return None

    n = norm(text)
    ramadan = has_ramadan_context(n)
    dka = flag_dka_symptoms(n)
    consciousness = flag_consciousness(n)
    dehydration = flag_dehydration_group(n)

    # mmol: birim belirsiz → fail-safe YELLOW
    # mmol artık glukoz bağlamında; yalnız "5.2 mmol" da yakalanır.
    if _MMOL.search(n):
        return NumericTriageResult(
            level="YELLOW",
            reason="mmol birimi tespit edildi; mg/dL varsayimi uygulanmadi (fail-safe)",
            glucose_mgdl=None,
            flags=["unit_mmol"],
            ramadan=ramadan,
            unit_unknown=True,
        )

    candidates = extract_glucose_candidates(text)
    if not candidates:
        return None

    best: NumericTriageResult | None = None
    for cand in candidates:
        level, reason, flags = _decide_for_value(
            cand.value,
            ramadan=ramadan,
            dka=dka,
            consciousness=consciousness,
            dehydration=dehydration,
        )
        result = NumericTriageResult(
            level=level,
            reason=reason,
            glucose_mgdl=cand.value,
            flags=flags,
            ramadan=ramadan,
            unit_unknown=False,
        )
        if best is None or _severity_rank(result.level) > _severity_rank(best.level):
            best = result
        elif (
            best is not None
            and _severity_rank(result.level) == _severity_rank(best.level)
            and result.level is not None
        ):
            # Aynı seviyede "normal ortama" (≈100 mg/dL acık) daha uzak değeri seç —
            # daha ekstrem glukoz okuması klinik olarak daha öncelikli kabul edilir.
            if best.glucose_mgdl is None or (
                result.glucose_mgdl is not None
                and abs(result.glucose_mgdl - 100.0)
                > abs((best.glucose_mgdl or 100.0) - 100.0)
            ):
                best = result

    if best is None or best.level is None:
        return None
    return best
