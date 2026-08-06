"""Triage paketi — numeric + regex + fusion + grey-zone.

Sınıflar: GREEN | YELLOW | REFUSE | EMERGENCY
Hard veto fusion'ı bypass eder; soft flags her zaman fusion'a girer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.api.triage.implicit_score import score_implicit
from src.api.triage.fusion import evaluate_fusion
from src.api.triage.grey_zone import TEMPERED_YELLOW_REASON, classify_grey_zone
from src.api.triage.numeric import evaluate_numeric_triage
from src.api.triage.regex_flags import (
    JAILBREAK_FLAG_LABELS,
    SOFT_YELLOW_WEIGHTS,
    evaluate_regex_flags,
)
from src.api.triage.text_utils import norm

_norm = norm

TriageLevel = Literal["GREEN", "YELLOW", "REFUSE", "EMERGENCY"]

_JAILBREAK_CANNED = (
    "Bu tür rol yapma veya sistem kurallarını atlatma isteklerine yanıt veremem.\n\n"
    "Güvenlik kurallarımı yok sayamam ve doktor gibi reçete/doz yazamam. "
    "Tip 2 diyabet eğitimi sorularınızı yanıtlayabilirim; "
    "kişisel tedavi kararları için hekiminize danışın."
)

_REFUSE_CANNED = (
    "İlaç dozu artırma/azaltma veya kişiye özel tedavi değişikliği konusunda öneri veremem.\n\n"
    "Doz ve tedavi kararları yalnızca hekiminiz tarafından verilmelidir. "
    "Görüşmenizde mevcut dozunuzun uygunluğu, yan etki riski ve kontrol randevunuzu sorunuz."
)

_EMERGENCY_CANNED = (
    "Belirttiğiniz belirtiler acil durum işaretleri olabilir.\n\n"
    "Lütfen hemen 112 Acil Çağrı Merkezi’ni arayın veya en yakın acil servise başvurun. "
    "Bu asistan acil tıbbi müdahale sağlayamaz."
)


@dataclass
class TriageDecision:
    """Detaylı triage sonucu (pipeline / log için)."""

    level: TriageLevel
    reason: str
    score: float | None = None
    flags: list[str] = field(default_factory=list)
    soft_flags: list[str] = field(default_factory=list)
    source: str = ""  # hard_numeric | hard_regex | fusion | grey_zone | guard | default
    tempered: bool = False


def detect_triage_detailed(message: str) -> TriageDecision:
    """Tam triage: hard veto → soft+numeric+örtük skor fusion → grey band LLM.

    Soft regex flags numeric sonucundan bağımsız her zaman hesaplanır.
    """
    text = message or ""
    numeric = evaluate_numeric_triage(text)
    regex = evaluate_regex_flags(text)

    # Soft flags: regex all_flags içinden soft etiketler (numeric'ten bağımsız)
    soft_flags: list[str] = []
    if regex is not None:
        soft_flags = [f for f in regex.all_flags if f in SOFT_YELLOW_WEIGHTS]
        if regex.level is None and regex.flags:
            soft_flags = [f for f in regex.flags if f in SOFT_YELLOW_WEIGHTS]

    hard_flags: list[str] = []
    if regex is not None and regex.level in {"EMERGENCY", "REFUSE"}:
        hard_flags = list(regex.flags)

    # (1)(2) Hard veto
    if numeric is not None and numeric.level == "EMERGENCY":
        return TriageDecision(
            level="EMERGENCY",
            reason=numeric.reason,
            flags=list(numeric.flags),
            soft_flags=soft_flags,
            source="hard_numeric",
        )
    if regex is not None and regex.level == "EMERGENCY":
        return TriageDecision(
            level="EMERGENCY",
            reason=regex.reason,
            flags=hard_flags,
            soft_flags=soft_flags,
            source="hard_regex",
        )
    if regex is not None and regex.level == "REFUSE":
        return TriageDecision(
            level="REFUSE",
            reason=regex.reason,
            flags=hard_flags,
            soft_flags=soft_flags,
            source="hard_regex",
        )

    num_y = numeric is not None and numeric.level == "YELLOW"
    imp = score_implicit(text)
    fus = evaluate_fusion(
        text,
        numeric_yellow=bool(num_y),
        soft_flags=soft_flags,
        implicit_score=imp,
    )

    # Defense-in-depth guard
    if fus.guarded_level == "EMERGENCY":
        return TriageDecision(
            level="EMERGENCY",
            reason=f"monotonicity_guard: {fus.reason}",
            score=fus.score,
            flags=hard_flags,
            soft_flags=soft_flags,
            source="guard",
        )
    if fus.guarded_level == "REFUSE":
        return TriageDecision(
            level="REFUSE",
            reason=f"monotonicity_guard: {fus.reason}",
            score=fus.score,
            flags=hard_flags,
            soft_flags=soft_flags,
            source="guard",
        )

    if fus.band == "above":
        return TriageDecision(
            level="YELLOW",
            reason=f"fusion above band: {fus.reason}",
            score=fus.score,
            soft_flags=soft_flags,
            source="fusion",
        )
    if fus.band == "below":
        return TriageDecision(
            level="GREEN",
            reason=f"fusion below band: {fus.reason}",
            score=fus.score,
            soft_flags=soft_flags,
            source="fusion",
        )

    # Grey band → LLM
    grey = classify_grey_zone(
        text,
        soft_flags=soft_flags,
        fusion_score=fus.score,
        numeric_yellow=bool(num_y),
    )
    return TriageDecision(
        level=grey.level,
        reason=grey.reason,
        score=fus.score,
        soft_flags=soft_flags,
        source="grey_zone",
        tempered=bool(grey.timed_out),
    )


def detect_triage(message: str) -> TriageLevel:
    """Mesajdan aciliyet seviyesini çıkarır (string API)."""
    return detect_triage_detailed(message).level


def canned_response(
    level: TriageLevel,
    *,
    flags: list[str] | None = None,
    tempered: bool = False,
    reason: str | None = None,
) -> str | None:
    """EMERGENCY / REFUSE canned; tempered YELLOW için güçlendirilmiş uyarı.

    GREEN / normal YELLOW → None (RAG devam).
    """
    if level == "EMERGENCY":
        return _EMERGENCY_CANNED
    if level == "REFUSE":
        flag_set = set(flags or ())
        if flag_set & JAILBREAK_FLAG_LABELS:
            return _JAILBREAK_CANNED
        return _REFUSE_CANNED
    if level == "YELLOW" and tempered:
        return reason or TEMPERED_YELLOW_REASON
    return None
