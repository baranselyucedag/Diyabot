"""Adım 3 — LLM grey-zone sınıflandırma (Nemotron).

context = triage bağlamı (soft flags, skor, numeric YELLOW) — RAG chunk değil.
Timeout / hata → DEFAULT YELLOW + temkinli reason (tempered).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

GreyLevel = Literal["GREEN", "YELLOW", "REFUSE", "EMERGENCY"]

TEMPERED_YELLOW_REASON = (
    "Belirsiz bir durum tespit edildi; otomatik sınıflandırma tamamlanamadı. "
    "Semptomlarınız kötüleşirse veya emin değilseniz 112'yi aramaktan çekinmeyin. "
    "Yakın zamanda hekim değerlendirmenizi öneririm."
)

_CLASSIFY_SYSTEM = """Sen Tip-2 diyabet hasta asistanının aciliyet sınıflandırıcısısın.
Sadece şu etiketlerden birini seç: GREEN, YELLOW, EMERGENCY, REFUSE.
- EMERGENCY: bilinç kaybı, bayılma, şiddetli nefes/göğüs, acil 112 ihtiyacı
- REFUSE: doz/ünite/ilaç artır-azalt veya tanı koyma talebi
- YELLOW: yakın hekim takibi gereken belirsiz/orta risk
- GREEN: genel eğitim sorusu, acil risk yok

Yanıtın SADECE geçerli JSON olsun, başka metin yok:
{"level":"YELLOW","reason":"kısa Türkçe gerekçe"}
"""


@dataclass
class GreyZoneResult:
    """Grey-zone çıktısı."""

    level: GreyLevel
    reason: str
    timed_out: bool = False
    raw: str = ""


def _skip_llm() -> bool:
    return (os.getenv("TRIAGE_SKIP_LLM") or "").strip() in {
        "1",
        "true",
        "True",
        "yes",
    }


def _use_llm_grey() -> bool:
    """LLM'e zorla geçiş (A/B testi) için TRIAGE_USE_LLM_GREY=1.

    Varsayılan: yerel LightGBM model kullanılır.
    """
    return (os.getenv("TRIAGE_USE_LLM_GREY") or "").strip() in {
        "1",
        "true",
        "True",
        "yes",
    }


def _parse_json_level(text: str) -> GreyZoneResult | None:
    """LLM metninden JSON level/reason çıkar."""
    raw = (text or "").strip()
    if not raw:
        return None
    # Kod çiti varsa içini al
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    blob = fence.group(1) if fence else raw
    # İlk { ... } bloğu
    m = re.search(r"\{[^{}]*\}", blob, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    level = str(data.get("level") or "").strip().upper()
    reason = str(data.get("reason") or "").strip()
    if level not in {"GREEN", "YELLOW", "REFUSE", "EMERGENCY"}:
        return None
    return GreyZoneResult(level=level, reason=reason or "grey-zone", raw=raw)  # type: ignore[arg-type]


def classify_grey_zone(
    message: str,
    *,
    soft_flags: list[str],
    fusion_score: float,
    numeric_yellow: bool,
    timeout_s: float = 10.0,
) -> GreyZoneResult:
    """Grey-band sınıflandırma: varsayılan yerel model, TRIAGE_USE_LLM_GREY=1 ise Nemotron.

    Timeout / hata → DEFAULT YELLOW + temkinli reason (tempered).
    """
    # Yerel LightGBM model (varsayılan). LLM'e dönmek için TRIAGE_USE_LLM_GREY=1.
    if not _use_llm_grey():
        from src.api.triage.grey_model import classify_grey_zone_model

        return classify_grey_zone_model(message)

    if _skip_llm():
        return GreyZoneResult(
            level="YELLOW",
            reason=TEMPERED_YELLOW_REASON,
            timed_out=True,
            raw="TRIAGE_SKIP_LLM",
        )

    from src.api.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, get_api_key

    user = (
        f"MESAJ:\n{(message or '').strip()}\n\n"
        f"TRIAGE_BAGLAM:\n"
        f"- fusion_score: {fusion_score:.3f}\n"
        f"- numeric_yellow: {numeric_yellow}\n"
        f"- soft_flags: {soft_flags}\n"
    )

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=DEFAULT_BASE_URL,
            api_key=get_api_key(),
            timeout=timeout_s,
        )
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=256,
            top_p=0.9,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                }
            },
        )
        if not resp.choices:
            raise RuntimeError("LLM boş choices")
        content = resp.choices[0].message.content or ""
        parsed = _parse_json_level(content)
        if parsed is None:
            logger.warning("grey_zone_parse_fail raw=%r", content[:400])
            return GreyZoneResult(
                level="YELLOW",
                reason=TEMPERED_YELLOW_REASON,
                timed_out=False,
                raw=content,
            )
        return parsed
    except Exception as exc:  # noqa: BLE001 — timeout / network / key
        logger.warning("grey_zone_timeout_or_error: %s", exc)
        return GreyZoneResult(
            level="YELLOW",
            reason=TEMPERED_YELLOW_REASON,
            timed_out=True,
            raw=str(exc),
        )
