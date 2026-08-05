"""Adım 3 — Weighted fusion + grey-band + monotonicity guard.

Hard veto (EMERGENCY/REFUSE) fusion'ı bypass eder (detect_triage erken return).
Bu modül soft sinyaller içindir. Guard = defense-in-depth (ikinci tarama).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.api.triage.regex_flags import SOFT_YELLOW_WEIGHTS, flag_emergency, flag_jailbreak, flag_refuse
from src.api.triage.text_utils import norm


@dataclass
class FusionConfig:
    """Elle seed ağırlıklar + band; tune_band.py ile ayarlanır."""

    w_numeric_yellow: float = 0.40
    w_regex_soft: float = 0.30
    w_ft_encoder: float = 0.30
    band_low: float = 0.30
    band_high: float = 0.60
    soft_weights: dict[str, float] = field(
        default_factory=lambda: dict(SOFT_YELLOW_WEIGHTS)
    )


DEFAULT_FUSION = FusionConfig()


@dataclass
class FusionResult:
    """Fusion çıktısı."""

    score: float
    numeric_yellow: bool
    soft_flags: list[str]
    soft_signal: float
    ft_score: float
    band: str  # "below" | "grey" | "above"
    guarded_level: str | None = None  # EMERGENCY/REFUSE if guard fired
    reason: str = ""


def soft_signal(flags: list[str], weights: dict[str, float] | None = None) -> float:
    """Soft flag ağırlık toplamı, [0, 1] clamp."""
    w = weights if weights is not None else SOFT_YELLOW_WEIGHTS
    total = sum(w.get(f, 0.0) for f in flags)
    return float(min(1.0, max(0.0, total)))


def fusion_score(
    *,
    numeric_yellow: bool,
    soft_flags: list[str],
    ft_score: float,
    config: FusionConfig | None = None,
) -> float:
    """Ağırlıklı skor [0, 1]. Soft flags numeric'ten bağımsız katkı verir."""
    cfg = config or DEFAULT_FUSION
    s_num = 1.0 if numeric_yellow else 0.0
    s_soft = soft_signal(soft_flags, cfg.soft_weights)
    s_ft = float(min(1.0, max(0.0, ft_score)))
    w_sum = cfg.w_numeric_yellow + cfg.w_regex_soft + cfg.w_ft_encoder
    if w_sum <= 0:
        return 0.0
    raw = (
        cfg.w_numeric_yellow * s_num
        + cfg.w_regex_soft * s_soft
        + cfg.w_ft_encoder * s_ft
    )
    return float(raw / w_sum)


def band_region(score: float, config: FusionConfig | None = None) -> str:
    """below | grey | above."""
    cfg = config or DEFAULT_FUSION
    if score > cfg.band_high:
        return "above"
    if score < cfg.band_low:
        return "below"
    return "grey"


def monotonicity_guard(message: str) -> str | None:
    """Defense-in-depth: hard veto kaçtıysa fusion sonrası tekrar yakala.

    Returns 'EMERGENCY' | 'REFUSE' | None.
    """
    n = norm(message or "")
    if flag_emergency(n):
        return "EMERGENCY"
    if flag_refuse(n) or flag_jailbreak(n):
        return "REFUSE"
    return None


def evaluate_fusion(
    message: str,
    *,
    numeric_yellow: bool,
    soft_flags: list[str],
    ft_score: float,
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fusion skor + band + guard."""
    cfg = config or DEFAULT_FUSION
    guarded = monotonicity_guard(message)
    s_soft = soft_signal(soft_flags, cfg.soft_weights)
    score = fusion_score(
        numeric_yellow=numeric_yellow,
        soft_flags=soft_flags,
        ft_score=ft_score,
        config=cfg,
    )
    region = band_region(score, cfg)
    reason_parts = [
        f"score={score:.3f}",
        f"band={region}",
        f"num_y={int(numeric_yellow)}",
        f"soft={s_soft:.2f}{soft_flags}",
        f"ft={ft_score:.3f}",
    ]
    if guarded:
        reason_parts.append(f"guard={guarded}")
    return FusionResult(
        score=score,
        numeric_yellow=numeric_yellow,
        soft_flags=list(soft_flags),
        soft_signal=s_soft,
        ft_score=ft_score,
        band=region,
        guarded_level=guarded,
        reason="; ".join(reason_parts),
    )
