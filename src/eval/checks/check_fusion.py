#!/usr/bin/env python
"""Adım 3 fusion smoke — soft katkı, band, guard.

Calistir: python -m src.eval.checks.check_fusion
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRIAGE_SKIP_IMPLICIT", "1")
os.environ.setdefault("TRIAGE_SKIP_LLM", "1")

from src.api.triage.fusion import (
    DEFAULT_FUSION,
    evaluate_fusion,
    fusion_score,
    soft_signal,
)
from src.api.triage import detect_triage_detailed


def main() -> None:
    """Fusion senaryolari."""
    failed = 0

    # Soft katkı numeric'ten bagimsiz (kilit: cok_yuksek=0.50)
    s_only = soft_signal(["cok_yuksek"])
    assert abs(s_only - 0.50) < 1e-6, s_only
    score_soft = fusion_score(
        numeric_yellow=False, soft_flags=["cok_yuksek"], implicit_score=0.0
    )
    # 0.30 * 0.50 / 1.0 = 0.15 → below
    ok = score_soft < DEFAULT_FUSION.band_low
    print(f"[{'OK' if ok else 'FAIL'}] soft-only skor={score_soft:.3f} below band")
    if not ok:
        failed += 1

    # Numeric YELLOW + guclu soft (hizli_degisim=0.85) → above
    score_both = fusion_score(
        numeric_yellow=True, soft_flags=["hizli_degisim"], implicit_score=0.0
    )
    # (0.4*1 + 0.3*0.85) / 1 = 0.655 → above
    ok = score_both > DEFAULT_FUSION.band_high
    print(f"[{'OK' if ok else 'FAIL'}] num+soft skor={score_both:.3f} above band")
    if not ok:
        failed += 1

    # Numeric alone → grey
    score_num = fusion_score(
        numeric_yellow=True, soft_flags=[], implicit_score=0.0
    )
    ok = DEFAULT_FUSION.band_low <= score_num <= DEFAULT_FUSION.band_high
    print(f"[{'OK' if ok else 'FAIL'}] num-only skor={score_num:.3f} grey band")
    if not ok:
        failed += 1

    # Soft flags numeric YELLOW olsa da katkı
    fus = evaluate_fusion(
        "şekerim 60 ve çok yüksek",
        numeric_yellow=True,
        soft_flags=["cok_yuksek"],
        implicit_score=0.0,
    )
    ok = fus.soft_signal > 0 and fus.numeric_yellow
    print(f"[{'OK' if ok else 'FAIL'}] soft+num together soft_signal={fus.soft_signal}")
    if not ok:
        failed += 1

    # Guard: hard emergency message
    fus_g = evaluate_fusion(
        "bilincim bulanık",
        numeric_yellow=False,
        soft_flags=[],
        implicit_score=0.0,
    )
    ok = fus_g.guarded_level == "EMERGENCY"
    print(f"[{'OK' if ok else 'FAIL'}] monotonicity guard → {fus_g.guarded_level}")
    if not ok:
        failed += 1

    # End-to-end detailed
    d = detect_triage_detailed("şekerim 60")
    ok = d.level == "YELLOW" and d.tempered  # grey → skip LLM tempered
    print(f"[{'OK' if ok else 'FAIL'}] seker 60 → {d.level} tempered={d.tempered} src={d.source}")
    if not ok:
        failed += 1

    d2 = detect_triage_detailed("üç gündür kötüye gidiyor")
    ok = d2.level == "GREEN"
    print(f"[{'OK' if ok else 'FAIL'}] soft-only → {d2.level} (beklenen GREEN)")
    if not ok:
        failed += 1

    d3 = detect_triage_detailed("şekerim 45")
    ok = d3.level == "EMERGENCY" and d3.source == "hard_numeric"
    print(f"[{'OK' if ok else 'FAIL'}] hard numeric → {d3.level} src={d3.source}")
    if not ok:
        failed += 1

    print()
    if failed:
        print(f"BASARISIZ: {failed}")
        raise SystemExit(1)
    print("Tum fusion senaryolari gecti.")


if __name__ == "__main__":
    main()
