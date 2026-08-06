#!/usr/bin/env python
"""Adım 1 sayısal triage motoru — plan + güvenlik düzeltme senaryoları.

Calistir: python -m src.eval.check_numeric_triage
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRIAGE_SKIP_IMPLICIT", "1")
os.environ.setdefault("TRIAGE_SKIP_LLM", "1")

from src.api.triage import detect_triage
from src.api.triage.numeric import (
    evaluate_numeric_triage,
    flag_dka_symptoms,
)
from src.api.triage.text_utils import norm


CASES: list[tuple[str, str | None, str]] = [
    ("şekerim 60", "YELLOW", "54-69 bosluk"),
    ("şekerim 45", "EMERGENCY", "<54 hard"),
    ("İnsülinim yok ama şekerim 45", "EMERGENCY", "buyuk I pozisyon kaymamali"),
    ("Ramazan orucundayım, şeker 68", "EMERGENCY", "Ramazan + <70"),
    ("glukoz 280, sürekli kusuyorum", "EMERGENCY", ">=250 + DKA"),
    ("glukoz 280, halsizim", "EMERGENCY", ">=250 + halsizlik"),
    ("şeker 260, sorun yok", "YELLOW", ">=250 semptomsuz"),
    ("şeker 250, sürekli kusuyorum", "EMERGENCY", "tam 250 sinir >= "),
    ("şeker 650, ağzım çok kurudu", "EMERGENCY", ">=600 + dehidratasyon"),
    ("şeker 600, ağzım kurudu", "EMERGENCY", "tam 600 sinir"),
    ("şeker 280, bilincim bulanık", "EMERGENCY", ">=250 + bilinc"),
    ("300 lira verdim", None, "glukoz baglami yok"),
    ("şekerim 280 ama sonuçlarım kusursuz", "YELLOW", "kusursuz DKA tetiklemesin"),
    ("5.2 mmol düşük mü", "YELLOW", "mmol yalniz fail-safe"),
    ("şeker 5 mmol/L", "YELLOW", "mmol + seker"),
    ("şekerim 150", None, "70-250 sessiz"),
    ("oruçluyum glukoz 320", "EMERGENCY", "Ramazan + >=300"),
    ("kan şekerim 700", "YELLOW", ">=600 semptomsuz"),
]


def main() -> None:
    """Senaryolari kos; hata varsa exit 1."""
    failed = 0
    print("=== numeric evaluate_numeric_triage ===\n")

    # Unit: kusursuz bayrak False olmali
    assert not flag_dka_symptoms(norm("sonuclarim kusursuz")), "kusursuz false positive"

    for msg, expected, note in CASES:
        res = evaluate_numeric_triage(msg)
        got = None if res is None else res.level
        ok = got == expected
        mark = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        detail = ""
        if res is not None:
            detail = f" glucose={res.glucose_mgdl} reason={res.reason!r}"
        print(f"[{mark}] expected={expected!r} got={got!r} | {note}")
        print(f"       msg={msg!r}{detail}\n")

    print("=== detect_triage kopru (smoke) ===\n")
    bridge = [
        ("şekerim 45", "EMERGENCY"),
        ("şekerim 60", "YELLOW"),
        ("kaç ünite insulin", "REFUSE"),
        ("prediyabet nedir", "GREEN"),
        ("şeker 280, bilincim bulanık", "EMERGENCY"),
    ]
    for msg, expected in bridge:
        got = detect_triage(msg)
        ok = got == expected
        mark = "OK" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] detect_triage({msg!r}) -> {got!r} (beklenen {expected!r})")

    print()
    if failed:
        print(f"BASARISIZ: {failed} senaryo")
        raise SystemExit(1)
    print("Tum senaryolar gecti.")


if __name__ == "__main__":
    main()
