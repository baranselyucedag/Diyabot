#!/usr/bin/env python
"""Grey-band eşik grid search (manuel seed → fine-tune).

Hedefler yön gösterici (küçük eval'de yüksek varyans):
  YELLOW recall >= 0.85, GREEN precision >= 0.90 — hard gate değil.

Calistir:
  set TRIAGE_SKIP_FT=1
  set TRIAGE_SKIP_LLM=1
  python -m src.eval.tune_band
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRIAGE_SKIP_FT", "1")
os.environ.setdefault("TRIAGE_SKIP_LLM", "1")

from src.api.triage.ft_encoder import score_ft
from src.api.triage.fusion import FusionConfig, evaluate_fusion
from src.api.triage.numeric import evaluate_numeric_triage
from src.api.triage.regex_flags import SOFT_YELLOW_WEIGHTS, evaluate_regex_flags


def _soft_flags(msg: str) -> list[str]:
    regex = evaluate_regex_flags(msg)
    if regex is None:
        return []
    return [f for f in regex.all_flags if f in SOFT_YELLOW_WEIGHTS]


def _is_hard(msg: str) -> bool:
    num = evaluate_numeric_triage(msg)
    if num is not None and num.level == "EMERGENCY":
        return True
    regex = evaluate_regex_flags(msg)
    return bool(regex and regex.level in {"EMERGENCY", "REFUSE"})


def _fusion_level(
    msg: str, cfg: FusionConfig
) -> str:
    """Hard veto yoksa fusion band → YELLOW/GREEN (grey→YELLOW tempered varsayım)."""
    if _is_hard(msg):
        num = evaluate_numeric_triage(msg)
        if num and num.level == "EMERGENCY":
            return "EMERGENCY"
        regex = evaluate_regex_flags(msg)
        if regex and regex.level:
            return regex.level
    num = evaluate_numeric_triage(msg)
    num_y = bool(num and num.level == "YELLOW")
    soft = _soft_flags(msg)
    ft = score_ft(msg)
    fus = evaluate_fusion(
        msg, numeric_yellow=num_y, soft_flags=soft, ft_score=ft, config=cfg
    )
    if fus.guarded_level:
        return fus.guarded_level
    if fus.band == "above":
        return "YELLOW"
    if fus.band == "below":
        return "GREEN"
    return "YELLOW"  # grey → tempered default


def main() -> None:
    """Gold üzerinde basit band grid."""
    gold_path = ROOT / "data" / "gold" / "gold_set.jsonl"
    if not gold_path.exists():
        print(f"Gold yok: {gold_path}")
        raise SystemExit(1)

    rows = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    # Soft precision notu: EMERGENCY haric; soft trigger vs YELLOW/GREEN
    print("=== Soft pattern kabataslak precision (EMERGENCY haric) ===\n")
    from collections import defaultdict

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    for r in rows:
        label = str(r.get("expected_triage") or "")
        if label in {"EMERGENCY", "RED_REFUSE"}:
            continue
        q = r.get("question") or ""
        soft = _soft_flags(q)
        for f in soft:
            if label == "YELLOW":
                tp[f] += 1
            elif label == "GREEN":
                fp[f] += 1
    for f in sorted(set(tp) | set(fp)):
        t, f_ = tp[f], fp[f]
        prec = t / (t + f_) if (t + f_) else 0.0
        print(f"  {f}: tp={t} fp={f_} prec={prec:.2f}")

    print("\n=== Band grid (SKIP_FT/LLM) ===\n")
    lows = [0.20, 0.25, 0.30, 0.35]
    highs = [0.55, 0.60, 0.65, 0.70]
    best = None
    for lo in lows:
        for hi in highs:
            if lo >= hi:
                continue
            cfg = FusionConfig(band_low=lo, band_high=hi)
            # Sadece GREEN/YELLOW satirlari (fusion etkisi)
            y_tp = y_fn = g_tp = g_fp = 0
            for r in rows:
                exp = str(r.get("expected_triage") or "")
                if exp not in {"GREEN", "YELLOW"}:
                    continue
                if _is_hard(r.get("question") or ""):
                    continue
                pred = _fusion_level(r.get("question") or "", cfg)
                if exp == "YELLOW":
                    if pred == "YELLOW":
                        y_tp += 1
                    else:
                        y_fn += 1
                elif exp == "GREEN":
                    if pred == "GREEN":
                        g_tp += 1
                    else:
                        g_fp += 1
            y_rec = y_tp / (y_tp + y_fn) if (y_tp + y_fn) else 0.0
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0.0
            score = y_rec + g_prec
            print(
                f"  low={lo:.2f} high={hi:.2f} Y_rec={y_rec:.2f} G_prec={g_prec:.2f} "
                f"(nY={y_tp+y_fn} nG={g_tp+g_fp})"
            )
            if best is None or score > best[0]:
                best = (score, lo, hi, y_rec, g_prec)

    if best:
        print(
            f"\nEn iyi (yon gosterici): low={best[1]} high={best[2]} "
            f"Y_rec={best[3]:.2f} G_prec={best[4]:.2f}"
        )
        print("Not: n kucukse varyans yuksek; hard CI gate degil.")


if __name__ == "__main__":
    main()
