#!/usr/bin/env python
"""Grey-band esik grid search.

Varsayilan set: data/gold/triage_test_set.json
  is_really_yellow true  → beklenen YELLOW
  is_really_yellow false → beklenen GREEN

Calistir (proje kokunden):
  set TRIAGE_SKIP_IMPLICIT=1
  set TRIAGE_SKIP_LLM=1
  python -m src.eval.benchmarks.tune_band
  python -m src.eval.benchmarks.tune_band data/gold/triage_test_set.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TRIAGE_SKIP_IMPLICIT", "1")
os.environ.setdefault("TRIAGE_SKIP_LLM", "1")

from src.api.triage.implicit_score import score_implicit
from src.api.triage.fusion import FusionConfig, evaluate_fusion
from src.api.triage.numeric import evaluate_numeric_triage
from src.api.triage.regex_flags import SOFT_YELLOW_WEIGHTS, evaluate_regex_flags

DEFAULT_SET = ROOT / "data" / "gold" / "triage_test_set.json"


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


def _fusion_level(msg: str, cfg: FusionConfig) -> str:
    """Hard veto yoksa fusion band → YELLOW/GREEN (grey→YELLOW)."""
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
    imp = score_implicit(msg)
    fus = evaluate_fusion(
        msg,
        numeric_yellow=num_y,
        soft_flags=soft,
        implicit_score=imp,
        config=cfg,
    )
    if fus.guarded_level:
        return fus.guarded_level
    if fus.band == "above":
        return "YELLOW"
    if fus.band == "below":
        return "GREEN"
    return "YELLOW"


def _expected_level(r: dict) -> str | None:
    """triage_test_set (is_really_yellow) veya gold (expected_triage)."""
    if "is_really_yellow" in r and r["is_really_yellow"] is not None:
        return "YELLOW" if r["is_really_yellow"] is True else "GREEN"
    exp = str(r.get("expected_triage") or "")
    if exp in {"GREEN", "YELLOW"}:
        return exp
    return None


def _load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise SystemExit("Beklenen JSON dizi veya JSONL.")
    return data


def main() -> None:
    set_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SET
    if not set_path.is_absolute():
        set_path = ROOT / set_path
    if not set_path.exists():
        print(f"Dosya yok: {set_path}")
        raise SystemExit(1)

    rows = _load_rows(set_path)
    print(f"=== Band tune seti: {set_path} (n={len(rows)}) ===\n")

    print("=== Soft precision (is_really_yellow / expected) ===\n")
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    for r in rows:
        exp = _expected_level(r)
        if exp is None:
            continue
        soft = _soft_flags(r.get("question") or "")
        for f in soft:
            if exp == "YELLOW":
                tp[f] += 1
            else:
                fp[f] += 1
    for f in sorted(set(tp) | set(fp), key=lambda x: (-(tp[x] / (tp[x] + fp[x]) if tp[x] + fp[x] else 0), x)):
        t, f_ = tp[f], fp[f]
        n = t + f_
        prec = t / n if n else 0.0
        print(f"  {f}: tp={t} fp={f_} n={n} prec={prec:.2f}")

    print("\n=== Band grid (SKIP_IMPLICIT/LLM) ===\n")
    lows = [0.20, 0.25, 0.30, 0.35]
    highs = [0.55, 0.60, 0.65, 0.70]
    best = None
    for lo in lows:
        for hi in highs:
            if lo >= hi:
                continue
            cfg = FusionConfig(band_low=lo, band_high=hi)
            y_tp = y_fn = g_tp = g_fp = 0
            for r in rows:
                exp = _expected_level(r)
                if exp is None:
                    continue
                q = r.get("question") or ""
                if _is_hard(q):
                    continue
                pred = _fusion_level(q, cfg)
                if exp == "YELLOW":
                    if pred == "YELLOW":
                        y_tp += 1
                    else:
                        y_fn += 1
                else:
                    if pred == "GREEN":
                        g_tp += 1
                    else:
                        g_fp += 1
            y_rec = y_tp / (y_tp + y_fn) if (y_tp + y_fn) else 0.0
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) else 0.0
            score = y_rec + g_prec
            print(
                f"  low={lo:.2f} high={hi:.2f} Y_rec={y_rec:.2f} G_prec={g_prec:.2f} "
                f"(nY={y_tp + y_fn} nG={g_tp + g_fp})"
            )
            if best is None or score > best[0]:
                best = (score, lo, hi, y_rec, g_prec)

    if best:
        print(
            f"\nEn iyi (yon gosterici): low={best[1]} high={best[2]} "
            f"Y_rec={best[3]:.2f} G_prec={best[4]:.2f}"
        )
        print("Not: soft agirliklar kilitli; band secimi yon gosterici.")
        print("Once klinik kacirmayan (Y_rec), sonra egitim kirletmeyen (G_prec).")


if __name__ == "__main__":
    main()
