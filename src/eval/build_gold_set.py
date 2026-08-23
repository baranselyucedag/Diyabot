#!/usr/bin/env python
"""Altın set üretici — `data/gold/authoring/cases.jsonl` → `gold_set.jsonl`.

Tek kaynak: cases.jsonl (her satır bir soru; chunk_ids kayıt içinde).
`--validate` sıkı kontroller + kota raporu; hata varsa üretim durur.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
AUTHORING_DIR = ROOT / "data" / "gold" / "authoring"
CASES_PATH = AUTHORING_DIR / "cases.jsonl"
OUT_PATH = ROOT / "data" / "gold" / "gold_set.jsonl"

VALID_TRIAGE = {"GREEN", "YELLOW", "EMERGENCY", "RED_REFUSE"}

# Plan kota hedefleri (retrieval'a giren sorular; RED_REFUSE hariç yaklaşık)
CATEGORY_QUOTAS: dict[str, int] = {
    "1-hastaligi-anlama": 20,
    "1-hedef-deger": 12,
    "1-prediyabet": 5,
    "2-acil": 10,
    "2-ilk-yardim": 5,
    "3-cihaz": 12,
    "3-olcum-yorum": 8,
    "4-ilac": 20,
    "4-ilac-yan-etki": 10,
    "5-beslenme": 25,
    "6-egzersiz": 15,
    "7-komplikasyon": 20,
    "8-psikososyal": 8,
    "8-sick-day": 5,
    "8-kadin-sagligi": 6,
    "8-mit": 6,
    "8-ehliyet": 3,
    "8-seyahat": 3,
    "guvenlik-red": 15,
}

TRIAGE_TARGETS = {
    "GREEN": 0.50,
    "YELLOW": 0.30,
    "EMERGENCY": 0.10,
    "RED_REFUSE": 0.10,
}


def tr_lower(text: str) -> str:
    return (
        text.replace("İ", "i")
        .replace("I", "ı")
        .replace("Ş", "ş")
        .replace("Ğ", "ğ")
        .replace("Ü", "ü")
        .replace("Ö", "ö")
        .replace("Ç", "ç")
        .lower()
    )


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", tr_lower(text or "").strip())


def load_chunk_ids() -> set[str]:
    ids: set[str] = set()
    for path in sorted(PROCESSED.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(json.loads(line)["chunk_id"])
    return ids


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Authoring dosyası yok: {path}")
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} JSON hatası: {exc}") from exc
    return rows


def validate_cases(
    cases: list[dict[str, Any]],
    chunk_ids: set[str],
) -> list[str]:
    """Sıkı doğrulama. Dönen liste hata mesajları (boş = OK)."""
    errors: list[str] = []
    required = (
        "case_id",
        "persona",
        "hooks",
        "q",
        "category",
        "triage",
        "must_include",
        "must_not_include",
        "keywords",
        "summary",
        "chunk_ids",
    )
    seen_q: dict[str, int] = {}

    for i, row in enumerate(cases, 1):
        for key in required:
            if key not in row:
                errors.append(f"satır {i}: eksik alan '{key}'")
        if "q" not in row:
            continue

        q = row["q"]
        norm = normalize_question(q)
        if not norm:
            errors.append(f"satır {i}: boş soru")
        elif norm in seen_q:
            errors.append(
                f"satır {i}: tekrarlayan soru (satır {seen_q[norm]} ile aynı): {q!r}"
            )
        else:
            seen_q[norm] = i

        triage = row.get("triage")
        if triage not in VALID_TRIAGE:
            errors.append(f"satır {i}: geçersiz triage {triage!r}")

        cat = row.get("category")
        if not cat or not isinstance(cat, str):
            errors.append(f"satır {i}: kategori eksik/geçersiz")

        for list_key in ("hooks", "must_include", "must_not_include", "keywords", "chunk_ids"):
            val = row.get(list_key)
            if val is not None and not isinstance(val, list):
                errors.append(f"satır {i}: '{list_key}' liste olmalı")

        chunk_list = row.get("chunk_ids") or []
        if triage == "RED_REFUSE":
            if chunk_list:
                errors.append(f"satır {i}: RED_REFUSE için chunk_ids boş olmalı")
        else:
            if not chunk_list and not row.get("allow_gap"):
                # gap'e izin: coverage_status=gap; uyarı olarak raporlanır, hata değil
                pass
            for cid in chunk_list:
                if cid not in chunk_ids:
                    errors.append(f"satır {i}: chunk korpusta yok: {cid} (soru: {q!r})")

        para = row.get("paraphrase_of")
        if para is not None and not isinstance(para, str):
            errors.append(f"satır {i}: paraphrase_of string veya null olmalı")

    # paraphrase_of referansları
    by_q = {r["q"]: r for r in cases if "q" in r}
    for i, row in enumerate(cases, 1):
        para = row.get("paraphrase_of")
        if para:
            if para not in by_q:
                errors.append(f"satır {i}: paraphrase_of hedefi yok: {para!r}")
            elif row.get("chunk_ids") != by_q[para].get("chunk_ids"):
                errors.append(
                    f"satır {i}: paraphrase chunk_ids ana soru ile aynı olmalı"
                )

    return errors


def quota_report(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    cat = Counter(r.get("category", "?") for r in cases)
    triage = Counter(r.get("triage", "?") for r in cases)
    n = len(cases) or 1
    verified = sum(
        1
        for r in cases
        if r.get("triage") != "RED_REFUSE" and (r.get("chunk_ids") or [])
    )
    weak = sum(1 for r in cases if r.get("weak"))
    gap = sum(
        1
        for r in cases
        if r.get("triage") != "RED_REFUSE" and not (r.get("chunk_ids") or [])
    )
    para = sum(1 for r in cases if r.get("paraphrase_of"))

    lines.append("--- KOTA / DAĞILIM RAPORU ---")
    lines.append(f"Toplam soru        : {len(cases)}")
    lines.append(f"Küratör (chunk'lı) : {verified}")
    lines.append(f"Weak / Gap / Parafraz: {weak} / {gap} / {para}")
    lines.append("")
    lines.append("Triage dağılımı (hedef ~%50/%30/%10/%10):")
    for t in ("GREEN", "YELLOW", "EMERGENCY", "RED_REFUSE"):
        count = triage.get(t, 0)
        pct = 100.0 * count / n
        target = 100.0 * TRIAGE_TARGETS[t]
        lines.append(f"  {t:12s} {count:3d}  ({pct:5.1f}% | hedef ~{target:.0f}%)")
    lines.append("")
    lines.append("Kategori sayıları (plan kotalarına göre):")
    all_cats = sorted(set(CATEGORY_QUOTAS) | set(cat))
    for c in all_cats:
        have = cat.get(c, 0)
        want = CATEGORY_QUOTAS.get(c)
        mark = ""
        if want is not None:
            if have < want:
                mark = f"  ← eksik ({want - have})"
            elif have >= want:
                mark = "  ✓"
        lines.append(f"  {c:28s} {have:3d}" + (f" / {want}" if want else "") + mark)
    unknown = [c for c in cat if c not in CATEGORY_QUOTAS]
    if unknown:
        lines.append("")
        lines.append(f"Taksonomi dışı kategoriler: {unknown}")
    lines.append("")
    target_total = 150
    lines.append(
        f"Hedef ilerleme: {verified}/{target_total} küratör onaylı "
        f"({100.0 * verified / target_total:.0f}%)"
    )
    return "\n".join(lines)


def build_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i, row in enumerate(cases, 1):
        is_refusal = row["triage"] == "RED_REFUSE"
        chunk_ids = list(row.get("chunk_ids") or [])

        if is_refusal:
            coverage = "not_applicable"
            expected: list[str] = []
        elif not chunk_ids:
            coverage = "gap"
            expected = []
        elif row.get("weak"):
            coverage = "weak"
            expected = chunk_ids
        else:
            coverage = "ok"
            expected = chunk_ids

        record = {
            "id": f"gold_{i:03d}",
            "case_id": row["case_id"],
            "case_persona": row["persona"],
            "personalization_hooks": row.get("hooks") or [],
            "question": row["q"],
            "category": row["category"],
            "expected_triage": row["triage"],
            "safety_critical": bool(row.get("safety_critical", False)),
            "must_include": row.get("must_include") or [],
            "must_not_include": row.get("must_not_include") or [],
            "retrieval_keywords": row.get("keywords") or [],
            "expected_answer_summary": row.get("summary") or "",
            "expected_answer": row.get("expected_answer") or "",
            "expected_chunk_ids": expected,
            "coverage_status": coverage,
            "curator_verified": not is_refusal and bool(expected),
            "paraphrase_of": row.get("paraphrase_of"),
            "weak": bool(row.get("weak", False)),
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Altın set üretici (cases.jsonl).")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Sıkı doğrulama; hata varsa üretme, kota raporunu yaz.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Sadece kota raporu; gold_set yazma.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    chunk_ids = load_chunk_ids()

    errors = validate_cases(cases, chunk_ids)
    report = quota_report(cases)
    print(report)

    if errors:
        print("\n--- DOĞRULAMA HATALARI ---")
        for e in errors:
            print(f"  ✗ {e}")
        if args.validate or not args.report_only:
            raise SystemExit(f"\n{len(errors)} hata — üretim durdu.")
    else:
        print("\nDoğrulama: OK")

    if args.report_only:
        return

    records = build_records(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    triage = Counter(r["expected_triage"] for r in records)
    cov = Counter(r["coverage_status"] for r in records)
    verified = sum(1 for r in records if r["curator_verified"])
    safety = sum(1 for r in records if r["safety_critical"])

    print("\n--- ALTIN SET RAPORU ---")
    print(f"Toplam soru      : {len(records)}")
    print(f"Vaka sayısı      : {len({r['case_id'] for r in records})}")
    print(f"Triage           : {dict(triage)}")
    print(f"Güvenlik-kritik  : {safety}")
    print(f"Kapsama          : {dict(cov)}")
    print(f"Küratör onaylı   : {verified}")
    print(f"\nKaydedildi: {args.output}")
    print("NOT: expected_chunk_ids KÜRATÖR ONAYLIDIR (elle doğrulandı).")


if __name__ == "__main__":
    main()
