"""Retention temizliği — retention_days'den eski konuşma verisini siler.

Kullanım:
    python -m src.api.memory.retention_cleanup                # dry-run (yalnızca listeler)
    python -m src.api.memory.retention_cleanup --confirm      # gerçekten siler
    python -m src.api.memory.retention_cleanup --days 90      # retention_days override

Güvenlik:
- Varsayılan DRY-RUN'dur; ``--confirm`` verilmeden HİÇBİR dosya silinmez.
- Yaş, ``data/memory/{conv_id}/index.json`` içindeki ``created_at`` alanından
  hesaplanır. Index yoksa/bozuksa o konuşma ADAY OLMAZ (bilinmeyen yaş güvenli
  tarafta kalır: silinmez).
- Silinen her konuşma ``log_event("retention", "conversation_deleted", ...)``
  ile audit trail olarak loglanır.

Bu script otomatik bir cron/scheduler DEĞİLDİR; elle veya işletim sisteminin
kendi zamanlayıcısıyla (Windows Task Scheduler, cron) çalıştırılır.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.api.memory import storage
from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.logger import log_event


def _as_utc(dt: datetime) -> datetime:
    """Naive datetime ise UTC kabul eder; aksi halde olduğu gibi döndürür."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def find_expired_conversations(
    now: Optional[datetime] = None,
    days: Optional[int] = None,
) -> list[tuple[str, float]]:
    """``retention_days``'den eski konuşmaları ``(conv_id, age_days)`` listesi olarak döndürür.

    Yaş, ``index.json`` içindeki ``created_at``'ten hesaplanır; index yoksa veya
    bozuksa o konuşma aday olmaz (bilinmeyen yaş silinmez).
    """
    if days is None:
        days = MEMORY_CONFIG["retention_days"]
    if now is None:
        now = datetime.now(timezone.utc)

    if not storage.PROFILES_DIR.is_dir():
        return []

    candidates: list[tuple[str, float]] = []
    for profile_path in sorted(storage.PROFILES_DIR.glob("*.json")):
        conv_id = profile_path.stem
        try:
            storage.validate_conv_id(conv_id)
        except ValueError:
            continue  # geçersiz/şüpheli dosya adı — dokunma

        index = storage.load_index(conv_id)
        if index is None:
            continue  # index yok → yaş bilinmiyor

        created = _as_utc(index.created_at)
        age_days = (now - created).total_seconds() / 86400.0
        if age_days > days:
            candidates.append((conv_id, age_days))

    return candidates


def delete_conversation(conv_id: str, age_days: float) -> None:
    """Bir konuşmanın profil dosyasını ve memory klasörünü siler + loglar."""
    profile_path = storage.PROFILES_DIR / f"{conv_id}.json"
    memory_path = storage.MEMORY_DIR / conv_id

    if profile_path.exists():
        profile_path.unlink()
    if memory_path.is_dir():
        shutil.rmtree(memory_path)

    log_event(
        "retention",
        "conversation_deleted",
        conv_id,
        age_days=round(age_days, 1),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="retention_days'den eski konuşma verisini siler (dry-run varsayılan)."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Gerçekten sil (verilmezse dry-run).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"retention_days override (varsayılan: {MEMORY_CONFIG['retention_days']}).",
    )
    args = parser.parse_args(argv)

    days = args.days if args.days is not None else MEMORY_CONFIG["retention_days"]
    candidates = find_expired_conversations(days=days)

    if not candidates:
        print(f"Silinecek konuşma yok (retention_days={days}).")
        return 0

    print(f"{len(candidates)} konuşma {days} günden eski:")
    for conv_id, age in candidates:
        print(f"  - {conv_id} (yaklaşık {age:.1f} gün eski)")

    if not args.confirm:
        print("\n[dry-run] Hiçbir şey silinmedi. Silmek için --confirm ekleyin.")
        return 0

    for conv_id, age in candidates:
        delete_conversation(conv_id, age)
        print(f"  silindi: {conv_id}")

    print(f"{len(candidates)} konuşma silindi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
