"""Bekleyen çakışmaların (pending conflicts) süre sonu temizliği.

Plan (cahing_mimari_tasarim.md §5):
- Süresi geçen PENDING çakışmalar EXPIRED_REJECTED olarak işaretlenir (sessizce
  silinmez) ve loglanır.
- Kullanıcıya bir sonraki konuşmada "önceki bilgiyi netleştiremedik, tekrar
  belirtir misin" takip notu bırakılır (append_followup_note).

Kilitleme:
- `expire_pending_conflicts_locked` LOCK İÇİNDE çağrılmalıdır (internal).
- `expire_pending_conflicts` public API'dir; conversation_lock'ı alıp locked
  versiyonu çağırır. memory_maintenance_task içinde ayrıca çağrılmaz — aynı
  lock içinde zaten çalışır (plan: "ayrı task YOK").
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Optional

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.logger import log_event
from src.api.memory.models import (
    Note,
    NotesStore,
    NoteStaleness,
    PendingConflictStatus,
)
from src.api.memory.storage import (
    get_conversation_lock,
    load_notes,
    load_pending_conflicts,
    save_pending_conflicts,
)
from src.api.memory.memory_store import save_notes_with_limit
from src.api.memory.timeutil import utcnow


def append_followup_note(conv_id: str, message: str) -> None:
    """notes.json'a category="advice" bir takip notu ekler."""
    notes = load_notes(conv_id)
    if notes is None:
        notes = NotesStore(version=1, updated_at=utcnow(), items=[])

    notes.items.append(
        Note(
            note_id=f"note_{uuid.uuid4().hex[:12]}",
            content=message,
            category="advice",
            source_turns=[],
            confidence=1.0,
            staleness=NoteStaleness.FRESH,
        )
    )
    notes.updated_at = utcnow()
    save_notes_with_limit(conv_id, notes)


def expire_pending_conflicts_locked(
    conv_id: str,
    now: Optional[datetime] = None,
    expiry_days: Optional[int] = None,
) -> int:
    """LOCK İÇİNDE çağrılır: süresi geçen PENDING çakışmaları expired_rejected yapar.

    Süresi dolan her çakışma için ayrıca kullanıcıya takip notu bırakılır
    (append_followup_note) ve olay loglanır.

    Returns:
        Süresi geçirilen (durumu değişen) çakışma sayısı.
    """
    now = now or utcnow()
    days = (
        expiry_days
        if expiry_days is not None
        else MEMORY_CONFIG["pending_conflict_expiry_days"]
    )
    cutoff = now - timedelta(days=days)

    store = load_pending_conflicts(conv_id)
    if store is None:
        return 0

    expired = 0
    for conflict in store.items:
        if (
            conflict.status == PendingConflictStatus.PENDING
            and conflict.created_at < cutoff
        ):
            conflict.status = PendingConflictStatus.EXPIRED_REJECTED
            expired += 1

            log_event(
                "pending_conflict",
                "expired_rejected",
                conv_id,
                conflict_id=conflict.conflict_id,
                type=conflict.type,
            )
            append_followup_note(
                conv_id,
                f"Önceki mesajınızdaki {conflict.type} ile ilgili bilgiyi "
                "netleştiremedik, tekrar belirtir misiniz?",
            )

    if expired:
        save_pending_conflicts(conv_id, store)

    return expired


async def expire_pending_conflicts(conv_id: str) -> int:
    """Public API: conversation_lock alıp expire_pending_conflicts_locked çağırır."""
    lock = get_conversation_lock(conv_id)
    async with lock:
        return await asyncio.to_thread(
            expire_pending_conflicts_locked, conv_id
        )
