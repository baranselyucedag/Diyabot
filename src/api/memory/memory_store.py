"""Bellek okuma/yazma katmanı — high-level işlemler (storage.py üzerinde).

Async/sync notu:
  - create_turn_atomic VE append_turn_and_update_index ASYNC'tir; ikisi de
    conversation-level asyncio.Lock (get_conversation_lock) altında çalışır.
    Aynı conv_id için turn ekleyen HERHANGİ bir yol bu kilidi kullanmalı —
    aksi halde turn_id çakışması / index yarışı geri döner (bkz. aşağıdaki
    "GÜVENLİK NOTU").
  - Diğer yükleyiciler (load_memory_context, load_recent_turns_for_prompt, vb.)
    senkrondur; bunlar disk I/O + portalocker üzerinden çalışır.

GÜVENLİK NOTU (önceki hata, düzeltildi):
  Daha önce `append_turn_and_update_index` KİLİTSİZ, senkron bir fonksiyon
  olarak yeniden eklenmişti — tam olarak daha önce bilerek sildiğimiz
  `_append_turn_and_update_index` ikizinin aynısı. Bu, kullanıcı turu
  `create_turn_atomic` (kilitli) ile, asistan turu ise bu fonksiyonla
  (kilitsiz) kaydedilirse, aynı konuşmaya eşzamanlı istek geldiğinde
  turn_id çakışmasına yol açabiliyordu. Şimdi bu fonksiyon da
  `create_turn_atomic` ile AYNI kilidi kullanıyor ve turn_id'yi kendisi,
  kilit İÇİNDE üretiyor — çağıranın önceden hazır bir `Turn` nesnesi (kendi
  ürettiği turn_id ile) geçmesine artık izin verilmiyor, çünkü turn_id'nin
  kilit dışında üretilmesi bizzat riskin kendisiydi.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.models import (
    MemoryIndex,
    Note,
    NotesStore,
    NoteStaleness,
    PendingConflictsStore,
    PatientProfile,
    Profile,
    ProfileMetadata,
    Turn,
    TurnRole,
)
from src.api.memory.storage import (
    append_turn,
    get_conversation_lock,
    load_index,
    load_notes,
    load_pending_conflicts,
    load_profile,
    load_summary,
    load_turns,
    save_index,
    save_notes,
    save_pending_conflicts,
)
from src.api.memory.timeutil import utcnow


# ---------------------------------------------------------------------------
# Context yükleme
# ---------------------------------------------------------------------------


def load_memory_context(conv_id: str) -> dict:
    """Tek çağrıda tüm bellek bileşenlerini yükler (yoksa boş defaultlar)."""
    profile = load_profile(conv_id)
    if profile is None:
        # Yeni konuşma için boş profil
        profile = Profile(
            version=1,
            updated_at=utcnow(),
            patient=PatientProfile(
                diabetes_type="Type 2",
                diagnosis_year=utcnow().year,
                medications=[],
                complications=[],
                monitoring={},
                allergies=[],
                goals=[],
                lifestyle={},
            ),
            metadata=ProfileMetadata(
                source_turn_ids=[],
                confidence_scores={},
                updated_at=utcnow(),
            ),
        )

    summary = load_summary(conv_id) or ""
    notes = load_notes(conv_id)
    if notes is None:
        notes = NotesStore(version=1, updated_at=utcnow(), items=[])

    pending = load_pending_conflicts(conv_id)
    if pending is None:
        pending = PendingConflictsStore(items=[])

    index = load_index(conv_id)
    if index is None:
        index = MemoryIndex(
            turn_count=0,
            last_summary_at_turn=0,
            last_note_extraction_at_turn=0,
            last_profile_update_at_turn=0,
            created_at=utcnow(),
        )

    return {
        "profile": profile,
        "summary": summary,
        "notes": notes,
        "pending_conflicts": pending,
        "index": index,
    }


# ---------------------------------------------------------------------------
# Turn yönetimi (atomik) — TEK senkron çekirdek, İKİ async giriş noktası
# ---------------------------------------------------------------------------
#
# Hem create_turn_atomic hem append_turn_and_update_index aynı
# _create_turn_atomic_sync çekirdeğini, aynı conversation_lock altında
# çağırır. Böylece iki ayrı "turn kaydetme yolu" olsa bile, ikisi de aynı
# güvenlik garantisine sahip olur — kod tekrarı da ortadan kalkar.


def _create_turn_atomic_sync(
    conv_id: str,
    role: TurnRole | str,
    content: str,
    triage: Optional[str],
) -> Turn:
    """`create_turn_atomic`'in SENKRON çekirdeği (thread pool içinde çalışır).

    index oku → turn_id üret → turn ekle → index artır → kaydet.

    ÖNEMLİ: turn_id burada, çekirdek kilit altındayken üretiliyor. Bu
    fonksiyonu kilit DIŞINDA, doğrudan çağırma — turn_id çakışması riski
    budur.
    """
    index = load_index(conv_id)
    if index is None:
        index = MemoryIndex(
            turn_count=0,
            last_summary_at_turn=0,
            last_note_extraction_at_turn=0,
            last_profile_update_at_turn=0,
            created_at=utcnow(),
        )

    turn_number = index.turn_count + 1
    turn_id = f"msg_{turn_number}"

    turn = Turn(
        turn_id=turn_id,
        role=role,  # "user" | "assistant"
        content=content,
        timestamp=utcnow(),
        triage=triage,
    )

    append_turn(conv_id, turn)

    index.turn_count = turn_number
    index.updated_at = utcnow()
    save_index(conv_id, index)

    return turn


async def create_turn_atomic(
    conv_id: str,
    role: TurnRole | str,
    content: str,
    triage: Optional[str] = None,
) -> Turn:
    """
    ATOMIK: conversation_lock içinde index oku → turn_id üret → turn ekle → index artır → kaydet.
    Eşzamanlı isteklerde turn_id çakışması engellenir.

    Disk I/O `asyncio.to_thread` ile thread pool'da çalışır; böylece async event
    loop bloke edilmez. Lock async tutulduğu için aynı konuşmanın eşzamanlı
    istekleri hâlâ sıralıdır.
    """
    lock = get_conversation_lock(conv_id)
    async with lock:
        return await asyncio.to_thread(
            _create_turn_atomic_sync, conv_id, role, content, triage
        )


async def append_turn_and_update_index(
    conv_id: str,
    role: TurnRole | str,
    content: str,
    triage: Optional[str] = None,
) -> Turn:
    """`create_turn_atomic` ile TAMAMEN AYNI davranış — ikinci bir giriş noktası.

    Pipeline'da kullanıcı turu için `create_turn_atomic`, asistan turu için
    bu fonksiyon çağrılıyorsa bile, ikisi de aynı `conversation_lock`'u
    kullanır ve turn_id'yi kilit içinde üretir — hangisi çağrılırsa
    çağrılsın güvenlik garantisi aynıdır.

    NOT: Önceden hazırlanmış, kendi turn_id'sini taşıyan bir `Turn` nesnesi
    KABUL ETMİYORUZ — role/content/triage alıyoruz ve turn_id'yi biz,
    kilit altında üretiyoruz. Kilit dışında üretilmiş bir turn_id'yi kabul
    etmek, önlemeye çalıştığımız çakışma riskinin ta kendisidir.

    Pratikte `create_turn_atomic` ile birebir aynı işi yaptığı için, yeni
    kod yazarken doğrudan `create_turn_atomic`'i çağırman ve bu fonksiyonu
    sadece geriye dönük uyumluluk için tutman önerilir.
    """
    return await create_turn_atomic(conv_id, role, content, triage)


# ---------------------------------------------------------------------------
# Not yönetimi
# ---------------------------------------------------------------------------


def save_notes_with_limit(
    conv_id: str,
    notes: NotesStore,
    max_notes: Optional[int] = None,
) -> None:
    """
    Eviction policy: TUTULACAKLAR ÖNDE tutulur, ARKADAKİLER silinir.

    Öncelik sırası (önemlilik → düşüklük):
      1. FRESH notlar stale notlardan önce.
      2. Yüksek confidence düşük confidencetan önce.
      3. Yeni notlar eski notlardan önce.

    max_notes aşıldığında listenin sonundaki (en düşük öncelikli) notlar kaldırılır.
    max_notes verilmezse MEMORY_CONFIG["max_notes_per_conversation"] kullanılır.
    """
    if max_notes is None:
        max_notes = MEMORY_CONFIG["max_notes_per_conversation"]

    # Kopya üzerinde çalış: çağıranın liste nesnesini yerinde mutate etme.
    items = list(notes.items)

    # Tutulacaklar önde:
    # - fresh (False) < stale (True)  → fresh başta
    # - yüksek confidence başta       → -confidence küçük = yüksek
    # - yeni tarih başta              → -timestamp küçük = yeni
    items.sort(
        key=lambda n: (
            n.staleness.value == "stale",
            -n.confidence,
            -n.created_at.timestamp(),
        ),
        reverse=False,
    )

    if len(items) > max_notes:
        items = items[:max_notes]  # İlk max_notes = en değerli notlar

    notes.items = items
    notes.updated_at = utcnow()
    save_notes(conv_id, notes)


def update_note_staleness(conv_id: str, note_id: str, staleness: str) -> None:
    """Not bul, staleness güncelle, save_notes. Not yoksa yazma yapmaz."""
    notes = load_notes(conv_id)
    if notes is None:
        return

    for note in notes.items:
        if note.note_id == note_id:
            note.staleness = NoteStaleness(staleness)  # enum'a çevirerek ata
            notes.updated_at = utcnow()
            save_notes(conv_id, notes)
            return

    # Not bulunamadı; gereksiz write yapma.


# ---------------------------------------------------------------------------
# Prompt için yardımcı yükleyiciler
# ---------------------------------------------------------------------------


def load_recent_turns_for_prompt(conv_id: str, count: Optional[int] = None) -> list[Turn]:
    """Son N turu kronolojik sırada döndür (prompt için).

    count verilmezse MEMORY_CONFIG["recent_turns_count"] kullanılır.
    """
    if count is None:
        count = MEMORY_CONFIG["recent_turns_count"]
    return load_turns(conv_id, limit=count)


def summarize_notes_for_prompt(notes: NotesStore, exclude_stale: bool = False) -> str:
    """Notları prompt'a eklenecek özet string'e çevir.

    Args:
        notes: Yüklü notlar.
        exclude_stale: True ise STALE işaretli notları listeden çıkar.
    """
    if not notes or not notes.items:
        return "(ilgili not yok)"

    lines = []
    for note in notes.items:
        is_stale = note.staleness.value == "stale"
        if exclude_stale and is_stale:
            continue
        cat = note.category
        conf = f"{note.confidence:.2f}"
        stale = " [STALE]" if is_stale else ""
        lines.append(f"- [{cat} | conf={conf}{stale}] {note.content}")

    return "\n".join(lines) if lines else "(ilgili not yok)"
