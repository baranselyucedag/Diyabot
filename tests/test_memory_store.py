"""memory_store testleri — context yükleme, eviction, atomik turn oluşturma."""

from __future__ import annotations

import asyncio

from src.api.memory.memory_store import (
    create_turn_atomic,
    load_memory_context,
    save_notes_with_limit,
)
from src.api.memory.models import Note, NotesStore, NoteStaleness
from src.api.memory.storage import load_index, load_notes, load_turns


def test_load_memory_context_defaults():
    ctx = load_memory_context("conv_c1")
    assert ctx["profile"] is not None
    assert ctx["notes"].items == []
    assert ctx["index"].turn_count == 0
    assert ctx["summary"] == ""


def test_save_notes_with_limit_eviction():
    notes = NotesStore(items=[])
    for i in range(5):
        notes.items.append(
            Note(
                note_id=f"n{i}",
                content=f"not {i}",
                category="observation",
                confidence=0.5 + i * 0.05,
                staleness=NoteStaleness.FRESH,
            )
        )
    # n0'ı stale yap → eviction önceliği en düşük olur
    notes.items[0].staleness = NoteStaleness.STALE

    save_notes_with_limit("conv_c2", notes, max_notes=3)

    saved = load_notes("conv_c2")
    assert saved is not None
    assert len(saved.items) == 3
    # stale olan not evicted edilmiş olmalı
    assert all(n.note_id != "n0" for n in saved.items)


def test_create_turn_atomic_concurrent():
    async def main():
        await asyncio.gather(
            create_turn_atomic("conv_c3", "user", "mesaj bir", None),
            create_turn_atomic("conv_c3", "user", "mesaj iki", None),
            create_turn_atomic("conv_c3", "assistant", "cevap", None),
        )

    asyncio.run(main())

    index = load_index("conv_c3")
    assert index is not None
    assert index.turn_count == 3

    turns = load_turns("conv_c3")
    ids = [t.turn_id for t in turns]
    assert len(ids) == 3
    assert len(set(ids)) == 3  # turn_id çakışması yok


def test_save_notes_with_limit_does_not_mutate_caller_list():
    """Çağıranın elindeki liste nesnesi yerinde değiştirilmemeli (kopya sort)."""
    notes = NotesStore(items=[])
    for i in range(3):
        notes.items.append(
            Note(note_id=f"m{i}", content=f"not {i}", category="observation", confidence=0.5)
        )
    notes.items[0].staleness = NoteStaleness.STALE  # sıralamada sona gider
    original_list = notes.items
    original_order = [n.note_id for n in original_list]

    save_notes_with_limit("conv_c4", notes, max_notes=5)

    # Liste nesnesinin kendisi ve sırası korunur; atama yeni listedir.
    assert [n.note_id for n in original_list] == original_order
