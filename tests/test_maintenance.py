"""maintenance + expiry testleri — profil güncelleme, followup, expiry, task akışı."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from src.api.memory.expiry import append_followup_note, expire_pending_conflicts_locked
from src.api.memory.maintenance import (
    _find_source_turn_id_for_evidence,
    apply_profile_update,
    memory_maintenance_task,
)
from src.api.memory.models import (
    CriticalVerifyResult,
    MaintenanceResult,
    Medication,
    MedicationStatus,
    Note,
    NotesStore,
    NoteStaleness,
    PatientProfile,
    PendingConflict,
    PendingConflictsStore,
    PendingConflictStatus,
    Profile,
    ProfileUpdate,
    Turn,
    TurnRole,
)
from src.api.memory.storage import (
    load_notes,
    load_pending_conflicts,
    load_profile,
    save_notes,
    save_pending_conflicts,
    save_profile,
)
from src.api.memory.memory_store import save_notes_with_limit
from src.api.memory.timeutil import utcnow


def _profile() -> Profile:
    return Profile(
        version=1,
        patient=PatientProfile(
            diabetes_type="Type 2",
            diagnosis_year=2022,
            medications=[],
            complications=[],
            monitoring={},
            allergies=[],
            goals=[],
            lifestyle={},
        ),
    )


def test_apply_profile_update_metadata():
    profile = _profile()
    change = {
        "field": "medications",
        "action": "add",
        "detail": {"name": "Metformin", "dose": "1000mg"},
        "confidence": 0.9,
    }
    updated = apply_profile_update("conv_m1", change, profile, source_turn_id="msg_1")

    assert len(updated.patient.medications) == 1
    assert updated.metadata.source_turn_ids == ["msg_1"]
    assert updated.metadata.confidence_scores["medications"] == 0.9

    # diske de yazıldı mı
    loaded = load_profile("conv_m1")
    assert loaded is not None
    assert loaded.patient.medications[0].name == "Metformin"


def test_find_source_turn_id_for_evidence():
    turns = [
        Turn(
            turn_id="msg_1",
            role=TurnRole.USER,
            content="Metformin kullanmaya başladım",
        )
    ]
    assert _find_source_turn_id_for_evidence("Metformin kullanmaya başladım", turns) == "msg_1"
    assert _find_source_turn_id_for_evidence("", turns) is None


def test_append_followup_note():
    append_followup_note("conv_m2", "bilgi netleştirilemedi")
    notes = load_notes("conv_m2")
    assert notes is not None
    assert any(n.category == "advice" for n in notes.items)


def test_append_followup_note_respects_eviction_limit(monkeypatch):
    monkeypatch.setitem(__import__("src.api.memory.config", fromlist=["MEMORY_CONFIG"]).MEMORY_CONFIG, "max_notes_per_conversation", 50)
    from src.api.memory.models import Note, NotesStore
    notes = NotesStore(items=[Note(note_id=f"n{i}", content="x", category="observation") for i in range(50)])
    save_notes_with_limit("conv_limit", notes)
    append_followup_note("conv_limit", "followup")
    saved = load_notes("conv_limit")
    assert saved is not None
    assert len(saved.items) == 50


def test_expire_pending_conflicts_fresh():
    store = PendingConflictsStore(
        items=[
            PendingConflict(
                conflict_id="c1",
                type="DUPLICATE_MED",
                existing={},
                proposed={},
                status=PendingConflictStatus.PENDING,
                created_at=utcnow(),
            )
        ]
    )
    save_pending_conflicts("conv_m3", store)
    expired = expire_pending_conflicts_locked("conv_m3")
    assert expired == 0


def test_expire_pending_conflicts_old():
    store = PendingConflictsStore(
        items=[
            PendingConflict(
                conflict_id="c1",
                type="DUPLICATE_MED",
                existing={},
                proposed={},
                status=PendingConflictStatus.PENDING,
                created_at=utcnow() - timedelta(days=2),
            )
        ]
    )
    save_pending_conflicts("conv_m4", store)
    expired = expire_pending_conflicts_locked("conv_m4", now=utcnow(), expiry_days=1)
    assert expired == 1

    after = load_pending_conflicts("conv_m4")
    assert after is not None
    assert after.items[0].status == PendingConflictStatus.EXPIRED_REJECTED


def test_memory_maintenance_task_no_update(monkeypatch):
    async def fake_call(profile, notes, recent_turns, needs_summary):
        return MaintenanceResult(profile_update=None, candidate_notes=[], summary=None)

    import src.api.memory.maintenance as maint

    monkeypatch.setattr(maint, "memory_maintenance_call", fake_call)

    turns = [Turn(turn_id="msg_1", role=TurnRole.USER, content="merhaba")]
    result = asyncio.run(memory_maintenance_task("conv_m5", turns, False))

    assert result["profile_updated"] is False
    assert result["notes_added"] == 0


def test_memory_maintenance_calls_critical_verification_for_high_risk(monkeypatch):
    import src.api.memory.maintenance as maint

    async def fake_call(profile, notes, recent_turns, needs_summary):
        return MaintenanceResult(
            profile_update=ProfileUpdate(
                has_update=True,
                field="medications",
                proposed_change={"action": "add", "detail": {"name": "Metformin"}},
                evidence_span="Metformin kullanıyorum",
                confidence=0.95,
            ),
            candidate_notes=[],
            summary=None,
        )

    calls = []

    async def fake_verify(detection, profile, context_turns):
        calls.append(detection.field)
        return CriticalVerifyResult(onayla=True, gerekce="net", eminlik=0.99)

    monkeypatch.setattr(maint, "memory_maintenance_call", fake_call)
    monkeypatch.setattr(maint, "critical_verification", fake_verify)
    result = asyncio.run(
        memory_maintenance_task(
            "conv_call3", [Turn(turn_id="msg_1", role=TurnRole.USER, content="Metformin kullanıyorum")]
        )
    )
    assert calls == ["medications"]
    assert result["profile_updated"] is True


def _seed_profile_and_note(conv_id: str) -> None:
    """Aktif Metformin'li profil + 'Metformin bıraktım' notu ile disk hazırlar."""
    profile = _profile()
    profile.patient.medications = [
        Medication(
            name="Metformin", dose="500mg", frequency="2x", started="",
            status=MedicationStatus.ACTIVE,
        )
    ]
    save_profile(conv_id, profile)
    save_notes(
        conv_id,
        NotesStore(items=[Note(note_id="n_stale", content="Metformin bıraktım", category="observation")]),
    )


def test_task_batch_writes_stale_notes(monkeypatch):
    """Başarılı güncelleme turunda stale tespiti diske TOPLU yazılır."""
    import src.api.memory.maintenance as maint

    _seed_profile_and_note("conv_batch")

    async def fake_call(profile, notes, recent_turns, needs_summary):
        return MaintenanceResult(
            profile_update=ProfileUpdate(
                has_update=True,
                field="medications",
                proposed_change={"action": "add", "detail": {"name": "İnsülin"}},
                evidence_span="İnsülin başladım",
                confidence=0.95,
            ),
            candidate_notes=[],
            summary=None,
        )

    async def fake_verify(detection, profile, context_turns):
        return CriticalVerifyResult(onayla=True, gerekce="net", eminlik=0.99)

    monkeypatch.setattr(maint, "memory_maintenance_call", fake_call)
    monkeypatch.setattr(maint, "critical_verification", fake_verify)

    turns = [Turn(turn_id="msg_1", role=TurnRole.USER, content="İnsülin başladım")]
    result = asyncio.run(memory_maintenance_task("conv_batch", turns, False))

    assert result["stale_notes"] == 1
    saved = load_notes("conv_batch")
    assert saved is not None
    assert saved.items[0].staleness == NoteStaleness.STALE


def test_task_skips_staleness_for_pending_field(monkeypatch):
    """Güncelleme pending'e düştüğü turda not stale İŞARETLENMEZ."""
    import src.api.memory.maintenance as maint

    _seed_profile_and_note("conv_pskip")

    async def fake_call(profile, notes, recent_turns, needs_summary):
        # Mevcut aktif ilacın tekrar eklenmesi → DUPLICATE_MED → FLAG_PENDING
        return MaintenanceResult(
            profile_update=ProfileUpdate(
                has_update=True,
                field="medications",
                proposed_change={"action": "add", "detail": {"name": "Metformin"}},
                evidence_span="Metformin kullanıyorum",
                confidence=0.95,
            ),
            candidate_notes=[],
            summary=None,
        )

    async def fake_verify(detection, profile, context_turns):
        return CriticalVerifyResult(onayla=True, gerekce="net", eminlik=0.99)

    monkeypatch.setattr(maint, "memory_maintenance_call", fake_call)
    monkeypatch.setattr(maint, "critical_verification", fake_verify)

    turns = [Turn(turn_id="msg_1", role=TurnRole.USER, content="Metformin kullanıyorum")]
    result = asyncio.run(memory_maintenance_task("conv_pskip", turns, False))

    assert result["stale_notes"] == 0
    saved = load_notes("conv_pskip")
    assert saved is not None
    assert saved.items[0].staleness == NoteStaleness.FRESH
    # Ama pending conflict gerçekten oluşmuş olmalı (skip sebebi bu).
    pending = load_pending_conflicts("conv_pskip")
    assert pending is not None
    assert any(p.status == PendingConflictStatus.PENDING for p in pending.items)
