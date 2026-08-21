"""Deterministik kurallar testleri — grounding, staleness, çakışma, stopword."""

from __future__ import annotations

from datetime import timedelta

from src.api.memory.deterministic import (
    check_conflicts,
    detect_stale_notes,
    triage_classify,
    verify_grounding_deterministic,
    word_set,
)
from src.api.memory.models import (
    CandidateNote,
    Medication,
    MedicationStatus,
    Note,
    PatientProfile,
    PendingConflict,
    PendingConflictsStore,
    Profile,
    Turn,
    TurnRole,
)
from src.api.memory.storage import save_pending_conflicts
from src.api.memory.timeutil import utcnow


def _turn(turn_id="msg_1", content="Metformin 1000mg alıyorum") -> Turn:
    return Turn(turn_id=turn_id, role=TurnRole.USER, content=content)


def _profile(meds: list[Medication] | None = None) -> Profile:
    return Profile(
        version=1,
        patient=PatientProfile(
            diabetes_type="Type 2",
            diagnosis_year=2022,
            medications=meds or [],
            complications=[],
            monitoring={},
            allergies=[],
            goals=[],
            lifestyle={},
        ),
    )


def test_word_set_filters_stopwords():
    ws = word_set("Metformin ve ile bir ilaç")
    assert "metformin" in ws
    # normalize(): ı->i, ç->c dönüşümü yapar
    assert "ilac" in ws
    assert "ve" not in ws
    assert "ile" not in ws
    assert "bir" not in ws


def test_grounding_no_source_id():
    note = CandidateNote(content="Metformin kullanıyorum", category="observation", source_turn_id=None)
    r = verify_grounding_deterministic(note, [_turn()])
    assert r["grounded"] is False
    assert r["reason"] == "no_source_id"


def test_grounding_source_not_in_context():
    note = CandidateNote(content="Metformin", category="observation", source_turn_id="msg_999")
    r = verify_grounding_deterministic(note, [_turn()])
    assert r["grounded"] is False
    assert r["reason"] == "source_id_not_in_context"


def test_grounding_overlap_ok():
    note = CandidateNote(content="Metformin 1000mg alıyorum", category="observation", source_turn_id="msg_1")
    r = verify_grounding_deterministic(note, [_turn(content="Metformin 1000mg alıyorum")])
    assert r["grounded"] is True
    assert r["reason"] == "ok"


def test_grounding_low_overlap():
    note = CandidateNote(content="İnsülin dozu değişti", category="observation", source_turn_id="msg_1")
    r = verify_grounding_deterministic(note, [_turn(content="Bugün hava çok güzel")])
    assert r["grounded"] is False
    assert r["reason"] == "low_overlap"


def test_check_conflicts_duplicate_med():
    profile = _profile(
        meds=[
            Medication(
                name="Metformin",
                dose="1000mg",
                frequency="2x",
                started="",
                status=MedicationStatus.ACTIVE,
            )
        ]
    )
    proposed = {"field": "medications", "action": "add", "detail": {"name": "Metformin"}}
    r = check_conflicts(proposed, profile)
    assert r["action"] == "FLAG_PENDING"
    assert r["conflict_type"] == "DUPLICATE_MED"


def test_check_conflicts_duplicate_allergy():
    profile = _profile()
    profile.patient.allergies = ["Penisilin"]
    r = check_conflicts(
        {"field": "allergies", "action": "add", "detail": {"value": "PENİSİLİN"}},
        profile,
    )
    assert r["has_conflict"] is True
    assert r["conflict_type"] == "DUPLICATE_ALLERGY"


def test_triage_classify():
    assert triage_classify("medications")["is_high_risk"] is True
    assert triage_classify("goals")["is_high_risk"] is False
    assert triage_classify("bilinmeyen_alan")["is_high_risk"] is True


def test_staleness_status_conflict():
    profile = _profile(
        meds=[
            Medication(
                name="Metformin",
                dose="",
                frequency="",
                started="",
                status=MedicationStatus.ACTIVE,
            )
        ]
    )
    note = Note(note_id="n1", content="Metformin bıraktım", category="observation")
    stale = detect_stale_notes(profile, [note], changed_field="medications", conv_id="conv_s1")
    assert any(i["note_id"] == "n1" for i in stale)


def test_staleness_turkish_casefold_matches_uppercase_keyword():
    profile = _profile(
        meds=[Medication(name="Metformin", dose="", frequency="", started="")]
    )
    note = Note(note_id="n_case", content="METFORMİN BIRAKTIM", category="observation")
    stale = detect_stale_notes(
        profile, [note], changed_field="medications", conv_id="conv_case"
    )
    assert any(i["note_id"] == "n_case" for i in stale)


def test_staleness_deduplicates_note():
    """Hem keyword (goal_replaced) hem yaş (OLD_PLAN) kuralına takılan not tek kayıt döner."""
    note = Note(
        note_id="n_dupe",
        content="hedef değiştirdim",
        category="plan",
        created_at=utcnow() - timedelta(days=40),
    )
    stale = detect_stale_notes(
        _profile(), [note], changed_field="goals", conv_id="conv_dupe"
    )
    assert len(stale) == 1
    assert stale[0]["type"] == "GOAL_REPLACED"  # ilk eşleşen sebep tutulur


def test_staleness_old_plan():
    note = Note(
        note_id="n2",
        content="haftada 3 yürüyüş",
        category="plan",
        created_at=utcnow() - timedelta(days=40),
    )
    stale = detect_stale_notes(_profile(), [note], changed_field=None, conv_id="conv_s2")
    assert any(i["note_id"] == "n2" and i["type"] == "OLD_PLAN" for i in stale)


def test_staleness_skips_keyword_check_when_field_pending():
    """changed_field için PENDING çakışma varsa keyword-bazlı stale işaretlenmez."""
    profile = _profile(
        meds=[Medication(name="Metformin", dose="", frequency="", started="")]
    )
    note = Note(note_id="n_pend", content="Metformin bıraktım", category="observation")
    save_pending_conflicts(
        "conv_pend",
        PendingConflictsStore(
            items=[
                PendingConflict(
                    conflict_id="c1",
                    type="UNVERIFIED",
                    existing={"field": "medications"},
                    proposed={"field": "medications", "proposed_change": {}},
                )
            ]
        ),
    )
    stale = detect_stale_notes(
        profile, [note], changed_field="medications", conv_id="conv_pend"
    )
    assert stale == []

    # Yaş kontrolü pending'den etkilenmez (profil durumundan bağımsız).
    old_plan = Note(
        note_id="n_old",
        content="haftada 3 yürüyüş",
        category="plan",
        created_at=utcnow() - timedelta(days=40),
    )
    stale2 = detect_stale_notes(
        profile, [old_plan], changed_field="medications", conv_id="conv_pend"
    )
    assert any(i["note_id"] == "n_old" for i in stale2)
