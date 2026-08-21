"""Modeller için testler — Pydantic şema doğrulaması ve timezone-aware datetime."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.memory.models import (
    CandidateNote,
    MaintenanceResult,
    Medication,
    MedicationStatus,
    Note,
    PatientProfile,
    Profile,
    ProfileMetadata,
    ProfileUpdate,
)


def _make_profile() -> Profile:
    return Profile(
        version=1,
        patient=PatientProfile(
            diabetes_type="Type 2",
            diagnosis_year=2022,
            medications=[
                Medication(
                    name="Metformin",
                    dose="1000mg",
                    frequency="2x",
                    started="2022-03",
                    status=MedicationStatus.ACTIVE,
                )
            ],
            complications=[],
            monitoring={},
            allergies=[],
            goals=[],
            lifestyle={},
        ),
        metadata=ProfileMetadata(source_turn_ids=[], confidence_scores={}),
    )


def test_profile_schema_roundtrip():
    p = _make_profile()
    assert p.patient.medications[0].name == "Metformin"
    assert p.patient.medications[0].status == MedicationStatus.ACTIVE


def test_profile_update_schema():
    pu = ProfileUpdate(
        has_update=True,
        field="medications",
        proposed_change={"action": "add", "detail": {"name": "Metformin"}},
        evidence_span="Metformin kullanıyorum",
        confidence=0.9,
    )
    assert pu.proposed_change["action"] == "add"
    assert pu.has_update is True


def test_candidate_note_category_literal_rejects_invalid():
    CandidateNote(content="x", category="observation", source_turn_id="msg_1")
    with pytest.raises(ValidationError):
        CandidateNote(content="x", category="medication", source_turn_id="msg_1")


def test_maintenance_result_defaults():
    r = MaintenanceResult()
    assert r.profile_update is None
    assert r.candidate_notes == []
    assert r.summary is None


def test_note_created_at_timezone_aware():
    n = Note(note_id="n1", content="x", category="observation")
    assert n.created_at.tzinfo is not None
