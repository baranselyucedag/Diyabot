"""Depolama katmanı testleri — atomic write, conv_id doğrulama, lock."""

from __future__ import annotations

import pytest

from src.api.memory import storage
from src.api.memory.models import (
    Medication,
    MedicationStatus,
    NotesStore,
    PatientProfile,
    Profile,
    ProfileMetadata,
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


def test_save_load_profile_roundtrip():
    storage.save_profile("conv_a1", _make_profile())
    loaded = storage.load_profile("conv_a1")
    assert loaded is not None
    assert loaded.patient.diabetes_type == "Type 2"
    assert loaded.patient.medications[0].name == "Metformin"


def test_load_missing_returns_none():
    assert storage.load_profile("conv_nonexist") is None


def test_read_corrupt_returns_none(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert storage.read_json(bad, NotesStore) is None


def test_validate_conv_id_ok():
    assert storage.validate_conv_id("conv_123") == "conv_123"
    assert storage.validate_conv_id("abc-DEF_9") == "abc-DEF_9"


@pytest.mark.parametrize(
    "bad",
    ["../etc", "a/b", "", "a b", "a" * 65, "-abc", "a.", "..", "a\\b"],
)
def test_validate_conv_id_rejects(bad):
    with pytest.raises(ValueError):
        storage.validate_conv_id(bad)


def test_path_traversal_blocked():
    with pytest.raises(ValueError):
        storage.save_profile("../etc/passwd", _make_profile())


def test_conversation_lock_same_id():
    l1 = storage.get_conversation_lock("conv_lock")
    l2 = storage.get_conversation_lock("conv_lock")
    assert l1 is l2


def test_active_lock_count():
    lock = storage.get_conversation_lock("conv_count")
    assert storage.active_lock_count() >= 1
    assert lock is not None
