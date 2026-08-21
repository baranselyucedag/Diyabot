"""At-rest şifreleme testleri (Fernet)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from src.api.memory import encryption, storage
from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.models import (
    PatientProfile,
    Profile,
    ProfileMetadata,
    Turn,
    TurnRole,
)


def _make_profile() -> Profile:
    return Profile(
        patient=PatientProfile(diabetes_type="Type 2", diagnosis_year=2022),
        metadata=ProfileMetadata(),
    )


@pytest.fixture
def valid_key() -> str:
    return Fernet.generate_key().decode("ascii")


def test_encryption_enabled_roundtrip(monkeypatch, valid_key):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", True)
    monkeypatch.setenv(MEMORY_CONFIG["master_key_env_var"], valid_key)

    storage.save_profile("conv_enc1", _make_profile())
    loaded = storage.load_profile("conv_enc1")

    assert loaded is not None
    assert loaded.patient.diabetes_type == "Type 2"


def test_encryption_enabled_missing_key_raises(monkeypatch):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", True)
    monkeypatch.delenv("MEMORY_MASTER_KEY", raising=False)

    with pytest.raises(encryption.MasterKeyError):
        encryption.get_fernet()


def test_encrypted_file_is_not_plaintext(monkeypatch, valid_key):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", True)
    monkeypatch.setenv(MEMORY_CONFIG["master_key_env_var"], valid_key)

    storage.save_profile("conv_enc2", _make_profile())

    raw = (storage.PROFILES_DIR / "conv_enc2.json").read_bytes()
    # Düz metin JSON değil (Fernet tokenı '{' ile başlamaz).
    assert not raw.lstrip().startswith(b"{")
    # Hasta verisi düz metin olarak SIZMASIN.
    assert b"diabetes_type" not in raw
    assert b"Type 2" not in raw


def test_encryption_disabled_stays_plaintext(monkeypatch):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", False)

    storage.save_profile("conv_plain", _make_profile())

    raw = (storage.PROFILES_DIR / "conv_plain.json").read_bytes()
    assert raw.lstrip().startswith(b"{")
    assert b"diabetes_type" in raw


def test_encrypt_decrypt_bytes_identity_when_disabled(monkeypatch):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", False)

    data = b'{"a": 1}'
    assert encryption.encrypt_bytes(data) == data
    assert encryption.decrypt_bytes(data) == data


def test_summary_encrypted_when_enabled(monkeypatch, valid_key):
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", True)
    monkeypatch.setenv(MEMORY_CONFIG["master_key_env_var"], valid_key)

    storage.save_summary("conv_enc3", "hasta özeti gizli")
    raw_summary = (storage.MEMORY_DIR / "conv_enc3" / "summary.txt").read_bytes()
    assert b"gizli" not in raw_summary

    assert storage.load_summary("conv_enc3") == "hasta özeti gizli"


def test_turns_roundtrip_encrypted_when_enabled(monkeypatch, valid_key):
    """turns.jsonl YAZMA (append_turn) ve OKUMA (load_turns) aynı anahtarla tutarlı mı?"""
    monkeypatch.setitem(MEMORY_CONFIG, "encryption_enabled", True)
    monkeypatch.setenv(MEMORY_CONFIG["master_key_env_var"], valid_key)

    storage.append_turn(
        "conv_turn",
        Turn(turn_id="msg_1", role=TurnRole.USER, content="şekerim 250", triage="YELLOW"),
    )
    storage.append_turn(
        "conv_turn",
        Turn(turn_id="msg_2", role=TurnRole.ASSISTANT, content="hekiminize danışın"),
    )

    # Şifreli: düz metin sızmamalı.
    raw = (storage.MEMORY_DIR / "conv_turn" / "turns.jsonl").read_bytes()
    assert "şekerim".encode("utf-8") not in raw
    assert "msg_1".encode("utf-8") not in raw

    loaded = storage.load_turns("conv_turn")
    assert [t.turn_id for t in loaded] == ["msg_1", "msg_2"]
    assert loaded[0].content == "şekerim 250"
    assert loaded[0].triage == "YELLOW"
    assert loaded[1].role == TurnRole.ASSISTANT
