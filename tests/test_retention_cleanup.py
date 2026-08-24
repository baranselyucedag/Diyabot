"""Retention cron testleri — dry-run güvenliği, --confirm, --days override."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.api.memory import retention_cleanup

from src.api.memory import storage
from src.api.memory.models import (
    MemoryIndex,
    PatientProfile,
    Profile,
    ProfileMetadata,
)


def _make_profile() -> Profile:
    return Profile(
        patient=PatientProfile(diabetes_type="Type 2", diagnosis_year=2020),
        metadata=ProfileMetadata(),
    )


def _make_conversation(conv_id: str, created_at: datetime) -> None:
    storage.save_profile(conv_id, _make_profile())
    storage.save_index(
        conv_id,
        MemoryIndex(turn_count=1, created_at=created_at, updated_at=created_at),
    )


def test_dry_run_reports_old_but_deletes_nothing(capsys):
    now = datetime.now(timezone.utc)
    _make_conversation("conv_old", now - timedelta(days=400))
    _make_conversation("conv_new", now - timedelta(days=10))

    candidates = retention_cleanup.find_expired_conversations(now=now, days=365)
    assert [c[0] for c in candidates] == ["conv_old"]

    rc = retention_cleanup.main([])  # dry-run (--confirm YOK)
    out = capsys.readouterr().out
    assert rc == 0
    assert "conv_old" in out
    assert "conv_new" not in out

    # Hiçbir şey silinmedi.
    assert (storage.PROFILES_DIR / "conv_old.json").exists()
    assert (storage.MEMORY_DIR / "conv_old").is_dir()
    assert (storage.PROFILES_DIR / "conv_new.json").exists()


def test_confirm_deletes_old_keeps_new(capsys):
    now = datetime.now(timezone.utc)
    _make_conversation("conv_del", now - timedelta(days=400))
    _make_conversation("conv_keep", now - timedelta(days=5))

    rc = retention_cleanup.main(["--confirm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "conv_del" in out

    assert not (storage.PROFILES_DIR / "conv_del.json").exists()
    assert not (storage.MEMORY_DIR / "conv_del").exists()
    assert (storage.PROFILES_DIR / "conv_keep.json").exists()
    assert (storage.MEMORY_DIR / "conv_keep").is_dir()


def test_days_override(capsys):
    now = datetime.now(timezone.utc)
    _make_conversation("conv_40d", now - timedelta(days=40))
    _make_conversation("conv_120d", now - timedelta(days=120))

    # days=90 → sadece 120 günlük aday.
    candidates = retention_cleanup.find_expired_conversations(now=now, days=90)
    assert [c[0] for c in candidates] == ["conv_120d"]

    # days=30 → her ikisi de aday.
    candidates2 = retention_cleanup.find_expired_conversations(now=now, days=30)
    assert {c[0] for c in candidates2} == {"conv_40d", "conv_120d"}

    # --days override dry-run üzerinden de çalışır.
    rc = retention_cleanup.main(["--days", "90"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "conv_120d" in out
    assert "conv_40d" not in out


def test_missing_index_is_skipped():
    """index.json yoksa konuşma aday OLMAZ (bilinmeyen yaş silinmez)."""
    now = datetime.now(timezone.utc)
    storage.save_profile("conv_noindex", _make_profile())  # index YOK

    candidates = retention_cleanup.find_expired_conversations(now=now, days=1)
    assert candidates == []
    assert (storage.PROFILES_DIR / "conv_noindex.json").exists()
