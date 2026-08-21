"""Pytest yapılandırması — hafıza (memory) modülü için ortak fixtures.

- Proje kökünü sys.path'e ekler (böylece `import src.api.memory` çalışır).
- Testlerin gerçek `data/` ve `logs/` dizinlerine yazmasını engellemek için
  storage ve logger'ın dizin sabitlerini her testte tmp dizine çevirir.
- Conversation lock'ları her test öncesi/sonrası temizler (event loop çapraz
  bulaşmasını önler).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path, monkeypatch):
    """storage/logger dizin sabitlerini her test için tmp dizine çevirir."""
    from src.api.memory import logger, storage

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(storage, "PROFILES_DIR", tmp_path / "data" / "profiles")
    monkeypatch.setattr(storage, "MEMORY_DIR", tmp_path / "data" / "memory")
    monkeypatch.setattr(logger, "LOG_DIR", tmp_path / "logs" / "memory")
    yield tmp_path


@pytest.fixture(autouse=True)
def _clear_conversation_locks():
    """Conversation lock'larını her test öncesi/sonrası temizler."""
    from src.api.memory import storage

    storage._conversation_locks.clear()
    yield
    storage._conversation_locks.clear()
