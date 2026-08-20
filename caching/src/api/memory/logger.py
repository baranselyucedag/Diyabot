"""Hafıza olay loglaması — logs/memory/{YYYY-MM-DD}.jsonl append.

Faz 4.2'nin minimal ama işlevsel halidir. staleness tespiti ve pending conflict
süre sonu gibi deterministik olayları yapılandırılmış JSON olarak kaydeder.

Loglar bilinçli olarak data/ dışında (logs/) tutulur: hasta verisi ve operasyonel
loglar farklı saklama/yedekleme politikalarına tabi olabilir.

Eski log dosyaları `log_retention_days` config değerine göre günde bir kez
otomatik temizlenir (rotation/retention).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import portalocker

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.timeutil import utcnow

LOG_DIR = Path("logs") / "memory"

# Günde bir kez retention temizliği yapılır; hangi gün yapıldığını takip ederiz.
_last_prune_day: str | None = None


def _maybe_prune_logs() -> None:
    """`log_retention_days` gününden eski günlük log dosyalarını siler.

    Her gün en fazla bir kez çalışır (maliyet: küçük bir glob + tarih parse).
    """
    global _last_prune_day

    today = utcnow().strftime("%Y-%m-%d")
    if _last_prune_day == today:
        return
    _last_prune_day = today

    retention_days = MEMORY_CONFIG.get("log_retention_days", 30)
    cutoff = utcnow() - timedelta(days=retention_days)

    for path in LOG_DIR.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue  # dosya adı YYYY-MM-DD değilse dokunma
        if file_date < cutoff:
            try:
                path.unlink()
            except OSError:
                pass  # silme başarısız olursa log akışını bozma


def log_event(component: str, event: str, conv_id: str, **kwargs) -> None:
    """Bir olayı günlük JSONL dosyasına ekler.

    Args:
        component: Kaynak bileşen (örn. "memory_maintenance", "staleness").
        event: Olay adı (örn. "staleness_conflict", "expired_rejected").
        conv_id: Konuşma kimliği.
        **kwargs: Ek alanlar (note_id, conflict_id, field, vb.).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _maybe_prune_logs()

    now = utcnow()
    path = LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"

    record = {
        "ts": now.isoformat(),
        "component": component,
        "event": event,
        "conv_id": conv_id,
        **kwargs,
    }
    # TODO(P3): yüksek frekansta her olayda open/close yerine process başına
    # güvenli bir writer/buffer düşünülebilir; MVP'de append maliyeti kabul edilir.
    # Kilit: storage.py ile aynı desen (LOCK_EX | LOCK_NB + timeout) — eşzamanlı
    # process'lerde log satırı karışmasını önler. Kilit alınamazsa log düşer ama
    # çağıran (ör. storage.read_json hata yolu) asla log yüzünden patlamaz.
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with portalocker.Lock(
            lock_path, timeout=10, flags=portalocker.LOCK_EX | portalocker.LOCK_NB
        ):
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (portalocker.exceptions.AlreadyLocked, OSError):
        pass  # log best-effort'tur; kilit/disk hatası ana akışı bozmamalı
