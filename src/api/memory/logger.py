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
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import portalocker

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.timeutil import utcnow

LOG_DIR = Path("logs") / "memory"

# Günde bir kez retention temizliği yapılır; hangi gün yapıldığını takip ederiz.
_last_prune_day: str | None = None

# structlog opsiyonel — yoksa stdlib logging'e düşer.
try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


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


def get_logger(name: str = "memory") -> Any:
    """Structlog logger döndürür; structlog yoksa stdlib logging'e düşer.

    Structlog varsa: JSON renderer + timestamp + level + caller info.
    Yoksa: stdlib logging.Logger (basicConfig uygulanmamışsa no-op).
    """
    if _STRUCTLOG_AVAILABLE and structlog is not None:
        # structlog configure edilmemişse basit yapılandırma
        if not structlog.is_configured():
            structlog.configure(
                processors=[
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso", utc=True),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                context_class=dict,
                logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
                cache_logger_on_first_use=True,
            )
        return structlog.get_logger(name)

    # Fallback: stdlib logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
