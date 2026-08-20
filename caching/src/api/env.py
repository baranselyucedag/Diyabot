"""Ortak .env yükleme — key frontend/.env dosyasından okunur.

Ana projedeki src/api/env.py'nin caching çalışma alanı için birebir kopyasıdır;
böylece `from src.api.env import load_project_env` import'u bu çalışma alanında
da kendine yeterli şekilde çalışır (harici src/api paketine bağımlı olmadan).
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ENV = _PROJECT_ROOT / "frontend" / ".env"
ROOT_ENV = _PROJECT_ROOT / ".env"


def load_project_env() -> Path | None:
    """Önce frontend/.env, yoksa kök .env yükler. Yüklenen yolu döner."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    if FRONTEND_ENV.is_file():
        load_dotenv(FRONTEND_ENV, override=True)
        return FRONTEND_ENV
    if ROOT_ENV.is_file():
        load_dotenv(ROOT_ENV, override=True)
        return ROOT_ENV
    return None
