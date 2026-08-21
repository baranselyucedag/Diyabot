"""At-rest şifreleme — Fernet (symmetric) ile hasta verisi koruması.

Davranış:
- ``encryption_enabled=False`` (varsayılan): :func:`encrypt_bytes` /
  :func:`decrypt_bytes` veriyi OLDUĞU GİBİ geçirir (şifreleme yok). Bu,
  mevcut disk verisi ve testlerle %100 geriye dönük uyumludur.
- ``encryption_enabled=True``: ``master_key_env_var`` (varsayılan
  ``MEMORY_MASTER_KEY``) ortam değişkeni ZORUNLU olur; yoksa veya geçersizse
  :class:`MasterKeyError` fırlatılır — sistem sessizce şifresiz çalışmaya
  düşmez.

Not: Karışık mod (bir dosya şifreli, diğeri değil) DESTEKLENMEZ. Şifreleme
açıldığı andan itibaren hasta verisi içeren tüm dosyalar şifreli yazılır;
eski düz metin dosyalar Fernet ile çözülemez. Bu bilinçli bir karardır —
açmadan önce veri taşınmalı veya temizlenmelidir.
"""

from __future__ import annotations

import os
from typing import Optional

from cryptography.fernet import Fernet

from src.api.memory.config import MEMORY_CONFIG


class MasterKeyError(RuntimeError):
    """encryption_enabled=True iken anahtar eksik/geçersiz olduğunda fırlatılır."""


def get_fernet() -> Optional[Fernet]:
    """Aktif Fernet nesnesini döndürür; şifreleme kapalıysa None.

    Şifreleme açıksa anahtarı ``master_key_env_var`` değişkeninden okur;
    eksik/geçersizse :class:`MasterKeyError` fırlatır.
    """
    if not MEMORY_CONFIG.get("encryption_enabled"):
        return None
    return Fernet(_load_master_key())


def _load_master_key() -> bytes:
    env_var = MEMORY_CONFIG.get("master_key_env_var", "MEMORY_MASTER_KEY")
    raw = (os.getenv(env_var) or "").strip()
    if not raw:
        raise MasterKeyError(
            f"encryption_enabled=True fakat {env_var} tanımlı değil. "
            "Şu komutla anahtar üretip ortam değişkenine koyun: "
            "python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    key_bytes = raw.encode("ascii")
    try:
        Fernet(key_bytes)  # base64 + 32 byte formatını doğrular
    except Exception as exc:  # noqa: BLE001 — ValueError/binascii hatası
        raise MasterKeyError(
            f"{env_var} geçerli bir Fernet anahtarı değil "
            f"(32 byte base64 bekleniyor): {exc}"
        ) from exc
    return key_bytes


def encrypt_bytes(data: bytes) -> bytes:
    """Baytları şifreler; şifreleme kapalıysa aynen döndürür."""
    fernet = get_fernet()
    if fernet is None:
        return data
    return fernet.encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Baytları çözer; şifreleme kapalıysa aynen döndürür."""
    fernet = get_fernet()
    if fernet is None:
        return data
    return fernet.decrypt(data)
