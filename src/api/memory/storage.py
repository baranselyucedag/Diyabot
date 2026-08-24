"""
Depolama katmanı — Pydantic modellerini diske güvenli şekilde yazar/okur.

Güvenlik garantileri:
- Atomic write: temp dosya + os.replace, yarım yazma/çökme durumunda eski dosya bozulmaz.
- Dosya kilidi (portalocker): aynı dosyaya eşzamanlı process erişiminde çakışma önlenir.
- Conversation lock (asyncio.Lock, weakref ile): aynı process içinde aynı konuşma için
  eşzamanlı async task'ların birbirini ezmesini önler. Kullanılmayan lock'lar otomatik
  olarak bellekten temizlenir (weakref sayesinde, elle temizlik gerekmez).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import weakref
from datetime import datetime
from pathlib import Path
from typing import Optional, Type, TypeVar

import portalocker
from pydantic import BaseModel

from src.api.memory.models import (
    MemoryIndex,
    NotesStore,
    PendingConflictsStore,
    Profile,
    Turn,
)
from src.api.memory.encryption import decrypt_bytes, encrypt_bytes
from src.api.memory.logger import log_event

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Dizin sabitleri
# ---------------------------------------------------------------------------
# NOT: Dizinler import anında OLUŞTURULMAZ (side effect yok). İlk yazma
# sırasında `path.parent.mkdir(parents=True, exist_ok=True)` ile tembel olarak
# oluşturulur. Böylece testler `monkeypatch` ile bu globalleri tmp dizine
# çevirebilir ve çalışma dizininde istenmeyen data/ ve logs/ klasörleri açılmaz.
#
# Dizinler CWD'ye DEĞİL proje köküne sabitlenir (uygulama nereden çalıştırılırsa
# çalıştırılsın aynı yer kullanılır). CHATBOT_DATA_DIR env değişkeniyle override
# edilebilir; testler yine module-global `monkeypatch.setattr` ile değiştirir.

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = Path(os.environ.get("CHATBOT_DATA_DIR") or (_PROJECT_ROOT / "data"))
PROFILES_DIR = DATA_DIR / "profiles"
MEMORY_DIR = DATA_DIR / "memory"


# ---------------------------------------------------------------------------
# conv_id doğrulama (path traversal koruması)
# ---------------------------------------------------------------------------
# conv_id doğrudan dosya yolu kurmakta kullanılır (PROFILES_DIR / f"{conv_id}.json").
# Kötü niyetli/kaza eseri gelen bir conv_id ("../../etc/passwd" gibi) path traversal
# saldırısına yol açabilir. Bu yüzden her disk işleminin girişinde katı bir
# beyaz liste doğrulaması yapılır.

_CONV_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_conv_id(conv_id: str) -> str:
    """conv_id'yi güvenli karakter kümesine karşı doğrular.

    Geçerli: alfanumerik + `_` ve `-`, ilk karakter alfanumerik, en fazla 64
    karakter. Geçersizse ValueError fırlatır (sessizce geçmez).
    """
    if not isinstance(conv_id, str) or not _CONV_ID_RE.match(conv_id):
        raise ValueError(
            f"Geçersiz conv_id: {conv_id!r}. Yalnızca [A-Za-z0-9_-] karakterleri "
            "kullanılabilir (ilk karakter alfanumerik, en fazla 64 karakter)."
        )
    return conv_id


# ---------------------------------------------------------------------------
# Conversation-level asyncio kilitleri (process içi eşzamanlılık koruması)
# ---------------------------------------------------------------------------
#
# weakref.WeakValueDictionary kullanıyoruz: bir conversation_id için hiçbir
# yerde artık aktif referans kalmadığında (o konuşmayla ilgili tüm task'lar
# bitip kilit nesnesi serbest kaldığında), kayıt otomatik olarak sözlükten
# silinir. Elle "temizlik" fonksiyonu çağırmaya veya periyodik bir cron job'a
# ihtiyaç yoktur — memory leak riski yapısal olarak ortadan kalkar.
#
# Not: aktif kullanımdaki bir task, kilidi bir yerel değişkende tuttuğu sürece
# (async with get_conversation_lock(conv_id): ... bloğu içindeyken) referans
# canlı kalır, bu yüzden kilit task bitmeden silinmez. `get_conversation_lock`
# dönüş değerini çağırana güçlü referans olarak verdiği için ayrı bir
# "anchor" dict tutmaya gerek yoktur (asyncio.Lock weakref uyumludur).

_conversation_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
    weakref.WeakValueDictionary()
)


def get_conversation_lock(conv_id: str) -> asyncio.Lock:
    """Verilen conversation_id için asyncio.Lock döndürür (yoksa oluşturur).

    Kilit, çağıran taraf onu `async with` bloğuyla kullandığı sürece canlı
    kalır. Kimse kullanmıyorsa weakref sayesinde otomatik temizlenir.
    """
    conv_id = validate_conv_id(conv_id)
    lock = _conversation_locks.get(conv_id)
    if lock is None:
        lock = asyncio.Lock()
        _conversation_locks[conv_id] = lock
    return lock


def active_lock_count() -> int:
    """Şu an bellekte tutulan aktif conversation lock sayısı (izleme/metrik için)."""
    return len(_conversation_locks)


# ---------------------------------------------------------------------------
# Düşük seviye dosya işlemleri (atomic write, kilitli okuma)
# ---------------------------------------------------------------------------


class _DateTimeEncoder(json.JSONEncoder):
    """datetime nesnelerini ISO 8601 string'e çeviren JSON encoder."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _lock_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def write_json_atomic(path: Path, data: BaseModel | dict) -> None:
    """Bir Pydantic modelini (veya dict'i) JSON olarak atomik şekilde yazar.

    Süreç:
      1. Aynı dizinde geçici bir dosyaya yazılır (asıl dosyaya DOKUNULMAZ).
      2. os.replace ile geçici dosya, asıl dosyanın yerine atomik olarak geçirilir.
         (os.replace işletim sistemi seviyesinde bölünemez bir işlemdir; yazma
         sırasında çökme olsa bile asıl dosya ya eski haliyle sağlam kalır ya
         da tamamen yeni haliyle günceldir — yarım/bozuk bir dosya oluşmaz.)
      3. Adım 2, portalocker ile korunur: aynı dosyaya eşzamanlı yazmaya
         çalışan başka bir process, bu işlem bitene kadar bekler.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = data.model_dump() if isinstance(data, BaseModel) else data
    raw_bytes = json.dumps(
        payload, ensure_ascii=False, indent=2, cls=_DateTimeEncoder
    ).encode("utf-8")
    final_bytes = encrypt_bytes(raw_bytes)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(final_bytes)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    with portalocker.Lock(_lock_path_for(path), timeout=10, flags=portalocker.LOCK_EX | portalocker.LOCK_NB):
        os.replace(tmp_path, path)


def read_json(path: Path, model: Type[T], repair_fn=None) -> Optional[T]:
    """JSON dosyasını okuyup verilen Pydantic modeline doğrular.

    Dosya yoksa None döner. Dosya bozuksa (geçersiz JSON) veya modele
    uymuyorsa (eksik/yanlış tipte alan) hata fırlatmak yerine None döner ve
    durum loglanır — tek bir bozuk kayıt tüm sistemi çökertmesin diye.

    repair_fn (opsiyonel): model doğrulaması ValueError ile patlarsa ham dict'i
    tamir edip tekrar denemek için çağrılır (örn. _repair_notes_store). Şema-özel
    tamir mantığı bu generic fonksiyonun DIŞINDA tutulur; ileride başka bir model
    için tamir gerekirse read_json'a dokunmadan yeni bir repair_fn yazılır.
    """
    if not path.exists():
        return None

    try:
        with portalocker.Lock(
            _lock_path_for(path), timeout=10, flags=portalocker.LOCK_SH | portalocker.LOCK_NB
        ):
            with open(path, "rb") as f:
                raw = json.loads(decrypt_bytes(f.read()).decode("utf-8"))
        try:
            return model.model_validate(raw)
        except ValueError:
            if repair_fn is None or not isinstance(raw, dict):
                raise
            return model.model_validate(repair_fn(raw))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _log_warning(f"read_json_failed path={path} error={exc}")
        return None


def _repair_notes_store(raw: dict) -> dict:
    """Bilinmeyen Note.category değerlerini "observation"a düşürür (geriye-uyum).

    Note.category Literal'a çevrildiği için eski disk kayıtlarındaki geçersiz
    kategori string'i tüm store'un okunamaz hale gelmesine yol açardı; kayıt
    korunur, kategori güvenli varsayılana çekilir ve durum loglanır.
    """
    valid_categories = {"symptom", "observation", "plan", "measurement", "advice"}
    repaired = dict(raw)
    repaired_items = []
    for item in raw.get("items", []):
        item = dict(item)
        if item.get("category") not in valid_categories:
            _log_warning(
                f"invalid_note_category_fallback category={item.get('category')!r}"
            )
            item["category"] = "observation"
        repaired_items.append(item)
    repaired["items"] = repaired_items
    return repaired


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Bir Pydantic modelini JSONL dosyasına tek satır olarak ekler (append-only).

    Satır ayracı (``\\n``) şifrelemenin DIŞINDADIR: her satır tek bir token/JSON
    olur ve dosya fiziksel olarak ``\\n`` ile bölünür. ``load_turns`` satırları
    bu ayraçtan böler — ayraç şifreli yükün İÇİNDE olsaydı art arda eklenen
    token'lar tek satırda birleşir ve okuma yalnızca İLK turu geri verirdi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = encrypt_bytes(record.model_dump_json().encode("utf-8")) + b"\n"

    with portalocker.Lock(_lock_path_for(path), timeout=10, flags=portalocker.LOCK_EX | portalocker.LOCK_NB):
        with open(path, "ab") as f:
            f.write(line)


def _log_warning(message: str) -> None:
    log_event("storage", "warning", "", message=message)


# ---------------------------------------------------------------------------
# Profil
# ---------------------------------------------------------------------------


def _profile_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return PROFILES_DIR / f"{conv_id}.json"


def load_profile(conv_id: str) -> Optional[Profile]:
    """Hasta profilini diskten okur. Yoksa/bozuksa None döner."""
    return read_json(_profile_path(conv_id), Profile)


def save_profile(conv_id: str, profile: Profile) -> None:
    """Hasta profilini diske atomik olarak yazar."""
    write_json_atomic(_profile_path(conv_id), profile)


# ---------------------------------------------------------------------------
# Notlar
# ---------------------------------------------------------------------------


def _notes_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return MEMORY_DIR / conv_id / "notes.json"


def load_notes(conv_id: str) -> Optional[NotesStore]:
    return read_json(_notes_path(conv_id), NotesStore, repair_fn=_repair_notes_store)


def save_notes(conv_id: str, notes: NotesStore) -> None:
    write_json_atomic(_notes_path(conv_id), notes)


# ---------------------------------------------------------------------------
# Bekleyen çakışmalar
# ---------------------------------------------------------------------------


def _pending_conflicts_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return MEMORY_DIR / conv_id / "pending_conflicts.json"


def load_pending_conflicts(conv_id: str) -> Optional[PendingConflictsStore]:
    return read_json(_pending_conflicts_path(conv_id), PendingConflictsStore)


def save_pending_conflicts(conv_id: str, conflicts: PendingConflictsStore) -> None:
    write_json_atomic(_pending_conflicts_path(conv_id), conflicts)


# ---------------------------------------------------------------------------
# İndeks (turn_count vb. sayaçlar)
# ---------------------------------------------------------------------------


def _index_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return MEMORY_DIR / conv_id / "index.json"


def load_index(conv_id: str) -> Optional[MemoryIndex]:
    return read_json(_index_path(conv_id), MemoryIndex)


def save_index(conv_id: str, index: MemoryIndex) -> None:
    write_json_atomic(_index_path(conv_id), index)


# ---------------------------------------------------------------------------
# Konuşma turları (append-only JSONL)
# ---------------------------------------------------------------------------


def _turns_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return MEMORY_DIR / conv_id / "turns.jsonl"


def append_turn(conv_id: str, turn: Turn) -> None:
    """Yeni bir turu turns.jsonl'e ekler. Var olan satırlara asla dokunulmaz."""
    append_jsonl(_turns_path(conv_id), turn)


def load_turns(conv_id: str, limit: Optional[int] = None) -> list[Turn]:
    """Turları KRONOLOJİK sırada (en eski -> en yeni) döndürür.

    Bu sıra doğrudan prompt'a basılabilir; ayrıca ters çevirmeye gerek yoktur.
    limit verilirse, kronolojik sıra korunarak SON `limit` tur döndürülür.
    limit 0 veya negatifse tüm turlar döndürülür.
    """
    path = _turns_path(conv_id)
    if not path.exists():
        return []

    turns: list[Turn] = []
    with portalocker.Lock(_lock_path_for(path), timeout=10, flags=portalocker.LOCK_SH | portalocker.LOCK_NB):
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = decrypt_bytes(line).decode("utf-8")
                    turns.append(Turn.model_validate_json(raw))
                except Exception as exc:  # bozuk tek satır tüm okumayı düşürmesin
                    _log_warning(f"skipping_corrupt_turn_line conv_id={conv_id} error={exc}")

    if limit is not None and limit > 0:
        return turns[-limit:]
    return turns


# ---------------------------------------------------------------------------
# Özet — bilinçli olarak DÜZ METİN (JSON değil), doğrudan prompt'a eklenebilsin diye
# ---------------------------------------------------------------------------


def _summary_path(conv_id: str) -> Path:
    conv_id = validate_conv_id(conv_id)
    return MEMORY_DIR / conv_id / "summary.txt"


def save_summary(conv_id: str, summary: str) -> None:
    """Özeti düz metin olarak atomik şekilde yazar (JSON sarmalama yok)."""
    path = _summary_path(conv_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        tmp.write(encrypt_bytes(summary.encode("utf-8")))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    # LOCK_NB bilinçli: blocking modda portalocker'ın timeout'u etkisizdir
    # (testlerde "timeout has no effect in blocking mode" uyarısı). Non-blocking
    # edinim + timeout sayesinde kilit kısa sürede alınamazsa istisna fırlar ve
    # çöken bir process'in bıraktığı kilit yüzünden sonsuza dek beklenmez.
    with portalocker.Lock(_lock_path_for(path), timeout=10, flags=portalocker.LOCK_EX | portalocker.LOCK_NB):
        os.replace(tmp_path, path)


def load_summary(conv_id: str) -> Optional[str]:
    """Özeti düz metin olarak okur. Dosya yoksa None döner."""
    path = _summary_path(conv_id)
    if not path.exists():
        return None

    try:
        with portalocker.Lock(_lock_path_for(path), timeout=10, flags=portalocker.LOCK_SH | portalocker.LOCK_NB):
            with open(path, "rb") as f:
                return decrypt_bytes(f.read()).decode("utf-8").strip()
    except OSError as exc:
        _log_warning(f"read_summary_failed conv_id={conv_id} error={exc}")
        return None
