"""Adım 3 — Grey-band yerel sınıflandırıcı (LightGBM, embedding-only).

Nemotron LLM yerine eğitilmiş küçük model (sadece bge-m3 embedding).
Hızlı, deterministik, ücretsiz. Hata → tempered YELLOW (güvenli varsayılan).

Embedding: implicit_score'un cache'lediği bge-m3 yeniden kullanılır
(çift 2GB model yüklemesi yapılmaz).

Env:
    TRIAGE_GREY_MODEL_PATH : joblib yolu (varsayılan v10)
    TRIAGE_SKIP_GREY_MODEL : 1 → model devre dışı (tempered YELLOW)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from src.api.triage.grey_zone import TEMPERED_YELLOW_REASON, GreyZoneResult

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "classification_model"
    / "model_output_v10"
    / "grey_band_classifier.joblib"
)

_lock = threading.Lock()
_bundle: dict | None = None


def _skip_model() -> bool:
    return (os.getenv("TRIAGE_SKIP_GREY_MODEL") or "").strip() in {
        "1",
        "true",
        "True",
        "yes",
    }


def _model_path() -> Path:
    return Path(os.getenv("TRIAGE_GREY_MODEL_PATH") or DEFAULT_MODEL_PATH)


def _ensure_bundle() -> dict:
    """joblib (model + scaler + threshold) lazy load, thread-safe."""
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is not None:
            return _bundle
        import joblib

        path = _model_path()
        if not path.exists():
            raise FileNotFoundError(f"Grey model joblib bulunamadı: {path}")
        _bundle = joblib.load(path)
        return _bundle


def reset_grey_model_cache() -> None:
    """Testler için model cache temizliği."""
    global _bundle
    with _lock:
        _bundle = None


def classify_grey_zone_model(
    message: str,
    *,
    device: str | None = None,
) -> GreyZoneResult:
    """Mesajı yerel modelle GREEN/YELLOW sınıflandırır.

    REFUSE/EMERGENCY buraya gelmez (hard veto daha önce yakalar).
    """
    text = (message or "").strip()
    if _skip_model():
        return GreyZoneResult(
            level="YELLOW",
            reason=TEMPERED_YELLOW_REASON,
            timed_out=True,
            raw="TRIAGE_SKIP_GREY_MODEL",
        )
    if not text:
        return GreyZoneResult(level="GREEN", reason="grey_model: boş mesaj", raw="")

    try:
        bundle = _ensure_bundle()
        model = bundle["model"]
        scaler = bundle["scaler"]
        threshold = float(bundle.get("threshold", 0.40))

        # bge-m3 embedding — implicit_score'un cache'li modelini yeniden kullan
        from src.api.triage.implicit_score import _ensure_loaded as _implicit_loaded
        from src.retrieval.embed import encode_texts

        emb_model, _ = _implicit_loaded(device=device)
        emb = np.asarray(
            encode_texts(emb_model, [text], batch_size=1, show_progress=False),
            dtype=np.float32,
        )

        X = scaler.transform(emb)
        classes = list(model.classes_)
        proba = model.predict_proba(X)[0]
        p_yellow = float(proba[classes.index("YELLOW")])
        level = "YELLOW" if p_yellow >= threshold else "GREEN"

        if level == "YELLOW":
            reason = (
                f"Otomatik değerlendirme: yakın hekim takibi önerilir "
                f"(p={p_yellow:.2f})."
            )
        else:
            reason = f"grey_model: GREEN (p={p_yellow:.2f})"
        return GreyZoneResult(
            level=level,
            reason=reason,
            timed_out=False,
            raw=f"p_yellow={p_yellow:.4f} threshold={threshold:.3f}",
        )
    except Exception as exc:  # noqa: BLE001 — model yüklenemedi / boyut uyuşmazlığı
        logger.warning("grey_model_error: %s", exc)
        return GreyZoneResult(
            level="YELLOW",
            reason=TEMPERED_YELLOW_REASON,
            timed_out=True,
            raw=str(exc),
        )
