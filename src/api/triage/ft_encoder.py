"""Adım 3 — FT encoder (bge-m3) örtük acil-dil skoru.

Hard-veto'nun zaten yakaladığı ifadeler (112, bayılma, bilinç…) anchor DEĞİL.
Odak: regex kaçıran örtük / zayıf semptom dili.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

# Örtük anchor'lar — hard EMERGENCY pattern'leriyle örtüşmesin
_IMPLICIT_ANCHORS: list[str] = [
    "her şey bulanık, biraz kötü hissediyorum",
    "çok halsizim, ayakta duramıyorum gibi",
    "şekerim garip gidiyor, içim rahat değil",
    "kusasım geliyor, midem altüst",
    "ellerim titriyor, terliyorum, garip hissediyorum",
    "ağzım kurudu, sürekli su içiyorum",
    "başım dönüyor, dengemi kaybediyorum gibi",
    "çok susadım, sık idrara çıkıyorum, keyifsizim",
    "şekerim bir türlü düzelmiyor, endişeliyim",
    "gece terledim, sabah bitkin uyandım",
    "gözlerim kararıyor, oturmam gerekiyor",
    "midem bulanıyor, yemek yiyemiyorum, bitkinim",
    "nefesim daralıyor gibi, huzursuzum",
    "vücudum uyuşuyor, güçsüzüm",
    "şekerim yüksek sanırım, çok kötü hissediyorum",
]

_lock = threading.Lock()
_model: Any = None
_anchor_matrix: np.ndarray | None = None


def _skip_ft() -> bool:
    """Smoke/test: TRIAGE_SKIP_FT=1 → skor 0 (model yükleme yok)."""
    return (os.getenv("TRIAGE_SKIP_FT") or "").strip() in {"1", "true", "True", "yes"}


def _ensure_loaded(device: str | None = None) -> tuple[Any, np.ndarray]:
    """Model + anchor matrisi lazy load (thread-safe)."""
    global _model, _anchor_matrix
    if _model is not None and _anchor_matrix is not None:
        return _model, _anchor_matrix
    with _lock:
        if _model is not None and _anchor_matrix is not None:
            return _model, _anchor_matrix
        from src.retrieval.embed import encode_texts, load_embedder

        model = load_embedder(device=device)
        mat = encode_texts(
            model,
            list(_IMPLICIT_ANCHORS),
            batch_size=8,
            show_progress=False,
        )
        _model = model
        _anchor_matrix = mat
        return _model, _anchor_matrix


def score_ft(message: str, *, device: str | None = None) -> float:
    """Mesajın örtük acil-dil skoru [0, 1] (max cosine vs anchors)."""
    text = (message or "").strip()
    if not text or _skip_ft():
        return 0.0
    model, anchors = _ensure_loaded(device=device)
    from src.retrieval.embed import encode_texts

    q = encode_texts(model, [text], batch_size=1, show_progress=False)
    # anchors (n, dim), q (1, dim) — L2-normalized → cosine = dot
    sims = (anchors @ q.T).ravel()
    if sims.size == 0:
        return 0.0
    return float(np.clip(float(sims.max()), 0.0, 1.0))


def reset_ft_cache() -> None:
    """Testler için model cache temizliği."""
    global _model, _anchor_matrix
    with _lock:
        _model = None
        _anchor_matrix = None
