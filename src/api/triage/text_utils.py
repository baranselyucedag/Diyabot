"""Triage metin yardımcıları — ortak normalizasyon.

Adım 1 (numeric) ve Adım 2 (regex) aynı `norm`u kullanır; tutarlılık için
tek kaynak burasıdır.
"""

from __future__ import annotations


def norm(text: str) -> str:
    """Türkçe sadeleştirme — 1:1 uzunluk korur (pozisyon eşlemesi için).

    'İ'.casefold() → 'i' + combining dot (2 char) olduğu için İ/I önce
    tek 'i'ye çevrilir; kalan combining dot temizlenir.
    """
    t = text or ""
    t = t.replace("İ", "i").replace("I", "i")
    t = t.casefold().replace("\u0307", "")  # combining dot above
    return (
        t.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )
