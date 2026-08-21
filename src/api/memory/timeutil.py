"""Zaman yardımcıları — timezone-aware UTC "şimdi" için tek kaynak.

`datetime.utcnow()` Python 3.12+ itibarıyla deprecated'tır ve timezone-naive
(nesnesiz) bir değer üretir. Naive UTC değerler, farklı timezone'daki sunucular
veya DST geçişlerinde tutarsız karşılaştırmalara yol açabilir. Bu yüzden tüm
modüller "şimdi" için bu tek helper'ı kullanır.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Timezone-aware UTC şimdiki zamanı döndürür."""
    return datetime.now(timezone.utc)
