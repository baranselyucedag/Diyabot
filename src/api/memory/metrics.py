"""İzleme sayaçları — basit in-memory Prometheus counter'ları.

Process yeniden başlayınca sayaçlar sıfırlanır (kalıcı metrik deposu yoktur).
Bu modül kasıtlı olarak hafiftir: yeni bir bağımlılık (prometheus-client)
gerekmez; sayaçlar mevcut ``log_event`` çağrılarından türetilir ve
``/metrics`` endpoint'inde Prometheus text formatında expose edilir.
"""

from __future__ import annotations

from collections import Counter

_COUNTERS: Counter[str] = Counter()

# (component, event) → metric adı eşlemesi (log_event çıktısından türetme).
# Not: "llm_client.call_failed" de LLM çağrısıdır; "toplam çağrı" anlamını
# korumak için hem done hem failed aynı sayacı artırır.
_METRIC_BY_EVENT: dict[tuple[str, str], str] = {
    ("llm_client", "call_done"): "memory_llm_calls_total",
    ("llm_client", "call_failed"): "memory_llm_calls_total",
    ("memory_maintenance", "done"): "memory_maintenance_task_total",
    ("staleness", "staleness_conflict"): "memory_staleness_conflicts_total",
}

_METRIC_ORDER = [
    "memory_llm_calls_total",
    "memory_maintenance_task_total",
    "memory_staleness_conflicts_total",
]


def record_event(component: str, event: str) -> None:
    """Eşleşen (component, event) için ilgili sayacı artırır (eşleşme yoksa no-op)."""
    metric = _METRIC_BY_EVENT.get((component, event))
    if metric is not None:
        _COUNTERS[metric] += 1


def get_counter(name: str) -> int:
    """Belirtilen sayacın mevcut değerini döndürür (yoksa 0)."""
    return _COUNTERS.get(name, 0)


def render_metrics() -> str:
    """Prometheus text exposition formatında sayaçları döndürür."""
    return (
        "\n".join(f"{name} {_COUNTERS.get(name, 0)}" for name in _METRIC_ORDER)
        + "\n"
    )


def reset_counters() -> None:
    """Tüm sayaçları sıfırlar (testler için)."""
    _COUNTERS.clear()
