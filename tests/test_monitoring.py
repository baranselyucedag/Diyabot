"""İzleme altyapısı testleri — /health, /metrics, APP_ENV override."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.memory import metrics, storage


def test_health_ok():
    from src.api.app import app

    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["memory_ready"] is True


def test_health_degraded_when_data_not_writable(monkeypatch, tmp_path):
    from src.api.app import app

    # DATA_DIR'i bir DOSYA'ya çevir → mkdir başarısız → 503.
    blocker = tmp_path / "blocker"
    blocker.write_text("file", encoding="utf-8")
    monkeypatch.setattr(storage, "DATA_DIR", blocker)

    client = TestClient(app)
    resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["memory_ready"] is False
    assert "reason" in body


def test_metrics_increment():
    from src.api.memory.logger import log_event

    metrics.reset_counters()

    log_event("llm_client", "call_done", "conv_x")
    log_event("llm_client", "call_done", "conv_x")
    log_event("memory_maintenance", "done", "conv_x")
    log_event("staleness", "staleness_conflict", "conv_x", note_id="n1")

    assert metrics.get_counter("memory_llm_calls_total") == 2
    assert metrics.get_counter("memory_maintenance_task_total") == 1
    assert metrics.get_counter("memory_staleness_conflicts_total") == 1


def test_metrics_endpoint():
    from src.api.app import app
    from src.api.memory.logger import log_event

    metrics.reset_counters()
    log_event("llm_client", "call_done", "conv_x")

    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "memory_llm_calls_total 1" in body
    assert "memory_maintenance_task_total 0" in body
    assert "memory_staleness_conflicts_total 0" in body


def test_config_env_override():
    from src.api.memory.config import build_effective_config

    # dev (varsayılan) → base değerler korunur.
    dev = build_effective_config("dev")
    assert dev["llm_request_timeout_seconds"] == 60
    assert dev["log_retention_days"] == 30

    # staging override.
    staging = build_effective_config("staging")
    assert staging["llm_request_timeout_seconds"] == 30
    assert staging["log_retention_days"] == 14

    # prod override.
    prod = build_effective_config("prod")
    assert prod["llm_request_timeout_seconds"] == 20
    assert prod["log_retention_days"] == 90


def test_config_default_unchanged_when_app_env_unset(monkeypatch):
    from src.api.memory.config import build_effective_config

    monkeypatch.delenv("APP_ENV", raising=False)

    # APP_ENV yok → dev davranışı (base).
    eff = build_effective_config()
    assert eff["llm_request_timeout_seconds"] == 60
    assert eff["log_retention_days"] == 30
    assert eff["encryption_enabled"] is False
