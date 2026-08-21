"""Merkezi bellek konfigürasyonu — Phase 3+ için tek kaynak.

Plan dokümanındaki (cahing_mimari_tasarim.md §8) tüm alanlar birebir burada
tanımlıdır. Diğer modüller varsayılan değerleri buradan okur; böylece eşikler,
periyotlar ve güvenlik parametreleri tek noktadan yönetilir.

Faz 5.3 (env-config):
- ``_BASE_CONFIG`` = dev (varsayılan) davranış. HİÇBİR ZAMAN mutate edilmez.
- ``_ENV_OVERRIDES`` = staging/prod için seçili alan override'ları.
- ``APP_ENV`` (varsayılan "dev") ortam değişkenine göre ``MEMORY_CONFIG`` hesaplanır.
- APP_ENV tanımlı değilse (mevcut hiçbir ortamda tanımlı değil) davranış
  eskisiyle BİREBİR aynıdır.
"""

from __future__ import annotations

import os

_BASE_CONFIG: dict = {
    # --- Prompt / tur yönetimi ---
    "recent_turns_count": 6,           # prompt'a basılacak son tur sayısı
    "summarize_every_n_turns": 5,      # özet üretme periyodu (CALL 2 içine gömülü)

    # --- Güvenlik eşikleri ---
    "gate_approval_threshold": 0.70,   # düşük/orta riskli güncelleme onay eşiği
    "critical_verify_threshold": 0.75, # yüksek riskli alanda eminlik eşiği

    # --- Deterministik grounding ---
    "grounding_min_word_overlap": 0.30,

    # --- Not yönetimi ---
    "note_categories": ["symptom", "observation", "plan", "measurement", "advice"],
    "max_notes_per_conversation": 50,

    # --- Staleness (eskime) ---
    "staleness_plan_max_days": 30,
    "staleness_check_scope": "affected_category_only",

    # --- Bekleyen çakışma süre sonu (klinik veri: 1 gün) ---
    "pending_conflict_expiry_days": 1,                 # 7 -> 1 (güvenlik)
    "pending_conflict_expiry_action": "expired_rejected_with_followup",

    # --- Maliyet / gözlenebilirlik ---
    "max_llm_calls_per_turn_target": 2,  # yüksek riskli turda 3'e çıkabilir
    "retention_days": 365,               # konuşma verisi saklama süresi (cron)
    "llm_calls_p99_alert_threshold": 3,  # p99 > 3 ise alert

    # --- LLM istemcisi dayanıklılığı ---
    "llm_request_timeout_seconds": 60,   # tek LLM isteği için timeout (sn)
    "llm_max_retries": 3,                # geçici hatalarda en fazla deneme sayısı
    "llm_retry_backoff_seconds": 2,      # üstel geri çekilmenin taban aralığı (sn)

    # --- Log yönetimi ---
    "log_retention_days": 30,            # logs/memory dosyalarının saklama süresi

    # --- At-rest şifreleme (Faz 5.1) ---
    # MVP'de kapalı. True ise `master_key_env_var` değişkeni ZORUNLU olur;
    # eksikse uygulama başlatılamaz (sessizce şifresiz devam ETMEZ).
    "encryption_enabled": False,
    "master_key_env_var": "MEMORY_MASTER_KEY",
}

# --- Ortam bazlı override (dev/staging/prod) — Faz 5.3 ---
_ENV_OVERRIDES: dict[str, dict] = {
    "staging": {
        "llm_request_timeout_seconds": 30,
        "llm_max_retries": 2,
        "log_retention_days": 14,
    },
    "prod": {
        "llm_request_timeout_seconds": 20,
        "llm_max_retries": 3,
        "log_retention_days": 90,
    },
}

APP_ENV = os.getenv("APP_ENV", "dev")


def build_effective_config(app_env: str | None = None) -> dict:
    """APP_ENV override'ları uygulanmış config'in KOPYASINI döndürür.

    ``_BASE_CONFIG`` asla mutate edilmez; çağırana ortama göre güncellenmiş
    yeni bir sözlük verilir. Testler için deterministiktir.
    """
    env = app_env if app_env is not None else os.getenv("APP_ENV", "dev")
    merged = dict(_BASE_CONFIG)
    merged.update(_ENV_OVERRIDES.get(env, {}))
    return merged


# Diğer modüllerin beklediği isim — ortam etkisi uygulanmış config.
MEMORY_CONFIG = build_effective_config()
