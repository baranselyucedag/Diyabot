"""Merkezi bellek konfigürasyonu — Phase 3+ için tek kaynak.

Plan dokümanındaki (cahing_mimari_tasarim.md §8) tüm alanlar birebir burada
tanımlıdır. Diğer modüller varsayılan değerleri buradan okur; böylece eşikler,
periyotlar ve güvenlik parametreleri tek noktadan yönetilir.
"""

from __future__ import annotations

MEMORY_CONFIG: dict = {
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
}
