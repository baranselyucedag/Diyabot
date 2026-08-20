"""Bellek bakım orkestratörü — Phase 3 (async, conversation_lock kilitli).

Plan (cahing_mimari_tasarim.md §3, §4, §6) akışına birebir uyumludur:

  ADIM A — CALL 2 (memory_maintenance_call): detection + not adayları + özet, tek LLM çağrısı.
  ADIM B — deterministik: triage risk sınıfı + çakışma kuralları + profil yazma.
  ADIM C — SADECE yüksek riskli alan tespit edildiyse CALL 3 (critical_verification).
  ADIM D — Profil/hafıza yazma (tek sıralı task, başka yazıcı yok).
  ADIM E — Pending conflict expiry (aynı conversation_lock içinde, ayrı task YOK).

LLM çağrı garantisi: güncelleme yoksa 1, düşük/orta riskli 2, yüksek riskli 3.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

from src.api.memory.config import MEMORY_CONFIG
from src.api.memory.deterministic import (
    check_conflicts,
    check_staleness_deterministic,
    triage_classify,
    verify_grounding_deterministic,
    word_set,
)
from src.api.memory.expiry import expire_pending_conflicts_locked
from src.api.memory.llm_client import llm_call_json_schema
from src.api.memory.logger import log_event
from src.api.memory.memory_store import (
    load_memory_context,
    save_notes_with_limit,
    summarize_notes_for_prompt,
)
from src.api.memory.models import (
    CriticalVerifyResult,
    MaintenanceResult,
    Medication,
    MedicationStatus,
    Note,
    NoteStaleness,
    PendingConflict,
    PendingConflictsStore,
    PendingConflictStatus,
    Profile,
    ProfileUpdate,
    Turn,
)
from src.api.memory.storage import (
    get_conversation_lock,
    load_pending_conflicts,
    save_index,
    save_pending_conflicts,
    save_profile,
    save_summary,
)
from src.api.memory.timeutil import utcnow


# ---------------------------------------------------------------------------
# Prompt sabitleri (plan dokümanından)
# ---------------------------------------------------------------------------

MAINTENANCE_PROMPT = """Aşağıdaki konuşma turlarını analiz et ve TEK bir JSON nesnesi döndür.

MEVCUT PROFİL (JSON):
{profile_json}

MEVCUT NOTLAR (tekrar etme):
{existing_notes_summary}

SON TURLAR:
{recent_turns}
{summary_hint}

KESİN KURALLAR:
1. source_turn_id MUTLAKA recent_turns içindeki bir ID olmalı. Bulunamıyorsa null yaz.
2. Notun claim'ini destekleyen kelimeler kaynak turn'da yoksa notu çıkarma.
3. summary sadece needs_summary=true ise doldurulur, yoksa null.
4. Güncelleme yoksa profile_update.has_update=false yaz.

Şunları TEK JSON'da döndür:
{{
  "profile_update": {{
    "has_update": true,
    "field": "medications",
    "proposed_change": {{"action": "add", "detail": {{"name": "Metformin"}}}},
    "evidence_span": "mesajdaki ilgili kısmın aynen kopyası",
    "confidence": 0.9
  }},
  "candidate_notes": [
    {{"content": "...", "category": "observation", "source_turn_id": "msg_1"}}
  ],
  "summary": "..."
}}
"""

CRITICAL_VERIFY_PROMPT = """Bu, hastanın {field} alanında YÜKSEK RİSKLİ bir değişiklik: {proposed_change}
Kanıt: "{evidence_span}"
Bağlam: {recent_turns}

ELEŞTİREL bir gözle değerlendir — acele karar verme:
- Bu net bir ifade mi, yoksa yorum/tahmin mi?
- Yanlış anlaşılma ihtimali var mı? (ör. hasta soruyor mu yoksa bildiriyor mu?)
- Mevcut profille çelişen bir taraf var mı?

JSON döndür: {{"onayla": true/false, "gerekce": "...", "eminlik": 0.0-1.0}}
"""


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def _format_turns(turns: list[Turn]) -> str:
    """Turn listesini prompt'a basılabilir düz metne çevirir."""
    if not turns:
        return "(son tur yok)"
    lines = []
    for t in turns:
        role = t.role.value if hasattr(t.role, "value") else str(t.role)
        triage = f" [triage={t.triage}]" if t.triage else ""
        lines.append(f"[{role}] {t.turn_id}{triage}: {t.content}")
    return "\n".join(lines)


def _find_source_turn_id_for_evidence(evidence_span: str, turns: list[Turn]) -> Optional[str]:
    """Bir evidence_span'ın hangi turn'e ait olduğını bulur.

    Önce tam substring eşleşmesi dener; bulunamazsa kelime örtüşmesi en yüksek
    turn'ü döner. Hiç eşleşme yoksa None.
    """
    if not evidence_span or not turns:
        return None

    evidence_lower = evidence_span.strip().lower()
    for turn in turns:
        if evidence_lower in turn.content.lower():
            return turn.turn_id

    evidence_words = word_set(evidence_span)
    if not evidence_words:
        return None

    best_turn: Optional[Turn] = None
    best_overlap = 0.0
    for turn in turns:
        turn_words = word_set(turn.content)
        if not turn_words:
            continue
        overlap = len(evidence_words & turn_words) / len(evidence_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn

    return best_turn.turn_id if best_turn else None


def _field_snapshot(profile: Profile, field: str) -> dict:
    """Profilin belirtilen alanının mevcut değerini dict olarak döner."""
    patient = profile.patient
    if field == "medications":
        return {"medications": [m.model_dump() for m in patient.medications]}
    if field == "complications":
        return {"complications": patient.complications}
    if field == "goals":
        return {"goals": patient.goals}
    if field == "allergies":
        return {"allergies": patient.allergies}
    if field == "monitoring":
        return {"monitoring": patient.monitoring}
    return {}


def _apply_change_to_profile(profile: Profile, field: str, action: str, detail: dict) -> None:
    """Profil alanını add/remove/update aksiyonuna göre kalıcı olarak değiştirir."""
    patient = profile.patient

    if field == "medications":
        if action == "add":
            patient.medications.append(
                Medication(
                    name=detail.get("name", ""),
                    dose=detail.get("dose", ""),
                    frequency=detail.get("frequency", ""),
                    started=detail.get("started", ""),
                    status=MedicationStatus.ACTIVE,
                )
            )
        elif action == "remove":
            # REWRITE_AS_UPDATE: ilacı silmek yerine durdur (status=stopped).
            name = detail.get("name", "").strip().lower()
            for med in patient.medications:
                if med.name.strip().lower() == name:
                    med.status = MedicationStatus.STOPPED
                    break
        elif action == "update":
            name = detail.get("name", "").strip().lower()
            for med in patient.medications:
                if med.name.strip().lower() == name:
                    if "dose" in detail:
                        med.dose = detail["dose"]
                    if "frequency" in detail:
                        med.frequency = detail["frequency"]
                    break

    elif field == "complications":
        if action == "add":
            value = detail.get("value", "").strip()
            if value and value not in patient.complications:
                patient.complications.append(value)

    elif field == "goals":
        if action == "update":
            value = detail.get("value", "").strip()
            if value and value not in patient.goals:
                patient.goals.append(value)

    elif field == "allergies":
        if action == "add":
            value = detail.get("value", "").strip()
            if value and value not in patient.allergies:
                patient.allergies.append(value)

    elif field == "monitoring":
        if action in ("add", "update"):
            patient.monitoring.update(detail)


def _add_pending_conflict(
    conv_id: str,
    pu: ProfileUpdate,
    conflict: dict,
    profile: Profile,
) -> None:
    """Çözülemeyen bir profil değişikliğini bekleyen çakışma olarak kaydeder."""
    store = load_pending_conflicts(conv_id) or PendingConflictsStore(items=[])
    store.items.append(
        PendingConflict(
            conflict_id=f"conf_{uuid.uuid4().hex[:12]}",
            type=conflict.get("conflict_type") or "PROFILE_CONFLICT",
            existing={"field": pu.field, "current": _field_snapshot(profile, pu.field)},
            proposed={
                "field": pu.field,
                "proposed_change": pu.proposed_change,
            },
            status=PendingConflictStatus.PENDING,
        )
    )
    save_pending_conflicts(conv_id, store)


# ---------------------------------------------------------------------------
# ADIM A — CALL 2
# ---------------------------------------------------------------------------


async def memory_maintenance_call(
    profile: Profile,
    existing_notes,
    recent_turns: list[Turn],
    needs_summary: bool,
) -> MaintenanceResult:
    """Tek LLM çağrısı: profil güncelleme + not adayları + özet."""
    summary_hint = "BU TUR ÖZET GEREKİYOR (turn_count % 5 == 0)" if needs_summary else ""
    prompt = MAINTENANCE_PROMPT.format(
        profile_json=profile.model_dump_json(),
        existing_notes_summary=summarize_notes_for_prompt(existing_notes),
        recent_turns=_format_turns(recent_turns),
        summary_hint=summary_hint,
    )
    return await llm_call_json_schema(
        prompt,
        MaintenanceResult,
        temperature=0.0,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# ADIM C — CALL 3 (koşullu: yüksek riskli alan)
# ---------------------------------------------------------------------------


async def critical_verification(
    detection: ProfileUpdate,
    profile: Profile,
    context_turns: list[Turn],
) -> CriticalVerifyResult:
    """Yüksek riskli profil değişikliğini ikinci (daha sıkı) LLM çağrısıyla doğrular."""
    prompt = CRITICAL_VERIFY_PROMPT.format(
        field=detection.field,
        proposed_change=json.dumps(detection.proposed_change, ensure_ascii=False),
        evidence_span=detection.evidence_span,
        recent_turns=_format_turns(context_turns),
    )
    return await llm_call_json_schema(
        prompt,
        CriticalVerifyResult,
        temperature=0.0,
        max_tokens=512,
    )


# ---------------------------------------------------------------------------
# ADIM B — profil güncelleme uygulama
# ---------------------------------------------------------------------------


def apply_profile_update(
    conv_id: str,
    change: dict,
    profile: Profile,
    source_turn_id: Optional[str] = None,
) -> Profile:
    """Profil alanını günceller, metadata'yı tazeler ve diske yazar.

    Args:
        conv_id: Konuşma kimliği.
        change: {"field": str, "action": str, "detail": dict, "confidence": float | None}
        profile: Güncellenecek profil.
        source_turn_id: Güncellemenin kanıtlandığı kaynak turn ID'si (opsiyonel).

    Returns:
        Güncellenmiş (ve kaydedilmiş) profil.
    """
    field = change.get("field", "")
    action = change.get("action", "")
    detail = change.get("detail", {})
    confidence = change.get("confidence")

    _apply_change_to_profile(profile, field, action, detail)

    # Metadata güncelleme (plan §3.2: source_turn_ids, confidence_scores, updated_at)
    profile.updated_at = utcnow()
    profile.metadata.updated_at = utcnow()
    if confidence is not None:
        profile.metadata.confidence_scores[field] = confidence
    if source_turn_id and source_turn_id not in profile.metadata.source_turn_ids:
        profile.metadata.source_turn_ids.append(source_turn_id)

    save_profile(conv_id, profile)
    return profile


# ---------------------------------------------------------------------------
# Ana orkestratör (ADIM A-E)
# ---------------------------------------------------------------------------


async def memory_maintenance_task(
    conv_id: str,
    recent_turns: list[Turn],
    needs_summary: bool = False,
) -> dict:
    """Bir konuşma turu sonrası bellek bakımını çalıştırır (conversation_lock altında).

    Args:
        conv_id: Konuşma kimliği.
        recent_turns: Son turlar (kronolojik).
        needs_summary: True ise özet üretilir (turn_count % 5 == 0).

    Returns:
        Özet dict: profile_updated, notes_added, summary_saved, stale_notes,
        expired_conflicts.
    """
    lock = get_conversation_lock(conv_id)
    async with lock:
        ctx = await asyncio.to_thread(load_memory_context, conv_id)
        profile: Profile = ctx["profile"]
        notes = ctx["notes"]
        index = ctx["index"]

        # ADIM A — CALL 2
        result = await memory_maintenance_call(profile, notes, recent_turns, needs_summary)

        changed_field: Optional[str] = None
        profile_updated = False

        # ADIM B — profil güncelleme (çakışma + risk + doğrulama)
        pu = result.profile_update
        if pu is not None and pu.has_update:
            changed_field = pu.field
            risk = triage_classify(pu.field)

            if risk["is_high_risk"]:
                # ADIM C — CALL 3 (koşullu)
                verify = await critical_verification(pu, profile, recent_turns)
                onaylandi = (
                    verify.onayla
                    and verify.eminlik >= MEMORY_CONFIG["critical_verify_threshold"]
                )
            else:
                onaylandi = pu.confidence >= MEMORY_CONFIG["gate_approval_threshold"]

            if onaylandi:
                proposed = {
                    "field": pu.field,
                    "action": pu.proposed_change.get("action"),
                    "detail": pu.proposed_change.get("detail", {}),
                }
                conflict = check_conflicts(proposed, profile)

                if conflict["action"] == "FLAG_PENDING":
                    await asyncio.to_thread(
                        _add_pending_conflict, conv_id, pu, conflict, profile
                    )
                elif conflict["action"] == "IGNORE":
                    pass  # bilinen tekrar; sessizce yok say
                elif conflict["action"] in ("PROCEED", "REWRITE_AS_UPDATE", "LATEST_WINS"):
                    change = {
                        "field": pu.field,
                        "action": pu.proposed_change.get("action"),
                        "detail": pu.proposed_change.get("detail", {}),
                        "confidence": pu.confidence,
                    }
                    source_turn_id = _find_source_turn_id_for_evidence(pu.evidence_span, recent_turns)
                    await asyncio.to_thread(
                        apply_profile_update,
                        conv_id,
                        change,
                        profile,
                        source_turn_id=source_turn_id,
                    )
                    profile_updated = True
                    index.last_profile_update_at_turn = index.turn_count
            else:
                # Doğrulama reddedildi → pending conflict olarak işaretle (sessiz geçme).
                await asyncio.to_thread(
                    _add_pending_conflict,
                    conv_id,
                    pu,
                    {"has_conflict": True, "conflict_type": "UNVERIFIED"},
                    profile,
                )

        # ADIM C (notlar) — grounding doğrulaması + kayıt
        grounded_notes: list[Note] = []
        for cand in result.candidate_notes:
            grounding = verify_grounding_deterministic(cand, recent_turns)
            if grounding["grounded"]:
                grounded_notes.append(
                    Note(
                        note_id=f"note_{uuid.uuid4().hex[:12]}",
                        content=cand.content,
                        category=cand.category,
                        source_turns=[cand.source_turn_id] if cand.source_turn_id else [],
                        confidence=round(grounding["overlap"], 3),
                        staleness=NoteStaleness.FRESH,
                    )
                )

        notes_added = 0
        if grounded_notes:
            notes.items.extend(grounded_notes)
            await asyncio.to_thread(
                save_notes_with_limit,
                conv_id,
                notes,
                max_notes=MEMORY_CONFIG["max_notes_per_conversation"],
            )
            notes_added = len(grounded_notes)
            index.last_note_extraction_at_turn = index.turn_count

        # ADIM D — staleness (güncel profil + güncel notlar ile)
        stale_items = await asyncio.to_thread(
            check_staleness_deterministic,
            profile,
            notes.items,
            changed_field=changed_field,
            conv_id=conv_id,
        )

        # Özet
        summary_saved = False
        if needs_summary and result.summary:
            await asyncio.to_thread(save_summary, conv_id, result.summary)
            summary_saved = True
            index.last_summary_at_turn = index.turn_count

        # ADIM E — pending conflict expiry (aynı lock içinde, ayrı task YOK)
        expired_conflicts = await asyncio.to_thread(
            expire_pending_conflicts_locked, conv_id
        )

        # Index güncelle
        index.updated_at = utcnow()
        await asyncio.to_thread(save_index, conv_id, index)

        log_event(
            "memory_maintenance",
            "done",
            conv_id,
            profile_updated=profile_updated,
            notes_added=notes_added,
            summary_saved=summary_saved,
            stale_notes=len(stale_items),
            expired_conflicts=expired_conflicts,
        )

        return {
            "profile_updated": profile_updated,
            "notes_added": notes_added,
            "summary_saved": summary_saved,
            "stale_notes": len(stale_items),
            "expired_conflicts": expired_conflicts,
        }
