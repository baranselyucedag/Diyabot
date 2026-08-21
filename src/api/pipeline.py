"""RAG pipeline: triage (detailed) → retrieve → rerank → LLM + Memory entegrasyonu."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.api.llm import DEFAULT_DISCLAIMER, generate_answer
from src.api.memory.maintenance import memory_maintenance_task
from src.api.memory.memory_store import (
    append_turn_and_update_index,
    create_turn_atomic,
    load_memory_context,
    load_recent_turns_for_prompt,
)
from src.api.memory.models import Turn
from src.api.triage import canned_response, detect_triage_detailed
from src.retrieval.embed import CHUNKS_DIR, DEFAULT_INDEX_DIR, TOP_K
from src.retrieval.rerank import retrieve_and_rerank
from src.retrieval.retrieve import build_content_map, build_section_map

logger = logging.getLogger(__name__)

LLM_TOP_N = 3
FOLLOW_UPS = [
    "Prediyabet nedir?",
    "Kan şekerimi nasıl takip etmeliyim?",
    "Egzersize nasıl başlamalıyım?",
]


def _hit_to_source(
    hit: Any,
    content_map: dict[str, str],
    section_map: dict[str, str | None],
) -> dict[str, Any]:
    """Rerank hit → frontend Source şeması.

    - `section`: chunk_id (debug/geriye-uyumluluk için KORUNUR; UI'da gösterilmez).
    - `section_label`: insan-okunur bölüm etiketi (section_path); yoksa None.
    """
    body = content_map.get(hit.chunk_id, hit.preview or "")
    flat = " ".join(body.split())
    snippet = flat[:220] + ("…" if len(flat) > 220 else "")
    return {
        "document": hit.source or hit.chunk_id,
        "section": hit.chunk_id,
        "section_label": section_map.get(hit.chunk_id),
        "snippet": snippet,
    }


def _format_profile_for_prompt(profile: Any) -> str:
    """Profili prompt için okunabilir metne çevirir."""
    if not profile or not hasattr(profile, "patient"):
        return ""
    p = profile.patient
    lines = [
        f"- Diyabet tipi: {p.diabetes_type}",
        f"- Tanı yılı: {p.diagnosis_year}",
    ]
    if p.medications:
        meds = ", ".join(
            f"{m.name} ({m.dose}, {m.frequency})"
            for m in p.medications
            if m.status.value == "active"
        )
        if meds:
            lines.append(f"- Aktif ilaçlar: {meds}")
    if p.complications:
        lines.append(f"- Komplikasyonlar: {', '.join(p.complications)}")
    if p.allergies:
        lines.append(f"- Alerjiler: {', '.join(p.allergies)}")
    if p.goals:
        lines.append(f"- Hedefler: {', '.join(p.goals)}")
    if p.monitoring:
        mon = ", ".join(f"{k}: {v}" for k, v in p.monitoring.items())
        if mon:
            lines.append(f"- İzlem: {mon}")
    return "\n".join(lines) if lines else "(profil boş)"


def _format_notes_for_prompt(notes: Any) -> str:
    """Notları prompt için özet string'e çevirir."""
    if not notes or not notes.items:
        return "(ilgili not yok)"
    lines = []
    for note in notes.items:
        stale = " [STALE]" if note.staleness.value == "stale" else ""
        lines.append(f"- [{note.category} | conf={note.confidence:.2f}{stale}] {note.content}")
    return "\n".join(lines)


def _format_pending_conflicts_for_prompt(pending: Any) -> str:
    """Pending conflict'leri prompt için formatlar."""
    if not pending or not pending.items:
        return ""
    lines = ["### BEKLEYEN NETLEŞTİRMELER"]
    for item in pending.items:
        if item.status.value == "pending":
            field = (item.existing or {}).get("field") or (item.proposed or {}).get("field")
            lines.append(f"- {item.type} ({field}): Kullanıcıdan netleştirme bekleniyor")
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_turns(recent_turns: list[Turn]) -> str:
    """Son turları prompt formatına çevirir."""
    if not recent_turns:
        return "(son tur yok)"
    lines = []
    for t in recent_turns:
        role = t.role.value if hasattr(t.role, "value") else str(t.role)
        triage = f" [triage={t.triage}]" if t.triage else ""
        lines.append(f"[{role}] {t.turn_id}{triage}: {t.content}")
    return "\n".join(lines)


def build_user_prompt_with_memory(
    question: str,
    contexts: list[dict],
    memory_ctx: dict,
    recent_turns: list[Turn],
) -> str:
    """Hafıza bileşenleriyle zenginleştirilmiş KULLANICI mesajı üretir.

    SYSTEM_PROMPT içermez — o, `generate_answer` içinde ayrı system mesajı olarak
    gönderilir (çift sarmalama yok).
    """
    profile_text = _format_profile_for_prompt(memory_ctx.get("profile"))
    summary_text = memory_ctx.get("summary") or ""
    notes_text = _format_notes_for_prompt(memory_ctx.get("notes"))
    pending_text = _format_pending_conflicts_for_prompt(memory_ctx.get("pending_conflicts"))

    context_text = "\n\n".join(
        f"[KAYNAK {i + 1} | {c.get('chunk_id', '')} | {c.get('source', '')}]\n{c.get('content', '')}"
        for i, c in enumerate(contexts)
    ) if contexts else "(kaynak yok)"

    return f"""### PROFİL
{profile_text}

{pending_text}
### KONUŞMA GEÇMİŞİ
**Özet:** {summary_text or "(özet yok)"}
**Son 3 Tur:**
{_format_turns(recent_turns[-3:])}
**İlgili Notlar:**
{notes_text}

### KAYNAKLAR
{context_text}

### SORU
{question}
"""


async def run_chat(
    message: str,
    conversation_id: str,
    history: list[dict] | None = None,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    retrieve_k: int = TOP_K,
    llm_top_n: int = LLM_TOP_N,
    device: str | None = None,
) -> dict[str, Any]:
    """Tek kullanıcı mesajı için uçtan uca RAG + Memory cevabı üretir."""
    text = (message or "").strip()
    if not text:
        raise ValueError("Boş mesaj.")

    # 1. Hafıza yükle
    memory_ctx = load_memory_context(conversation_id)

    # 2. Son N tur (frontend'den geliyor; yoksa diskten yüklenir)
    if history:
        recent_turns = [Turn(**t) for t in history[-6:]]
    else:
        recent_turns = load_recent_turns_for_prompt(conversation_id, 6)

    # 3. Gerçek triage (mevcut triage katmanı)
    decision = detect_triage_detailed(text)
    triage = decision.level
    logger.info(
        "triage level=%s source=%s score=%s tempered=%s reason=%s",
        triage,
        decision.source,
        decision.score,
        decision.tempered,
        decision.reason[:200],
    )

    # 4. Canned yanıt (EMERGENCY / REFUSE / tempered YELLOW) — LLM'e gidilmez
    canned = canned_response(
        triage,
        flags=decision.flags,
        tempered=decision.tempered,
        reason=decision.reason if decision.tempered else None,
    )
    if canned:
        return {
            "answer": canned,
            "triage_level": triage,
            "sources": [],
            "disclaimer": DEFAULT_DISCLAIMER,
            "follow_ups": FOLLOW_UPS[:2],
        }

    # 5. RAG (mevcut)
    content_map = build_content_map(chunks_dir)
    section_map = build_section_map(chunks_dir)
    hits = retrieve_and_rerank(
        query=text,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        top_k=retrieve_k,
        device=device,
    )
    top = hits[: max(1, llm_top_n)]
    contexts = [
        {
            "chunk_id": h.chunk_id,
            "source": h.source,
            "preview": h.preview,
            "content": content_map.get(h.chunk_id, h.preview),
        }
        for h in top
    ]
    sources = [_hit_to_source(h, content_map, section_map) for h in top]

    if not contexts:
        return {
            "answer": (
                "Bu konuda doğrulanmış eğitim kaynağımda net bir eşleşme bulamadım. "
                "Tip 2 diyabet eğitimi kapsamında prediyabet, beslenme, egzersiz veya "
                "kan şekeri takibi sorabilirsiniz. Kişisel kararlar için hekiminize danışın."
            ),
            "triage_level": triage,
            "sources": [],
            "disclaimer": DEFAULT_DISCLAIMER,
            "follow_ups": FOLLOW_UPS,
        }

    # 6. Prompt (hafıza + RAG) → LLM cevap (CALL 1, mevcut)
    prompt = build_user_prompt_with_memory(text, contexts, memory_ctx, recent_turns)
    answer = generate_answer(text, contexts, user_prompt=prompt)

    # 7. YELLOW: hekim uyarısı + grey-zone reason (şeffaflık MVP)
    if triage == "YELLOW":
        if decision.reason and decision.source == "grey_zone" and not decision.tempered:
            answer = answer + f"\n\n_Triage notu: {decision.reason}_"
        if "hekim" not in answer.casefold():
            answer = (
                answer
                + "\n\nNot: Anlattığınız durum yakın zamanda hekim değerlendirmesi gerektirebilir."
            )

    # 8. Hafızaya tur ekle — ikisi de aynı conversation_lock altında, atomik.
    #    turn_id'yi ARTIK elle üretmiyoruz (eski hali kilit dışında tahmin
    #    ediyordu, eşzamanlı istekte çakışma riski taşıyordu). Her iki
    #    fonksiyon da (create_turn_atomic / append_turn_and_update_index)
    #    turn_id'yi kilit İÇİNDE, kendisi üretiyor.
    user_turn = await create_turn_atomic(conversation_id, "user", text, None)
    assistant_turn = await append_turn_and_update_index(
        conversation_id, "assistant", answer, triage
    )

    # 9. ASYNC: Memory Maintenance Task (fire-and-forget; expiry dahil)
    needs_summary = (memory_ctx["index"].turn_count + 2) % 5 == 0
    asyncio.create_task(
        memory_maintenance_task(
            conversation_id,
            recent_turns + [user_turn, assistant_turn],
            needs_summary=needs_summary,
        )
    )

    return {
        "answer": answer,
        "triage_level": triage,
        "sources": sources,
        "disclaimer": DEFAULT_DISCLAIMER,
        "follow_ups": FOLLOW_UPS,
    }
