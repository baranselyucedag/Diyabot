"""RAG pipeline: triage (detailed) → retrieve → rerank → LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.api.llm import DEFAULT_DISCLAIMER, generate_answer
from src.api.triage import canned_response, detect_triage_detailed
from src.retrieval.embed import CHUNKS_DIR, DEFAULT_INDEX_DIR, TOP_K
from src.retrieval.rerank import retrieve_and_rerank
from src.retrieval.retrieve import build_content_map

logger = logging.getLogger(__name__)

LLM_TOP_N = 3
FOLLOW_UPS = [
    "Prediyabet nedir?",
    "Kan şekerimi nasıl takip etmeliyim?",
    "Egzersize nasıl başlamalıyım?",
]


def _hit_to_source(hit: Any, content_map: dict[str, str]) -> dict[str, str]:
    """Rerank hit → frontend Source şeması."""
    body = content_map.get(hit.chunk_id, hit.preview or "")
    flat = " ".join(body.split())
    snippet = flat[:220] + ("…" if len(flat) > 220 else "")
    return {
        "document": hit.source or hit.chunk_id,
        "section": hit.chunk_id,
        "snippet": snippet,
    }


def run_chat(
    message: str,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    retrieve_k: int = TOP_K,
    llm_top_n: int = LLM_TOP_N,
    device: str | None = None,
) -> dict[str, Any]:
    """Tek kullanıcı mesajı için uçtan uca RAG cevabı üretir."""
    text = (message or "").strip()
    if not text:
        raise ValueError("Boş mesaj.")

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

    content_map = build_content_map(chunks_dir)
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

    answer = generate_answer(text, contexts)
    sources = [_hit_to_source(h, content_map) for h in top]

    # YELLOW: hekim uyarısı + grey-zone reason (şeffaflık MVP)
    if triage == "YELLOW":
        if decision.reason and decision.source == "grey_zone" and not decision.tempered:
            answer = answer + f"\n\n_Triage notu: {decision.reason}_"
        if "hekim" not in answer.casefold():
            answer = (
                answer
                + "\n\nNot: Anlattığınız durum yakın zamanda hekim değerlendirmesi gerektirebilir."
            )

    return {
        "answer": answer,
        "triage_level": triage,
        "sources": sources,
        "disclaimer": DEFAULT_DISCLAIMER,
        "follow_ups": FOLLOW_UPS,
    }
