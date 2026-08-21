"""Pipeline entegrasyon testleri (Faz 4).

Kapsam (plan §7):
- run_chat history ile çalışır
- run_chat gerçek triage_level döndürür
- run_chat hafızaya tur kaydeder (create_turn_atomic + append_turn_and_update_index)
- run_chat memory_maintenance_task'ı needs_summary parametresiyle başlatır
- build_user_prompt_with_memory prompt formatı (SYSTEM_PROMPT gömülmez)

Retrieval ve LLM çağrıları monkeypatch ile sahte yapılır; ağ/model erişimi YOKTUR.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import src.api.pipeline as pipeline
from src.api.memory.models import (
    MemoryIndex,
    Note,
    NotesStore,
    NoteStaleness,
    PatientProfile,
    Profile,
    ProfileMetadata,
    Turn,
    TurnRole,
)
from src.api.memory.storage import load_index, load_turns, save_index
from src.api.triage import detect_triage_detailed


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------


def _fake_hit(chunk_id: str, source: str, preview: str) -> SimpleNamespace:
    return SimpleNamespace(chunk_id=chunk_id, source=source, preview=preview)


def _fake_task_factory(calls: list[dict]):
    """memory_maintenance_task yerine geçen sahte: çağrı argümanlarını kaydeder.

    Normal bir fonksiyon olduğu için argümanlar `asyncio.create_task`'a verilmeden
    ÖNCE (çağrı anında) kaydedilir; dönen coroutine'in çalışıp çalışmaması testi
    etkilemez.
    """

    def fake_task(conv_id: str, turns: list[Turn], needs_summary: bool = False):
        calls.append({"conv_id": conv_id, "turns": turns, "needs_summary": needs_summary})

        async def _coro() -> dict:
            return {"ok": True}

        return _coro()

    return fake_task


def _install_pipeline_mocks(monkeypatch, task_calls: list[dict] | None = None):
    """Retrieval + LLM + maintenance'ı sahteler; üçüncü taraf/network erişimi yok."""
    monkeypatch.setattr(pipeline, "build_content_map", lambda chunks_dir: {})
    monkeypatch.setattr(pipeline, "build_section_map", lambda chunks_dir: {})
    monkeypatch.setattr(
        pipeline,
        "retrieve_and_rerank",
        lambda **kwargs: [_fake_hit("c1", "rehber.pdf", "kaynak içeriği")],
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda question, contexts, **kw: "cevap metni",
    )
    if task_calls is not None:
        monkeypatch.setattr(pipeline, "memory_maintenance_task", _fake_task_factory(task_calls))


def _make_ctx():
    profile = Profile(
        version=1,
        patient=PatientProfile(diabetes_type="Type 2", diagnosis_year=2022),
        metadata=ProfileMetadata(),
    )
    notes = NotesStore(
        items=[
            Note(
                note_id="n1",
                content="Metformin kullanıyorum",
                category="observation",
                staleness=NoteStaleness.FRESH,
            )
        ]
    )
    return {
        "profile": profile,
        "summary": "özet metni",
        "notes": notes,
        "pending_conflicts": None,
        "index": MemoryIndex(turn_count=0),
    }


# ---------------------------------------------------------------------------
# build_user_prompt_with_memory
# ---------------------------------------------------------------------------


def test_build_user_prompt_with_memory_format_and_no_system_prompt():
    ctx = _make_ctx()
    turns = [Turn(turn_id="msg_1", role=TurnRole.USER, content="Metformin kullanıyorum")]
    contexts = [{"chunk_id": "c1", "source": "rehber.pdf", "content": "kaynak içeriği"}]

    prompt = pipeline.build_user_prompt_with_memory("Prediyabet nedir?", contexts, ctx, turns)

    # Hafıza bileşenleri var
    assert "### PROFİL" in prompt
    assert "Type 2" in prompt
    assert "özet metni" in prompt
    assert "Metformin kullanıyorum" in prompt
    # RAG kaynakları var
    assert "### KAYNAKLAR" in prompt
    assert "kaynak içeriği" in prompt
    # Soru var
    assert "### SORU" in prompt
    assert "Prediyabet nedir?" in prompt
    # SYSTEM_PROMPT kullanıcı mesajına gömülmemeli (ayrı system mesajı)
    assert "Kesin Yasaklar" not in prompt
    assert "## 1. Kimlik ve Kapsam" not in prompt


def test_build_user_prompt_with_memory_empty_notes_and_no_pending():
    ctx = _make_ctx()
    ctx["notes"] = NotesStore(items=[])
    ctx["pending_conflicts"] = None
    ctx["summary"] = ""

    prompt = pipeline.build_user_prompt_with_memory("Soru", [], ctx, [])

    assert "(ilgili not yok)" in prompt
    assert "(özet yok)" in prompt
    assert "(son tur yok)" in prompt
    assert "(kaynak yok)" in prompt
    # pending conflict bölümü olmamalı
    assert "BEKLEYEN NETLEŞTİRMELER" not in prompt


# ---------------------------------------------------------------------------
# run_chat
# ---------------------------------------------------------------------------


def test_run_chat_returns_correct_shape_and_real_triage(monkeypatch):
    _install_pipeline_mocks(monkeypatch, task_calls=[])

    message = "Prediyabet nedir?"
    result = asyncio.run(pipeline.run_chat(message, "conv_p1"))

    assert result["answer"] == "cevap metni"
    assert result["triage_level"] == detect_triage_detailed(message).level
    assert result["triage_level"] == "GREEN"
    assert result["sources"][0]["document"] == "rehber.pdf"
    assert result["disclaimer"]
    assert result["follow_ups"]


def test_run_chat_records_turns_atomically(monkeypatch):
    _install_pipeline_mocks(monkeypatch, task_calls=[])

    asyncio.run(pipeline.run_chat("Prediyabet nedir?", "conv_p2"))

    index = load_index("conv_p2")
    assert index is not None
    assert index.turn_count == 2

    turns = load_turns("conv_p2")
    assert [t.role for t in turns] == [TurnRole.USER, TurnRole.ASSISTANT]
    assert turns[0].content == "Prediyabet nedir?"
    assert turns[1].content == "cevap metni"
    assert turns[0].turn_id == "msg_1"
    assert turns[1].turn_id == "msg_2"  # user msg_1 → assistant msg_2


def test_run_chat_triggers_maintenance_task_with_needs_summary(monkeypatch):
    calls: list[dict] = []
    _install_pipeline_mocks(monkeypatch, task_calls=calls)

    asyncio.run(pipeline.run_chat("Prediyabet nedir?", "conv_p3"))

    assert len(calls) == 1
    assert calls[0]["conv_id"] == "conv_p3"
    assert calls[0]["needs_summary"] is False  # (0 + 2) % 5 != 0
    assert len(calls[0]["turns"]) == 2  # user + assistant


def test_run_chat_needs_summary_true_when_hitting_multiple_of_5(monkeypatch):
    # index.turn_count = 3 → bu turla birlikte 5 olur → needs_summary True
    save_index("conv_p4", MemoryIndex(turn_count=3))

    calls: list[dict] = []
    _install_pipeline_mocks(monkeypatch, task_calls=calls)

    asyncio.run(pipeline.run_chat("Prediyabet nedir?", "conv_p4"))

    assert calls[0]["needs_summary"] is True


def test_run_chat_with_history_uses_provided_turns(monkeypatch):
    calls: list[dict] = []
    _install_pipeline_mocks(monkeypatch, task_calls=calls)

    history = [
        {"turn_id": "msg_1", "role": "user", "content": "önceki mesaj", "timestamp": "2026-01-01T00:00:00Z"},
    ]

    result = asyncio.run(
        pipeline.run_chat("Prediyabet nedir?", "conv_p5", history=history)
    )

    assert result["answer"] == "cevap metni"
    # Maintenance'a geçen tur listesi: history (1) + user + assistant = 3
    assert len(calls[0]["turns"]) == 3
    assert calls[0]["turns"][0].content == "önceki mesaj"


def test_run_chat_empty_message_raises(monkeypatch):
    with pytest.raises(ValueError):
        asyncio.run(pipeline.run_chat("   ", "conv_p6"))


# ---------------------------------------------------------------------------
# _hit_to_source + build_section_map (kaynak gösterimi — chunk_id yerine bölüm)
# ---------------------------------------------------------------------------


def test_hit_to_source_section_label_combined():
    hit = _fake_hit("c1", "rehber.pdf", "")
    content_map = {"c1": "### Bölüm X > Alt Y\n\niçerik"}
    section_map = {"c1": "Bölüm X > Alt Y"}

    out = pipeline._hit_to_source(hit, content_map, section_map)

    assert out["section_label"] == "Bölüm X > Alt Y"
    assert out["section"] == "c1"  # debug chunk_id korunuyor
    assert out["document"] == "rehber.pdf"


def test_hit_to_source_section_label_only_section():
    hit = _fake_hit("c2", "rehber.pdf", "")
    content_map = {"c2": "### ÖNSÖZ\n\niçerik"}
    section_map = {"c2": "ÖNSÖZ"}

    out = pipeline._hit_to_source(hit, content_map, section_map)

    assert out["section_label"] == "ÖNSÖZ"


def test_hit_to_source_section_label_none_when_missing():
    hit = _fake_hit("c3", "rehber.pdf", "")
    content_map = {"c3": "içerik"}
    section_map = {"c3": None}

    out = pipeline._hit_to_source(hit, content_map, section_map)

    assert out["section_label"] is None
    assert out["section"] == "c3"  # debug alanı hâlâ dolu (chunk_id)
    assert "section_label" in out  # alan her zaman var, değer None olabilir


def test_build_section_map_fallback_chain(monkeypatch):
    import src.retrieval.retrieve as retrieve

    monkeypatch.setattr(
        retrieve,
        "load_chunk_records",
        lambda chunks_dir: [
            {"chunk_id": "a", "content": "x", "section_path": "Bölüm 1 > Alt", "section": "Alt", "chapter": "Bölüm 1"},
            {"chunk_id": "b", "content": "y", "section_path": "ÖNSÖZ", "section": "ÖNSÖZ", "chapter": None},
            {"chunk_id": "c", "content": "z"},  # section_path + section yok
        ],
    )

    m = retrieve.build_section_map()

    assert m["a"] == "Bölüm 1 > Alt"
    assert m["b"] == "ÖNSÖZ"
    assert m["c"] is None  # chunk_id'ye ASLA düşmez
