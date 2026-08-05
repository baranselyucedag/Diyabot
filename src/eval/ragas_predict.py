#!/usr/bin/env python
"""Ragas aşama 1: gold sorular → predictions.jsonl.

Üretimle aynı yol: triage + retrieve + rerank + Nemotron (chat) cevap.
Hakem/Kimi burada YOK — skorlama src/eval/ragas_score.py'dedir.
Embedder/reranker bir kez yüklenir; CLI: --limit / --all.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.env import load_project_env
from src.api.llm import generate_answer
from src.api.pipeline import LLM_TOP_N
from src.api.triage import canned_response, detect_triage_detailed
from src.retrieval.embed import (
    CHUNKS_DIR,
    DEFAULT_INDEX_DIR,
    TOP_K,
    encode_query,
    load_embedder,
    load_index,
)
from src.retrieval.rerank import (
    MAX_DOC_CHARS,
    PREVIEW_CHARS,
    RERANK_BATCH,
    RERANK_MODEL_ID,
    RerankHit,
    load_reranker,
)
from src.retrieval.retrieve import (
    RetrievalHit,
    build_content_map,
    preview_text,
    score_query_against_index,
    topk_indices,
)

# dump.py paket dışı import kullanıyor; eval kökünü path'e ekle
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from dump import new_run_dir  # noqa: E402

GOLD_PATH = ROOT / "data" / "gold" / "gold_set.jsonl"
GUARDRAIL_TRIAGES = {"RED", "REFUSE", "EMERGENCY", "RED_REFUSE"}


def load_gold(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    """Gold set JSONL'ini okur; yalnızca curator_verified=true satırları alır.

    Retrieval eval ile aynı filtreyi kullanır ki Ragas sonuçları mevcut
    Hit@k / MRR benchmarklarıyla karşılaştırılabilir olsun.
    """
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Gold set yok: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("curator_verified"):
            continue
        rows.append(row)
    return rows


def select_gold(
    gold: list[dict[str, Any]],
    *,
    limit: int | None,
    all_items: bool,
) -> list[dict[str, Any]]:
    """CLI limit/all bayraklarına göre çalıştırılacak soru alt kümesini seçer.

    Varsayılan küçük bir smoke setidir; --all ile tüm küratör onaylı sorulara
    geçilir. Limit hem RAG hem guardrail satırlarına birlikte uygulanır.
    """
    if all_items or limit is None:
        return gold
    return gold[: max(0, limit)]


def warm_retrieve_and_rerank(
    query: str,
    *,
    embeddings: np.ndarray,
    meta: Any,
    content_map: dict[str, str],
    embedder: Any,
    reranker: Any,
    top_k: int = TOP_K,
) -> list[RerankHit]:
    """Önceden yüklenmiş embedder + reranker ile top-k retrieve→rerank yapar.

    run_chat'in her çağrıda modeli yeniden yüklemesini (tercih 2A) eval için
    atlar; aksi halde yüzlerce soruda GPU yükleme maliyeti dominant olur.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Boş sorgu.")

    q_vec = encode_query(embedder, q)
    scores = score_query_against_index(q_vec, embeddings)
    idxs = topk_indices(scores, top_k)

    hits: list[RetrievalHit] = []
    for rank, i in enumerate(idxs.tolist(), start=1):
        cid = meta.chunk_ids[i]
        src = meta.sources[i] if i < len(meta.sources) else ""
        body = content_map.get(cid, "")
        hits.append(
            RetrievalHit(
                rank=rank,
                chunk_id=cid,
                score=round(float(scores[i]), 4),
                source=src,
                preview=preview_text(body),
            )
        )

    if not hits:
        return []

    docs = [
        (content_map.get(h.chunk_id, h.preview or "") or "")[:MAX_DOC_CHARS]
        for h in hits
    ]
    pairs = [[q, doc] for doc in docs]
    raw = reranker.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
    rr_scores = np.asarray(raw, dtype=np.float64).reshape(-1)
    order = np.argsort(-rr_scores)
    k = min(top_k, len(order))

    out: list[RerankHit] = []
    for new_rank, j in enumerate(order[:k].tolist(), start=1):
        h = hits[j]
        body = content_map.get(h.chunk_id, h.preview or "")
        out.append(
            RerankHit(
                rank=new_rank,
                chunk_id=h.chunk_id,
                score=round(float(rr_scores[j]), 4),
                source=h.source,
                preview=preview_text(body, PREVIEW_CHARS),
            )
        )
    return out


def predict_one(
    gold_row: dict[str, Any],
    *,
    embeddings: np.ndarray,
    meta: Any,
    content_map: dict[str, str],
    embedder: Any,
    reranker: Any,
    llm_top_n: int = LLM_TOP_N,
    retrieve_k: int = TOP_K,
) -> dict[str, Any]:
    """Tek gold sorusu için üretimle aynı cevabı üretir ve Ragas satırı döner.

    EMERGENCY/RED'de canned şablon kullanılır (retrieval atlanır). Diğerlerinde
    top-3 context Nemotron'a gider; ranked_chunk_ids top-10 olarak saklanır.
    """
    question = (gold_row.get("question") or "").strip()
    expected_triage = str(gold_row.get("expected_triage") or "GREEN")
    decision = detect_triage_detailed(question)
    detected = decision.level
    canned = canned_response(
        detected,
        flags=decision.flags,
        tempered=decision.tempered,
        reason=decision.reason if decision.tempered else None,
    )

    base: dict[str, Any] = {
        "gold_id": gold_row.get("id"),
        "question": question,
        "reference": gold_row.get("expected_answer_summary") or "",
        "expected_chunk_ids": list(gold_row.get("expected_chunk_ids") or []),
        "expected_triage": expected_triage,
        "detected_triage": detected,
        "must_include": list(gold_row.get("must_include") or []),
        "must_not_include": list(gold_row.get("must_not_include") or []),
        "category": gold_row.get("category"),
        "paraphrase_of": gold_row.get("paraphrase_of"),
        "safety_critical": bool(gold_row.get("safety_critical")),
        "is_guardrail": expected_triage in GUARDRAIL_TRIAGES
        or detected in {"RED", "REFUSE", "EMERGENCY"},
    }

    if canned is not None:
        base.update(
            {
                "answer": canned,
                "retrieved_contexts": [],
                "ranked_chunk_ids": [],
                "skipped_rag": True,
            }
        )
        return base

    hits = warm_retrieve_and_rerank(
        question,
        embeddings=embeddings,
        meta=meta,
        content_map=content_map,
        embedder=embedder,
        reranker=reranker,
        top_k=retrieve_k,
    )
    ranked_ids = [h.chunk_id for h in hits]
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
    context_texts = [
        (c.get("content") or c.get("preview") or "") for c in contexts
    ]

    if not contexts:
        answer = (
            "Bu konuda doğrulanmış eğitim kaynağımda net bir eşleşme bulamadım. "
            "Kişisel kararlar için hekiminize danışın."
        )
    else:
        answer = generate_answer(question, contexts)

    if detected == "YELLOW" and "hekim" not in answer.casefold():
        answer = (
            answer
            + "\n\nNot: Anlattığınız durum yakın zamanda hekim değerlendirmesi gerektirebilir."
        )

    base.update(
        {
            "answer": answer,
            "retrieved_contexts": context_texts,
            "ranked_chunk_ids": ranked_ids,
            "skipped_rag": False,
        }
    )
    return base


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    """Prediction satırlarını JSONL olarak yazar (UTF-8, satır başına bir kayıt).

    Sonraki ragas_score aşaması bu dosyayı okur; metrik değişince tekrar
    cevap üretmeye gerek kalmaz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_predict(
    *,
    limit: int | None = 10,
    all_items: bool = False,
    device: str | None = None,
    workers: int = 3,
    out_dir: Path | None = None,
) -> Path:
    """Gold → predictions.jsonl uçtan uca koşusu; çıktı klasör yolunu döner.

    Embedder/reranker bir kez yüklenir; Nemotron çağrıları workers ile
    paralel yapılır (NVIDIA free tier ~40 RPM dikkate alınmalı).
    """
    load_project_env()
    gold = select_gold(load_gold(), limit=limit, all_items=all_items)
    if not gold:
        raise SystemExit("Küratör onaylı gold soru yok.")

    run_dir = out_dir or new_run_dir("ragas")
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "predictions.jsonl"

    print(f"Gold soru: {len(gold)}")
    print(f"Çıktı    : {run_dir}")

    embeddings, meta = load_index(DEFAULT_INDEX_DIR)
    content_map = build_content_map(CHUNKS_DIR)
    print("Embedder yükleniyor (warm)...")
    embedder = load_embedder(model_id=meta.model_id, device=device)
    print(f"Reranker yükleniyor (warm): {RERANK_MODEL_ID}")
    reranker = load_reranker(model_id=RERANK_MODEL_ID, device=device)

    # Retrieval GPU'da sıralı; LLM I/O bound → thread pool
    # Önce tüm retrieval'ları sırayla yap, sonra LLM'leri paralel üret.
    pending_llm: list[tuple[dict[str, Any], list[dict[str, Any]], list[str]]] = []
    done_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, g in enumerate(gold, start=1):
        question = (g.get("question") or "").strip()
        decision = detect_triage_detailed(question)
        detected = decision.level
        canned = canned_response(
            detected,
            flags=decision.flags,
            tempered=decision.tempered,
            reason=decision.reason if decision.tempered else None,
        )
        expected_triage = str(g.get("expected_triage") or "GREEN")

        base: dict[str, Any] = {
            "gold_id": g.get("id"),
            "question": question,
            "reference": g.get("expected_answer_summary") or "",
            "expected_chunk_ids": list(g.get("expected_chunk_ids") or []),
            "expected_triage": expected_triage,
            "detected_triage": detected,
            "must_include": list(g.get("must_include") or []),
            "must_not_include": list(g.get("must_not_include") or []),
            "category": g.get("category"),
            "paraphrase_of": g.get("paraphrase_of"),
            "safety_critical": bool(g.get("safety_critical")),
            "is_guardrail": expected_triage in GUARDRAIL_TRIAGES
            or detected in {"RED", "REFUSE", "EMERGENCY"},
        }

        if canned is not None:
            base.update(
                {
                    "answer": canned,
                    "retrieved_contexts": [],
                    "ranked_chunk_ids": [],
                    "skipped_rag": True,
                }
            )
            done_rows.append(base)
            print(f"  [{i}/{len(gold)}] {g.get('id')} guardrail={detected}")
            continue

        hits = warm_retrieve_and_rerank(
            question,
            embeddings=embeddings,
            meta=meta,
            content_map=content_map,
            embedder=embedder,
            reranker=reranker,
            top_k=TOP_K,
        )
        ranked_ids = [h.chunk_id for h in hits]
        top = hits[: max(1, LLM_TOP_N)]
        contexts = [
            {
                "chunk_id": h.chunk_id,
                "source": h.source,
                "preview": h.preview,
                "content": content_map.get(h.chunk_id, h.preview),
            }
            for h in top
        ]
        pending_llm.append((base, contexts, ranked_ids))
        print(f"  [{i}/{len(gold)}] {g.get('id')} retrieve OK → LLM kuyruğu")

    def _gen(item: tuple[dict[str, Any], list[dict[str, Any]], list[str]]) -> dict[str, Any]:
        base, contexts, ranked_ids = item
        context_texts = [(c.get("content") or c.get("preview") or "") for c in contexts]
        if not contexts:
            answer = (
                "Bu konuda doğrulanmış eğitim kaynağımda net bir eşleşme bulamadım. "
                "Kişisel kararlar için hekiminize danışın."
            )
        else:
            answer = generate_answer(base["question"], contexts)
        if base["detected_triage"] == "YELLOW" and "hekim" not in answer.casefold():
            answer += (
                "\n\nNot: Anlattığınız durum yakın zamanda hekim değerlendirmesi gerektirebilir."
            )
        out = {
            **base,
            "answer": answer,
            "retrieved_contexts": context_texts,
            "ranked_chunk_ids": ranked_ids,
            "skipped_rag": False,
        }
        return out

    print(f"\nLLM üretimi başlıyor ({len(pending_llm)} soru, workers={workers})...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_gen, item): item[0].get("gold_id") for item in pending_llm}
        for fut in as_completed(futures):
            gid = futures[fut]
            try:
                row = fut.result()
                done_rows.append(row)
                print(f"  LLM OK: {gid}")
            except Exception as exc:  # noqa: BLE001
                print(f"  LLM HATA ({gid}): {exc}", file=sys.stderr)
                # Hatalı satırı boş cevapla işaretle ki score aşaması atlayabilsin
                base_fail = next(b for b, _, _ in pending_llm if b.get("gold_id") == gid)
                done_rows.append(
                    {
                        **base_fail,
                        "answer": "",
                        "retrieved_contexts": [],
                        "ranked_chunk_ids": [],
                        "skipped_rag": False,
                        "error": str(exc),
                    }
                )

    # Orijinal gold sırasını koru
    order = {g.get("id"): i for i, g in enumerate(gold)}
    done_rows.sort(key=lambda r: order.get(r.get("gold_id"), 10**9))

    write_predictions(pred_path, done_rows)
    meta_out = {
        "n_gold": len(gold),
        "n_predictions": len(done_rows),
        "n_guardrail": sum(1 for r in done_rows if r.get("skipped_rag")),
        "n_rag": sum(1 for r in done_rows if not r.get("skipped_rag")),
        "llm_top_n": LLM_TOP_N,
        "retrieve_k": TOP_K,
        "rerank_model": RERANK_MODEL_ID,
        "embed_model": meta.model_id,
        "sec": round(time.perf_counter() - t0, 1),
    }
    (run_dir / "predict_meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTamam — {pred_path}")
    print(json.dumps(meta_out, ensure_ascii=False, indent=2))
    return run_dir


def main() -> None:
    """CLI girişi: python -m src.eval.ragas_predict [--limit 10 | --all]."""
    parser = argparse.ArgumentParser(
        description="Ragas aşama 1: gold → predictions.jsonl (RAG + Nemotron)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Kaç gold soru işlensin (varsayılan 10). --all ile yok sayılır.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tüm curator_verified gold sorularını işle.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Paralel Nemotron çağrı sayısı (free tier RPM'e dikkat).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Çıktı klasörü (verilmezse data/eval_results/ragas_<stamp>).",
    )
    args = parser.parse_args()
    run_predict(
        limit=None if args.all else args.limit,
        all_items=args.all,
        device=args.device,
        workers=args.workers,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
