"""Cross-encoder rerank — mmarco-mMiniLMv2.

Tercihler (kullanıcı):
  1A — retrieve top-10'un hepsini rerank et, sıralı top-10 dön
  2A — model her çağrıda yüklenir, bitince GPU boşaltılır
  3B — çıktıda sadece rerank listesi (retrieve skoru yok)

Ne yapmaz: SPLADE, LLM, füzyon reranker.
Önkoşul: indeks (embed build) + retrieve.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval.embed import CHUNKS_DIR, DEFAULT_INDEX_DIR, TOP_K, resolve_device
from src.retrieval.retrieve import (
    RetrievalHit,
    build_content_map,
    preview_text,
    retrieve,
)

# Kilit (docs/retrieval_decision.md)
RERANK_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MAX_DOC_CHARS = 1500
RERANK_BATCH = 2
PREVIEW_CHARS = 300


@dataclass
class RerankHit:
    """Tek rerank sonucu — sadece rerank skoru (tercih 3B)."""

    rank: int
    chunk_id: str
    score: float
    source: str
    preview: str


def gpu_free(*objs: Any) -> None:
    """Model referanslarını düşürüp CUDA cache temizler (tercih 2A)."""
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_reranker(model_id: str = RERANK_MODEL_ID, device: str | None = None):
    """CrossEncoder reranker modelini yükler."""
    from sentence_transformers import CrossEncoder

    dev = resolve_device(device)
    print(f"Reranker yükleniyor: {model_id} (device={dev})")
    return CrossEncoder(model_id, device=dev)


def hits_need_full_text(
    hits: list[RetrievalHit],
    content_map: dict[str, str],
) -> list[str]:
    """Her hit için rerank'e gidecek tam (kısaltılmış) chunk metnini üretir."""
    docs: list[str] = []
    for h in hits:
        body = content_map.get(h.chunk_id, h.preview or "")
        docs.append(body[:MAX_DOC_CHARS])
    return docs


def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    content_map: dict[str, str] | None = None,
    chunks_dir: Path = CHUNKS_DIR,
    top_k: int = TOP_K,
    device: str | None = None,
    model_id: str = RERANK_MODEL_ID,
) -> list[RerankHit]:
    """Retrieve adaylarını cross-encoder ile yeniden sıralar (tercih 1A).

    Tercih 2A: model burada yüklenir, predict sonrası boşaltılır.
    Tercih 3B: dönen listede retrieve skoru yoktur; score = rerank skoru.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Boş sorgu ile rerank yapılamaz.")
    if not hits:
        return []

    if content_map is None:
        content_map = build_content_map(chunks_dir)

    docs = hits_need_full_text(hits, content_map)
    pairs = [[q, doc] for doc in docs]

    model = load_reranker(model_id=model_id, device=device)
    try:
        raw_scores = model.predict(
            pairs,
            batch_size=RERANK_BATCH,
            show_progress_bar=False,
        )
        scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    finally:
        gpu_free(model)

    order = np.argsort(-scores)
    k = min(top_k, len(order))
    out: list[RerankHit] = []
    for new_rank, j in enumerate(order[:k].tolist(), start=1):
        h = hits[j]
        body = content_map.get(h.chunk_id, h.preview or "")
        out.append(
            RerankHit(
                rank=new_rank,
                chunk_id=h.chunk_id,
                score=round(float(scores[j]), 4),
                source=h.source,
                preview=preview_text(body, PREVIEW_CHARS),
            )
        )
    return out


def retrieve_and_rerank(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    top_k: int = TOP_K,
    device: str | None = None,
) -> list[RerankHit]:
    """Önce dense retrieve (top_k), sonra aynı adayları rerank eder.

    Kolay CLI/pipeline yardımcısı; çıktı yine sadece rerank listesi (3B).
    """
    content_map = build_content_map(chunks_dir)
    hits = retrieve(
        query=query,
        index_dir=index_dir,
        chunks_dir=chunks_dir,
        top_k=top_k,
        device=device,
        content_map=content_map,
    )
    return rerank_hits(
        query=query,
        hits=hits,
        content_map=content_map,
        chunks_dir=chunks_dir,
        top_k=top_k,
        device=device,
    )


def hits_to_dicts(hits: list[RerankHit]) -> list[dict[str, Any]]:
    """RerankHit listesini JSON-serileştirilebilir dict'e çevirir."""
    return [asdict(h) for h in hits]


def print_hits(query: str, hits: list[RerankHit]) -> None:
    """Rerank sonuçlarını terminale basar."""
    print(f"\nSorgu: {query}")
    print(f"Rerank top-{len(hits)} ({RERANK_MODEL_ID})\n")
    for h in hits:
        print(f"[{h.rank}] score={h.score:.4f}  {h.chunk_id}")
        print(f"    source : {h.source}")
        print(f"    preview: {h.preview}")
        print()


def main() -> None:
    """CLI: python -m src.retrieval.rerank \"şekerim kaç olmalı?\""""
    parser = argparse.ArgumentParser(
        description="Retrieve top-10 + mmarco rerank (çıktı: sadece rerank listesi)."
    )
    parser.add_argument("query", nargs="?", help="Aranacak soru")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu")
    parser.add_argument("--json", action="store_true", help="JSON çıktı")
    args = parser.parse_args()

    if not args.query:
        parser.error(
            'query gerekli. Örnek: python -m src.retrieval.rerank "Prediyabet nedir?"'
        )

    try:
        hits = retrieve_and_rerank(
            query=args.query,
            index_dir=args.index_dir,
            chunks_dir=args.chunks_dir,
            top_k=args.top_k,
            device=args.device,
        )
    except FileNotFoundError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        print("Önce: python -m src.retrieval.embed build", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(hits_to_dicts(hits), ensure_ascii=False, indent=2))
    else:
        print_hits(args.query, hits)


if __name__ == "__main__":
    main()
