"""Dense retrieval — BAAI/bge-m3, top-10.

Tercihler (kullanıcı):
  1A — sonuç: chunk_id, skor, source, content önizleme (~300 karakter)
  2A — CLI'da model her sorguda yüklenir, bitince GPU boşaltılır

Ne yapmaz: SPLADE/hibrit, rerank, LLM.
Önkoşul: python -m src.retrieval.embed build
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

# Paket içi import (python -m src.retrieval.retrieve)
from src.retrieval.embed import (
    CHUNKS_DIR,
    DEFAULT_INDEX_DIR,
    TOP_K,
    encode_query,
    load_chunk_records,
    load_embedder,
    load_index,
)

PREVIEW_CHARS = 300


@dataclass
class RetrievalHit:
    """Tek bir retrieval sonucu (tercih 1A alanları)."""

    rank: int
    chunk_id: str
    score: float
    source: str
    preview: str


def preview_text(text: str, n: int = PREVIEW_CHARS) -> str:
    """Chunk metnini tek satıra sıkıştırıp ilk n karakteri alır."""
    flat = " ".join((text or "").split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1] + "…"


def build_content_map(
    chunks_dir: Path = CHUNKS_DIR,
) -> dict[str, str]:
    """chunk_id → tam content sözlüğü (önizleme için)."""
    return {r["chunk_id"]: r["content"] for r in load_chunk_records(chunks_dir)}


def gpu_free(*objs: Any) -> None:
    """Model/tensor referanslarını düşürüp CUDA önbelleğini temizler (tercih 2A)."""
    for obj in objs:
        del obj
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def score_query_against_index(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Normalize edilmiş sorgu (1, dim) ile indeks (N, dim) cosine skorlarını üretir.

    Skor = dot product (vektörler L2-normalize varsayılır).
    """
    if query_vec.ndim != 2 or query_vec.shape[0] != 1:
        raise ValueError(f"query_vec şekli (1, dim) olmalı, gelen: {query_vec.shape}")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 2D olmalı, gelen: {embeddings.shape}")
    if query_vec.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"dim uyuşmazlığı: query={query_vec.shape[1]} index={embeddings.shape[1]}"
        )
    # (1, dim) @ (dim, N) -> (1, N) -> (N,)
    return (query_vec @ embeddings.T).reshape(-1)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Skor vektöründen en yüksek k indeksi azalan sırada döner."""
    n = len(scores)
    if n == 0:
        return np.array([], dtype=np.int64)
    k = min(k, n)
    if k == n:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def retrieve(
    query: str,
    index_dir: Path = DEFAULT_INDEX_DIR,
    chunks_dir: Path = CHUNKS_DIR,
    top_k: int = TOP_K,
    device: str | None = None,
    content_map: dict[str, str] | None = None,
    embeddings: np.ndarray | None = None,
    meta=None,
) -> list[RetrievalHit]:
    """Sorguyu bge-m3 ile gömüp indeks üzerinde top-k chunk döner.

    Tercih 2A: model burada yüklenir, encode sonrası GPU boşaltılır.
    embeddings/meta verilmezse diskten load_index çağrılır.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Boş sorgu ile arama yapılamaz.")

    if embeddings is None or meta is None:
        embeddings, meta = load_index(index_dir)

    if content_map is None:
        content_map = build_content_map(chunks_dir)

    model = load_embedder(model_id=meta.model_id, device=device)
    try:
        q_vec = encode_query(model, q)
    finally:
        gpu_free(model)

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
    return hits


def hits_to_dicts(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    """RetrievalHit listesini JSON-serileştirilebilir dict listesine çevirir."""
    return [asdict(h) for h in hits]


def print_hits(query: str, hits: list[RetrievalHit]) -> None:
    """Sonuçları terminale okunaklı basar."""
    print(f"\nSorgu: {query}")
    print(f"Top-{len(hits)} (dense bge-m3)\n")
    for h in hits:
        print(f"[{h.rank}] score={h.score:.4f}  {h.chunk_id}")
        print(f"    source : {h.source}")
        print(f"    preview: {h.preview}")
        print()


def main() -> None:
    """CLI: python -m src.retrieval.retrieve \"şekerim kaç olmalı?\""""
    parser = argparse.ArgumentParser(
        description="Dense retrieval (bge-m3, top-10). Rerank yok."
    )
    parser.add_argument("query", nargs="?", help="Aranacak soru metni")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Sonuçları JSON satırı olarak yaz.",
    )
    args = parser.parse_args()

    if not args.query:
        parser.error("query gerekli. Örnek: python -m src.retrieval.retrieve \"Prediyabet nedir?\"")

    try:
        hits = retrieve(
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
