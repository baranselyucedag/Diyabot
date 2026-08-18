"""Üretim embedding katmanı — sadece BAAI/bge-m3.

Ne yapar / ne yapmaz:
  - Chunk metinlerini vektöre çevirir, diske yazar (npy + meta).
  - Sorgu vektörü üretir.
  - Arama (top-k), rerank, LLM YOK — sonraki adımlar.

Kilit (docs/retrieval_decision.md):
  model = BAAI/bge-m3
  aday havuzu (sonraki retrieve adımı) = top-10
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_DIR = ROOT / "data" / "processed"
# Varsayılan indeks çıktısı
DEFAULT_INDEX_DIR = ROOT / "data" / "index" / "bge-m3"

# Kilitlenen model (değiştirme — karar dosyasına bak)
EMBED_MODEL_ID = "BAAI/bge-m3"
# Sonraki retrieve adımında kullanılacak aday sayısı (kullanıcı tercihi)
TOP_K = 10
# 4GB VRAM için küçük batch
DEFAULT_BATCH_SIZE = 4


@dataclass
class EmbedIndexMeta:
    """İndeks yanındaki meta.json içeriği."""

    model_id: str
    n_chunks: int
    dim: int
    chunk_ids: list[str]
    # İsteğe bağlı: her chunk için kısa kaynak etiketi
    sources: list[str]
    created_unix: float
    normalize: bool = True
    top_k_default: int = TOP_K


def load_chunk_records(
    chunks_dir: Path = CHUNKS_DIR,
) -> list[dict[str, Any]]:
    """data/processed/*.chunks.jsonl dosyalarını okur; sıralı kayıt listesi döner.

    Her kayıt en az chunk_id + content taşır; source varsa meta'ya yazılır.
    """
    records: list[dict[str, Any]] = []
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunk klasörü yok: {chunks_dir}")

    paths = sorted(chunks_dir.glob("*.chunks.jsonl"))
    if not paths:
        raise FileNotFoundError(f"*.chunks.jsonl bulunamadı: {chunks_dir}")

    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "chunk_id" not in row or "content" not in row:
                raise ValueError(f"Eksik alan (chunk_id/content): {path.name}")
            records.append(row)
    return records


def resolve_device(device: str | None = None) -> str:
    """Kullanılacak cihazı seçer: 'cuda' | 'cpu'. None ise CUDA varsa cuda."""
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_embedder_cache: dict[tuple[str, str], Any] = {}
_embedder_lock = threading.Lock()


def load_embedder(model_id: str = EMBED_MODEL_ID, device: str | None = None):
    """SentenceTransformer ile bge-m3 modelini yükler ve döner.

    Singleton: aynı (model, device) için tek instance döner. 4GB GPU'da aynı
    modelin iki kez yüklenmesi (triage + retrieval) OOM yapıyordu; bu cache
    hem implicit_score hem retrieve hem grey_model'in AYNI instance'ı paylaşmasını
    sağlar. Caller'ın gpu_free yapması cache'i etkilemez (model yaşar).
    """
    from sentence_transformers import SentenceTransformer

    dev = resolve_device(device)
    key = (model_id, dev)
    with _embedder_lock:
        if key in _embedder_cache:
            return _embedder_cache[key]
        print(f"Embedding modeli yükleniyor: {model_id} (device={dev})")
        model = SentenceTransformer(model_id, device=dev)
        _embedder_cache[key] = model
        return model


def encode_texts(
    model,
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Metin listesini L2-normalize edilmiş float32 vektör matrisine çevirir.

    Satır i = texts[i] vektörü. Cosine benzerlik = normalize edilmiş dot product.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_query(
    model,
    query: str,
    batch_size: int = 1,
) -> np.ndarray:
    """Tek bir kullanıcı sorusunu (1, dim) şekilli normalize vektöre çevirir.

    Retrieve adımında chunk matrisi ile çarpılmak üzere hazırlanır.
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("Boş sorgu embed edilemez.")
    vec = encode_texts(model, [q], batch_size=batch_size, show_progress=False)
    return vec  # shape (1, dim)


def build_index(
    chunks_dir: Path = CHUNKS_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_id: str = EMBED_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> EmbedIndexMeta:
    """Tüm chunk'ları embed edip index_dir altına kaydeder.

    Çıktılar:
      - embeddings.npy  : (N, dim) float32, satır = chunk
      - meta.json       : chunk_ids, model, dim, top_k_default=10, ...
    """
    records = load_chunk_records(chunks_dir)
    chunk_ids = [r["chunk_id"] for r in records]
    texts = [r["content"] for r in records]
    sources = [str(r.get("source") or r.get("document_id") or "") for r in records]

    print(f"Chunk sayısı: {len(texts)}")
    t0 = time.perf_counter()
    model = load_embedder(model_id=model_id, device=device)
    matrix = encode_texts(model, texts, batch_size=batch_size, show_progress=True)
    elapsed = time.perf_counter() - t0
    print(f"Encoding bitti: {matrix.shape}  ({elapsed:.1f}s)")

    # GPU boşalt (4GB cihazlarda sonraki adımlar için)
    try:
        import gc

        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    meta = EmbedIndexMeta(
        model_id=model_id,
        n_chunks=int(matrix.shape[0]),
        dim=int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        chunk_ids=chunk_ids,
        sources=sources,
        created_unix=time.time(),
        normalize=True,
        top_k_default=TOP_K,
    )
    save_index(index_dir, matrix, meta)
    return meta


def save_index(
    index_dir: Path,
    embeddings: np.ndarray,
    meta: EmbedIndexMeta,
) -> None:
    """embeddings.npy + meta.json dosyalarını yazar (klasör yoksa oluşturur)."""
    index_dir.mkdir(parents=True, exist_ok=True)
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"

    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    np.save(emb_path, embeddings)
    meta_path.write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Kaydedildi: {emb_path}")
    print(f"Kaydedildi: {meta_path}")


def load_index(
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> tuple[np.ndarray, EmbedIndexMeta]:
    """Diskteki embeddings.npy + meta.json çiftini okuyup (matris, meta) döner."""
    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"İndeks eksik: {index_dir} (önce: python -m src.retrieval.embed build)"
        )

    matrix = np.load(emb_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = EmbedIndexMeta(**raw)

    if matrix.shape[0] != meta.n_chunks:
        raise ValueError(
            f"Satır sayısı uyuşmuyor: npy={matrix.shape[0]} meta.n_chunks={meta.n_chunks}"
        )
    if len(meta.chunk_ids) != meta.n_chunks:
        raise ValueError("meta.chunk_ids uzunluğu n_chunks ile uyuşmuyor")
    return matrix, meta


def main() -> None:
    """CLI: build | info — sadece embedding indeks işlemleri."""
    parser = argparse.ArgumentParser(
        description="bge-m3 embedding indeksi (top_k_default=10)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Chunk'ları embed edip diske yaz.")
    p_build.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    p_build.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    p_build.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_build.add_argument("--device", type=str, default=None, help="cuda | cpu")

    p_info = sub.add_parser("info", help="Mevcut indeks özetini yazdır.")
    p_info.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)

    args = parser.parse_args()

    if args.cmd == "build":
        meta = build_index(
            chunks_dir=args.chunks_dir,
            index_dir=args.index_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
        print(
            f"\nOK — {meta.n_chunks} chunk, dim={meta.dim}, "
            f"top_k_default={meta.top_k_default}, model={meta.model_id}"
        )
    elif args.cmd == "info":
        matrix, meta = load_index(args.index_dir)
        print(f"index_dir : {args.index_dir}")
        print(f"model     : {meta.model_id}")
        print(f"shape     : {matrix.shape}")
        print(f"n_chunks  : {meta.n_chunks}")
        print(f"dim       : {meta.dim}")
        print(f"top_k     : {meta.top_k_default}")
        print(f"normalize : {meta.normalize}")


if __name__ == "__main__":
    main()
