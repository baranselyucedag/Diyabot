#!/usr/bin/env python
"""Klasik IR vs öğrenilmiş sparse vs hybrid karşılaştırması.

Kolonlar:
  A) BM25F              — klasik sparse
  B) SPLADE             — öğrenilmiş sparse
  C) BM25F + bge-m3     — hybrid (RRF)
  D) SPLADE + bge-m3    — hybrid (RRF)
  (+ dense-only satırı rapor için)

Gold: data/gold/gold_set.jsonl (sadece curator_verified).
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dump import attach_ci_to_row, new_run_dir, write_per_query, write_summary  # noqa: E402
from embed_benchmark import load_chunks, load_gold  # noqa: E402
from metrics import KS, mean_metrics, per_query_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
DENSE_MODEL = "BAAI/bge-m3"

BM25F_WEIGHTS = {"title": 2.0, "body": 1.0}
RRF_K = 60
CANDIDATE_POOL = 50

TOKEN_RE = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", re.UNICODE)
HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)

SPLADE_BATCH = 4
DENSE_BATCH = 4


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def split_fields(content: str) -> dict[str, str]:
    text = content or ""
    match = HEADING_RE.search(text)
    if match:
        title = match.group(1).strip()
        body = (text[: match.start()] + text[match.end() :]).strip()
    else:
        lines = text.splitlines()
        title = lines[0].strip() if lines else ""
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else text
    return {"title": title, "body": body}


class BM25FIndex:
    def __init__(self, texts: list[str], weights: dict[str, float] | None = None):
        self.weights = weights or BM25F_WEIGHTS
        fields = [split_fields(t) for t in texts]
        self.title_tokens = [tokenize(f["title"]) for f in fields]
        self.body_tokens = [tokenize(f["body"]) for f in fields]
        self.title_tokens = [toks or ["_empty_"] for toks in self.title_tokens]
        self.body_tokens = [toks or ["_empty_"] for toks in self.body_tokens]
        self.bm25_title = BM25Okapi(self.title_tokens)
        self.bm25_body = BM25Okapi(self.body_tokens)

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query) or ["_empty_"]
        s = self.weights["title"] * np.asarray(
            self.bm25_title.get_scores(q), dtype=np.float64
        )
        s += self.weights["body"] * np.asarray(
            self.bm25_body.get_scores(q), dtype=np.float64
        )
        return s

    def topk(self, query: str, k: int) -> list[int]:
        scores = self.scores(query)
        if k >= len(scores):
            return list(np.argsort(-scores))
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        return list(idx[np.argsort(-scores[idx])])


def gpu_free(*objs: Any) -> None:
    import gc

    import torch

    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _topk_from_scores(scores: np.ndarray, k: int) -> list[int]:
    if k >= len(scores):
        return list(np.argsort(-scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return list(idx[np.argsort(-scores[idx])])


def _to_numpy_2d(emb: Any) -> np.ndarray:
    if hasattr(emb, "to_dense"):
        emb = emb.to_dense()
    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().float().numpy()
    return np.asarray(emb, dtype=np.float32)


def run_splade_rankings(
    model_id: str,
    doc_texts: list[str],
    queries: list[str],
    k: int,
    device: str,
) -> tuple[list[list[int]], float]:
    from sentence_transformers import SparseEncoder

    t0 = time.perf_counter()
    print(f"  SPLADE yükleniyor: {model_id}")
    model = SparseEncoder(model_id, device=device)

    print("  SPLADE doküman encoding...")
    doc_emb = model.encode_document(
        doc_texts,
        batch_size=SPLADE_BATCH,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    print("  SPLADE sorgu encoding...")
    q_emb = model.encode_query(
        queries,
        batch_size=SPLADE_BATCH,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    print("  SPLADE GPU'dan düşürülüyor...")
    d_mat = _to_numpy_2d(doc_emb)
    q_mat = _to_numpy_2d(q_emb)
    gpu_free(model, doc_emb, q_emb)

    print("  SPLADE similarity (CPU)...")
    scores_mat = q_mat @ d_mat.T
    rankings = [_topk_from_scores(scores_mat[i], k) for i in range(len(queries))]

    elapsed = time.perf_counter() - t0
    gpu_free(q_mat, d_mat, scores_mat)
    return rankings, elapsed


def run_dense_rankings(
    model_id: str,
    doc_texts: list[str],
    queries: list[str],
    k: int,
    device: str,
) -> tuple[list[list[int]], float]:
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    print(f"  Dense yükleniyor: {model_id}")
    model = SentenceTransformer(model_id, device=device)

    print("  Dense doküman encoding...")
    doc_vecs = np.asarray(
        model.encode(
            doc_texts,
            batch_size=DENSE_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    print("  Dense sorgu encoding...")
    q_vecs = np.asarray(
        model.encode(
            queries,
            batch_size=DENSE_BATCH,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    print("  Dense GPU'dan düşürülüyor (skorlama CPU)...")
    gpu_free(model)

    scores_mat = q_vecs @ doc_vecs.T
    rankings = [_topk_from_scores(scores_mat[i], k) for i in range(len(queries))]

    elapsed = time.perf_counter() - t0
    gpu_free(doc_vecs, q_vecs, scores_mat)
    return rankings, elapsed


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K, top_n: int = 10) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] += 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [idx for idx, _ in ordered[:top_n]]


def metrics_for_ranking(
    gold: list[dict], chunk_ids: list[str], ranked_indices: list[list[int]]
) -> dict[str, float]:
    return mean_metrics(per_query_metrics(gold, chunk_ids, ranked_indices))


def print_table(rows: list[dict]) -> None:
    headers = (
        ["system", "sec"]
        + [f"Hit@{k}" for k in KS]
        + [f"R@{k}" for k in KS]
        + [f"nDCG@{k}" for k in KS]
        + ["MRR@10", "MRR CI"]
    )
    cells = []
    for r in rows:
        cells.append(
            [
                str(r["system"]),
                str(r["sec"]),
                *[f"{r[f'hit@{k}']:.3f}" for k in KS],
                *[f"{r[f'recall@{k}']:.3f}" for k in KS],
                *[f"{r[f'ndcg@{k}']:.3f}" for k in KS],
                f"{r['mrr@10']:.3f}",
                str(r.get("mrr@10_ci", "")),
            ]
        )
    widths = [max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print("\n" + "=" * 110)
    print("SPARSE / HYBRID BENCHMARK — GOLD SET (küratör onaylı)")
    print("=" * 110)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in cells:
        print(fmt(row))
    best = max(rows, key=lambda r: r["mrr@10"])
    print("-" * 110)
    print(f"En iyi (MRR@10): {best['system']}  ({best['mrr@10']:.3f})")
    print(f"SPLADE={SPLADE_MODEL}  |  Dense={DENSE_MODEL}  |  Füzyon=RRF(k={RRF_K})")
    print("=" * 110 + "\n")


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_ids, chunk_texts = load_chunks()
    gold = load_gold()
    max_k = max(KS)
    pool = max(CANDIDATE_POOL, max_k)
    queries = [g["question"] for g in gold]

    print(f"Chunks: {len(chunk_ids)} | Queries: {len(gold)} | Device: {device}")
    print("NOT: Modeller sırayla yüklenir; biri bitince GPU boşaltılır (4GB güvenli).")
    if not gold:
        raise SystemExit("Küratör onaylı soru yok. Önce: python -m src.eval.build_gold_set --validate")
    if not chunk_ids:
        raise SystemExit("Chunk bulunamadı.")

    run_dir = new_run_dir("sparse")

    print("\n[1/3] BM25F ranking...")
    t0 = time.perf_counter()
    bm25 = BM25FIndex(chunk_texts)
    bm25_pool = [bm25.topk(q, pool) for q in queries]
    bm25_top = [r[:max_k] for r in bm25_pool]
    bm25_sec = round(time.perf_counter() - t0, 1)
    print(f"  hazır ({bm25_sec}s)")

    print("\n[2/3] SPLADE ranking (sonra GPU boşaltılacak)...")
    splade_pool, splade_sec = run_splade_rankings(
        SPLADE_MODEL, chunk_texts, queries, pool, device
    )
    splade_top = [r[:max_k] for r in splade_pool]
    splade_sec = round(splade_sec, 1)
    print(f"  hazır ({splade_sec}s)")

    print("\n[3/3] Dense bge-m3 ranking (sonra GPU boşaltılacak)...")
    dense_pool, dense_sec = run_dense_rankings(
        DENSE_MODEL, chunk_texts, queries, pool, device
    )
    dense_top = [r[:max_k] for r in dense_pool]
    dense_sec = round(dense_sec, 1)
    print(f"  hazır ({dense_sec}s)")

    print("\nHybrid RRF birleştirme (CPU)...")
    t0 = time.perf_counter()
    hybrid_bm25 = [
        rrf_fuse([bm25_pool[i], dense_pool[i]], top_n=max_k) for i in range(len(queries))
    ]
    hybrid_splade = [
        rrf_fuse([splade_pool[i], dense_pool[i]], top_n=max_k) for i in range(len(queries))
    ]
    hybrid_sec = round(time.perf_counter() - t0, 1)

    systems = [
        ("A_BM25F", bm25_top, bm25_sec),
        ("B_SPLADE", splade_top, splade_sec),
        ("C_dense_bge-m3", dense_top, dense_sec),
        ("D_BM25F+bge-m3", hybrid_bm25, round(bm25_sec + dense_sec + hybrid_sec, 1)),
        ("E_SPLADE+bge-m3", hybrid_splade, round(splade_sec + dense_sec + hybrid_sec, 1)),
    ]

    rows: list[dict[str, Any]] = []
    per_by_sys: dict[str, list[dict]] = {}
    for name, ranked_all, sec in systems:
        per_q = per_query_metrics(gold, chunk_ids, ranked_all)
        metrics = mean_metrics(per_q)
        write_per_query(run_dir, name, per_q, sec=sec)
        per_by_sys[name] = per_q
        row = {"system": name, "sec": sec, **metrics}
        rows.append(attach_ci_to_row(row, per_q))
        print(
            f">>> {name}: MRR@10={metrics['mrr@10']:.3f}  "
            f"Hit@1={metrics['hit@1']:.3f}  ({sec}s)"
        )

    print_table(rows)
    summary = write_summary(
        run_dir,
        title="Sparse / Hybrid Benchmark",
        system_rows=rows,
        per_query_by_system=per_by_sys,
        meta={
            "n_queries": len(gold),
            "n_chunks": len(chunk_ids),
            "splade": SPLADE_MODEL,
            "dense": DENSE_MODEL,
            "rrf_k": RRF_K,
        },
    )
    print(f"Döküm: {run_dir}")
    print(f"Özet : {summary}")


if __name__ == "__main__":
    main()
