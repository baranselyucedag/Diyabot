#!/usr/bin/env python
"""Gold-set embedding benchmark — terminal karşılaştırma tablosu."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.core.dump import attach_ci_to_row, new_run_dir, write_per_query, write_summary  # noqa: E402
from src.eval.core.metrics import KS, mean_metrics, per_query_metrics  # noqa: E402
from src.api.env import load_project_env  # noqa: E402

CHUNKS_DIR = ROOT / "data" / "processed"
GOLD_PATH = ROOT / "data" / "gold" / "gold_set.jsonl"

TURKISH_E5_TASK = (
    "Given a Turkish search query, retrieve relevant passages written in "
    "Turkish that best answer the query"
)


@dataclass
class ModelSpec:
    key: str
    kind: str  # st | openai
    model_id: str
    query_fn: Callable[[str], str]
    passage_fn: Callable[[str], str]


MODELS = [
    ModelSpec("bge-m3", "st", "BAAI/bge-m3", lambda x: x, lambda x: x),
    ModelSpec(
        "turkish-e5-large",
        "st",
        "ytu-ce-cosmos/turkish-e5-large",
        lambda x: f"Instruct: {TURKISH_E5_TASK}\nQuery: {x}",
        lambda x: x,
    ),
    ModelSpec(
        "e5-base",
        "st",
        "intfloat/multilingual-e5-base",
        lambda x: f"query: {x}",
        lambda x: f"passage: {x}",
    ),
    ModelSpec("openai-3-small", "openai", "text-embedding-3-small", lambda x: x, lambda x: x),
    ModelSpec("openai-3-large", "openai", "text-embedding-3-large", lambda x: x, lambda x: x),
]


def load_chunks() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for path in sorted(CHUNKS_DIR.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ids.append(row["chunk_id"])
            texts.append(row["content"])
    return ids, texts


def load_gold() -> list[dict[str, Any]]:
    """Sadece KÜRATÖR ONAYLI sorular (expected_chunk_ids elle doğrulanmış)."""
    rows = []
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("curator_verified"):
            continue
        if not row.get("expected_chunk_ids"):
            continue
        rows.append(row)
    return rows


class LocalSTEncoder:
    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )


class OpenAIEncoder:
    def __init__(self, model_id: str):
        from openai import OpenAI

        load_project_env()
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY yok. frontend/.env içine yazın.")
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out: list[list[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc=self.model_id):
            resp = self.client.embeddings.create(
                model=self.model_id,
                input=texts[i : i + batch_size],
            )
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend(d.embedding for d in ordered)
        arr = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr / norms


def topk_indices(q: np.ndarray, d: np.ndarray, k: int) -> np.ndarray:
    scores = q @ d.T
    if k >= scores.shape[1]:
        return np.argsort(-scores, axis=1)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def evaluate(
    spec: ModelSpec,
    chunk_ids: list[str],
    chunk_texts: list[str],
    gold: list[dict],
) -> tuple[dict, list[dict]]:
    max_k = max(KS)
    encoder = (
        LocalSTEncoder(spec.model_id) if spec.kind == "st" else OpenAIEncoder(spec.model_id)
    )

    print(f"\n>>> {spec.key} ({spec.model_id})")
    t0 = time.perf_counter()
    doc_vecs = encoder.encode([spec.passage_fn(t) for t in chunk_texts])
    query_vecs = encoder.encode([spec.query_fn(g["question"]) for g in gold], batch_size=16)
    ranked_idx = topk_indices(query_vecs, doc_vecs, max_k)
    elapsed = time.perf_counter() - t0

    ranked_lists = [ranked_idx[qi].tolist() for qi in range(len(gold))]
    per_q = per_query_metrics(gold, chunk_ids, ranked_lists)
    means = mean_metrics(per_q)
    row = {"model": spec.key, "system": spec.key, "sec": round(elapsed, 1), **means}
    return row, per_q


def print_table(rows: list[dict]) -> None:
    headers = (
        ["model", "sec"]
        + [f"Hit@{k}" for k in KS]
        + [f"R@{k}" for k in KS]
        + [f"nDCG@{k}" for k in KS]
        + ["MRR@10", "MRR CI"]
    )

    cells = []
    for r in rows:
        cells.append(
            [
                str(r["model"]),
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
    print("EMBEDDING BENCHMARK — GOLD SET (küratör onaylı)")
    print("=" * 110)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in cells:
        print(fmt(row))

    best = max(rows, key=lambda r: r["mrr@10"])
    print("-" * 110)
    print(f"En iyi (MRR@10): {best['model']}  ({best['mrr@10']:.3f})")
    print("NOT: expected_chunk_ids KÜRATÖR ONAYLIDIR (elle doğrulandı).")
    print("=" * 110 + "\n")


def main() -> None:
    chunk_ids, chunk_texts = load_chunks()
    gold = load_gold()
    print(f"Chunks: {len(chunk_ids)} | Queries: {len(gold)} | Models: {len(MODELS)}")

    if not gold:
        raise SystemExit(
            "Küratör onaylı soru yok. Önce: python -m src.eval.goldset.build_gold_set --validate"
        )
    if not chunk_ids:
        raise SystemExit("Chunk bulunamadı (data/processed/*.chunks.jsonl).")

    run_dir = new_run_dir("embed")
    rows: list[dict] = []
    per_by_sys: dict[str, list[dict]] = {}

    for spec in MODELS:
        row, per_q = evaluate(spec, chunk_ids, chunk_texts, gold)
        write_per_query(run_dir, spec.key, per_q, sec=row["sec"])
        per_by_sys[spec.key] = per_q
        rows.append(attach_ci_to_row(row, per_q))

    print_table(rows)
    summary = write_summary(
        run_dir,
        title="Embedding Benchmark",
        system_rows=rows,
        per_query_by_system=per_by_sys,
        meta={"n_queries": len(gold), "n_chunks": len(chunk_ids)},
    )
    print(f"Döküm: {run_dir}")
    print(f"Özet : {summary}")


if __name__ == "__main__":
    main()
