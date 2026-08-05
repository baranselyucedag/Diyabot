"""Ortak retrieval metrikleri — soru-başına vektör + ortalama."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

KS = [1, 3, 5, 10]


def hit_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for c in ranked[:k] if c in relevant) / len(relevant)


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, c in enumerate(ranked[:k], 1)
        if c in relevant
    )
    ideal = sum(
        1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def mrr_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    for i, c in enumerate(ranked[:k], 1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def per_query_metrics(
    gold: list[dict[str, Any]],
    chunk_ids: list[str],
    ranked_indices: list[list[int]],
    ks: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Her soru için metrik dict + ranked_top10 chunk_id listesi."""
    ks = ks or KS
    max_k = max(ks)
    out: list[dict[str, Any]] = []
    for g, idxs in zip(gold, ranked_indices):
        relevant = set(g["expected_chunk_ids"])
        ranked = [chunk_ids[i] for i in idxs]
        row: dict[str, Any] = {
            "gold_id": g.get("id"),
            "question": g.get("question"),
            "category": g.get("category"),
            "paraphrase_of": g.get("paraphrase_of"),
            "expected_chunk_ids": list(g["expected_chunk_ids"]),
            "ranked_top10": ranked[:10],
        }
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(relevant, ranked, k)
            row[f"recall@{k}"] = recall_at_k(relevant, ranked, k)
            row[f"ndcg@{k}"] = ndcg_at_k(relevant, ranked, k)
        row["mrr@10"] = mrr_at_k(relevant, ranked, 10)
        row["rr"] = row["mrr@10"]
        # truncate ranked list stored for dump
        _ = max_k
        out.append(row)
    return out


def mean_metrics(per_query: list[dict[str, Any]], ks: list[int] | None = None) -> dict[str, float]:
    ks = ks or KS
    keys = [f"hit@{k}" for k in ks] + [f"recall@{k}" for k in ks] + [
        f"ndcg@{k}" for k in ks
    ] + ["mrr@10"]
    return {
        name: round(float(np.mean([q[name] for q in per_query])), 4)
        for name in keys
    }


def metrics_for_ranking(
    gold: list[dict],
    chunk_ids: list[str],
    ranked_indices: list[list[int]],
) -> dict[str, float]:
    """Geriye uyumlu: sadece ortalamalar."""
    return mean_metrics(per_query_metrics(gold, chunk_ids, ranked_indices))
