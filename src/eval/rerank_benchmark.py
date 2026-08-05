#!/usr/bin/env python
"""Reranker benchmark — E (SPLADE + bge-m3 RRF) üzerine cross-encoder.

Kolonlar:
  D0) SPLADE+bge-m3 (RRF)           — reranker yok (baseline)
  D1) + bge-reranker-v2-m3
  D2) + jina-reranker-v2-base-multilingual
  D3) + mmarco-mMiniLMv2 (hafif)
  D4) RRF(mmarco, bge-reranker)     — füzyon kolonu

Gold: curator_verified. Modeller sırayla yüklenir (4GB VRAM).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dump import attach_ci_to_row, new_run_dir, write_per_query, write_summary  # noqa: E402
from embed_benchmark import load_chunks, load_gold  # noqa: E402
from metrics import KS, mean_metrics, per_query_metrics  # noqa: E402
from sparse_benchmark import (  # noqa: E402
    CANDIDATE_POOL,
    DENSE_MODEL,
    RRF_K,
    SPLADE_MODEL,
    gpu_free,
    rrf_fuse,
    run_dense_rankings,
    run_splade_rankings,
)

RERANK_POOL = 20
RERANK_BATCH = 2
MAX_DOC_CHARS = 1500

RERANKERS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "D1_bge-reranker-v2-m3",
        "BAAI/bge-reranker-v2-m3",
        {},
    ),
    (
        "D2_jina-reranker-v2-base-multilingual",
        "jinaai/jina-reranker-v2-base-multilingual",
        {"trust_remote_code": True},
    ),
    (
        "D3_mmarco-mMiniLMv2",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        {},
    ),
]

# Füzyon için hangi iki reranker birleşecek
FUSION_A = "D3_mmarco-mMiniLMv2"
FUSION_B = "D1_bge-reranker-v2-m3"
FUSION_NAME = "D4_RRF(mmarco,bge)"


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
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(headers))
    ]

    def fmt(row: list[str]) -> str:
        return "  ".join(row[i].ljust(widths[i]) for i in range(len(row)))

    print("\n" + "=" * 114)
    print("RERANKER BENCHMARK — base = SPLADE + bge-m3 (RRF)")
    print("=" * 114)
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in cells:
        print(fmt(row))
    best = max(rows, key=lambda r: r["mrr@10"])
    print("-" * 114)
    print(f"En iyi (MRR@10): {best['system']}  ({best['mrr@10']:.3f})")
    print(
        f"Retrieval: SPLADE={SPLADE_MODEL} + Dense={DENSE_MODEL}  "
        f"RRF(k={RRF_K}) pool={RERANK_POOL}"
    )
    print("=" * 114 + "\n")


def build_base_rankings(
    chunk_texts: list[str],
    queries: list[str],
    device: str,
    max_k: int,
) -> tuple[list[list[int]], float]:
    pool = max(CANDIDATE_POOL, RERANK_POOL, max_k)

    print("\n[1/2] SPLADE retrieval...")
    splade_pool, splade_sec = run_splade_rankings(
        SPLADE_MODEL, chunk_texts, queries, pool, device
    )
    print(f"  SPLADE bitti ({splade_sec:.1f}s)")

    print("\n[2/2] Dense bge-m3 retrieval...")
    dense_pool, dense_sec = run_dense_rankings(
        DENSE_MODEL, chunk_texts, queries, pool, device
    )
    print(f"  Dense bitti ({dense_sec:.1f}s)")

    print("\nRRF birleştirme (CPU)...")
    t0 = time.perf_counter()
    fused = [
        rrf_fuse([splade_pool[i], dense_pool[i]], top_n=RERANK_POOL)
        for i in range(len(queries))
    ]
    rrf_sec = time.perf_counter() - t0
    total = splade_sec + dense_sec + rrf_sec
    return fused, total


def patch_xlm_roberta_for_jina() -> None:
    import torch
    import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm

    if hasattr(xlm, "create_position_ids_from_input_ids"):
        return

    def create_position_ids_from_input_ids(
        input_ids, padding_idx, past_key_values_length=0
    ):
        mask = input_ids.ne(padding_idx).int()
        incremental = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental.long() + padding_idx

    xlm.create_position_ids_from_input_ids = create_position_ids_from_input_ids
    print("  (Jina uyumluluk yaması: create_position_ids_from_input_ids)")


def rerank_with_cross_encoder(
    model_id: str,
    model_kwargs: dict[str, Any],
    queries: list[str],
    chunk_texts: list[str],
    candidate_rankings: list[list[int]],
    top_n: int,
    device: str,
) -> tuple[list[list[int]], float]:
    from sentence_transformers import CrossEncoder

    t0 = time.perf_counter()
    print(f"  Reranker yükleniyor: {model_id}")
    if "jina" in model_id.lower():
        patch_xlm_roberta_for_jina()

    model = CrossEncoder(model_id, device=device, **model_kwargs)

    rankings: list[list[int]] = []
    for qi, q in enumerate(queries):
        cands = candidate_rankings[qi]
        if not cands:
            rankings.append([])
            continue
        pairs = [[q, chunk_texts[idx][:MAX_DOC_CHARS]] for idx in cands]
        scores = model.predict(
            pairs,
            batch_size=RERANK_BATCH,
            show_progress_bar=False,
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        order = np.argsort(-scores)
        reranked = [cands[int(j)] for j in order[:top_n]]
        rankings.append(reranked)
        if (qi + 1) % 10 == 0 or qi + 1 == len(queries):
            print(f"    sorgu {qi + 1}/{len(queries)}")

    elapsed = time.perf_counter() - t0
    print("  Reranker GPU'dan düşürülüyor...")
    gpu_free(model)
    return rankings, elapsed


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chunk_ids, chunk_texts = load_chunks()
    gold = load_gold()
    max_k = max(KS)
    queries = [g["question"] for g in gold]

    print(f"Chunks: {len(chunk_ids)} | Queries: {len(gold)} | Device: {device}")
    print("NOT: Retrieval + her reranker sırayla; biri bitince GPU boşaltılır.")
    if not gold:
        raise SystemExit("Küratör onaylı soru yok.")
    if not chunk_ids:
        raise SystemExit("Chunk bulunamadı.")

    run_dir = new_run_dir("rerank")

    base_pool, base_sec = build_base_rankings(chunk_texts, queries, device, max_k)
    base_top = [r[:max_k] for r in base_pool]

    rows: list[dict[str, Any]] = []
    per_by_sys: dict[str, list[dict]] = {}
    ranked_by_sys: dict[str, list[list[int]]] = {}

    # Baseline D0
    per0 = per_query_metrics(gold, chunk_ids, base_top)
    m0 = mean_metrics(per0)
    write_per_query(run_dir, "D0_SPLADE+bge-m3", per0, sec=round(base_sec, 1))
    per_by_sys["D0_SPLADE+bge-m3"] = per0
    ranked_by_sys["D0_SPLADE+bge-m3"] = base_top
    rows.append(
        attach_ci_to_row(
            {"system": "D0_SPLADE+bge-m3", "sec": round(base_sec, 1), **m0}, per0
        )
    )
    print(
        f"\n>>> D0_SPLADE+bge-m3: MRR@10={m0['mrr@10']:.3f}  "
        f"Hit@1={m0['hit@1']:.3f}  ({base_sec:.1f}s)"
    )

    for name, model_id, kwargs in RERANKERS:
        print(f"\n>>> {name}")
        try:
            ranked, sec = rerank_with_cross_encoder(
                model_id=model_id,
                model_kwargs=kwargs,
                queries=queries,
                chunk_texts=chunk_texts,
                candidate_rankings=base_pool,
                top_n=max_k,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  HATA ({model_id}): {exc}")
            print("  Bu reranker atlandı, sonrakine geçiliyor.")
            continue
        per_q = per_query_metrics(gold, chunk_ids, ranked)
        metrics = mean_metrics(per_q)
        write_per_query(run_dir, name, per_q, sec=round(sec, 1))
        per_by_sys[name] = per_q
        ranked_by_sys[name] = ranked
        rows.append(
            attach_ci_to_row({"system": name, "sec": round(sec, 1), **metrics}, per_q)
        )
        print(
            f"  MRR@10={metrics['mrr@10']:.3f}  "
            f"Hit@1={metrics['hit@1']:.3f}  ({sec:.1f}s)"
        )

    # D4: RRF füzyon (mmarco + bge) — skor yok, sıra füzyonu
    if FUSION_A in ranked_by_sys and FUSION_B in ranked_by_sys:
        print(f"\n>>> {FUSION_NAME}")
        t0 = time.perf_counter()
        fused = [
            rrf_fuse(
                [ranked_by_sys[FUSION_A][i], ranked_by_sys[FUSION_B][i]],
                top_n=max_k,
            )
            for i in range(len(queries))
        ]
        sec = round(time.perf_counter() - t0, 1)
        per_q = per_query_metrics(gold, chunk_ids, fused)
        metrics = mean_metrics(per_q)
        write_per_query(run_dir, FUSION_NAME, per_q, sec=sec)
        per_by_sys[FUSION_NAME] = per_q
        rows.append(attach_ci_to_row({"system": FUSION_NAME, "sec": sec, **metrics}, per_q))
        print(
            f"  MRR@10={metrics['mrr@10']:.3f}  "
            f"Hit@1={metrics['hit@1']:.3f}  ({sec}s)"
        )
    else:
        print(
            f"\n{FUSION_NAME} atlandı — {FUSION_A} veya {FUSION_B} sonuçları yok."
        )

    print_table(rows)
    summary = write_summary(
        run_dir,
        title="Reranker Benchmark",
        system_rows=rows,
        per_query_by_system=per_by_sys,
        meta={
            "n_queries": len(gold),
            "n_chunks": len(chunk_ids),
            "base": "SPLADE+bge-m3 RRF",
            "rerank_pool": RERANK_POOL,
            "max_doc_chars": MAX_DOC_CHARS,
            "rrf_k": RRF_K,
        },
    )
    print(f"Döküm: {run_dir}")
    print(f"Özet : {summary}")


if __name__ == "__main__":
    main()
