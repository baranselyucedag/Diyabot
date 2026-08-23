#!/usr/bin/env python
"""RAGChecker — RAGAS'tan bağımsız, claim-level RAG teşhis değerlendirmesi.

BAĞIMSIZLIK (kritik kısıtlama):
    - ``src/eval/ragas_predict.py`` ve ``src/eval/ragas_score.py`` İMPORT EDİLMEZ.
    - Judge çözümleyici (``resolve_judge_config`` / ``build_judge_llm``) kullanılmaz.
    - Tek girdi: mevcut bir ``predictions.jsonl`` (RAGAS'ın ürettiği) + chunk
      içeriği için ``data/processed/*.chunks.jsonl`` (kendi stdlib okuyucumuzla,
      ``src.retrieval`` dahil hiçbir proje modülü import edilmez).
    - Main ortamdan BAĞIMSIZ bir venv'de çalıştırılır (RAGChecker litellm üzerinden
      httpx>=0.28 / transformers<5 ister; main ise httpx<0.28 / transformers 5.x).

LLM ÇAĞRILARI (litellm BYPASS — rate-limit'e dayanıklı):
    - ``custom_llm_api_func`` ile litellm'in ``batch_completion``'ı hiç çağrılmaz.
    - İstekler ``--concurrency`` kadar paralel (varsayılan 2, burst yok), her istek
      yalnızca başarısızsa exponential backoff ile tekrar denenir (başarılılar korunur).

CHECKPOINT / RESUME:
    - Satırlar ``--chunk-size`` (varsayılan 20) parçalara bölünür; her parça bitince
      ``ragchecker_checkpoint.json``'a atomik yazılır. Çökerse en fazla son parça
      kaybedilir; tekrar çalıştırınca biten satırlar atlanır.

İKİ-PAS TASARIM (farklı context genişliği):
    PAS 1 — retriever metrikleri (claim_recall, context_precision) → **top-k** context.
    PAS 2 — generator + overall metrikleri → **top-3** context (LLM'in gördüğü).
    overall (precision/recall/f1) context'e bağımlı değildir; PAS 2'de sıfır ek
    LLM maliyetiyle hesaplanır.

Kullanım (main ortama DOKUNMAZ):
    .venv-ragchecker/Scripts/python.exe src/eval/ragchecker_run.py --predictions eval_results/ragas_*/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from ragchecker import RAGChecker, RAGResults
from ragchecker.metrics import (  # noqa: E402
    claim_recall,
    context_precision,
    context_utilization,
    faithfulness,
    f1,
    hallucination,
    noise_sensitivity_in_irrelevant,
    noise_sensitivity_in_relevant,
    precision,
    recall,
    self_knowledge,
)

# Metrik grupları — FAZ 0'da kurulu paketten doğrulanan gerçek adlar (birer string).
RETRIEVER_METRICS = [claim_recall, context_precision]
OVERALL_METRICS = [precision, recall, f1]
GENERATOR_METRICS = [
    context_utilization,
    hallucination,
    noise_sensitivity_in_relevant,
    noise_sensitivity_in_irrelevant,
    self_knowledge,
    faithfulness,
]

DEFAULT_JUDGE_MODEL = "openai/meta/llama-3.3-70b-instruct"
DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1"
CHECKPOINT_NAME = "ragchecker_checkpoint.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ENV = PROJECT_ROOT / "frontend" / ".env"
ROOT_ENV = PROJECT_ROOT / ".env"
CHUNKS_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------------------------------- #
# Kendi kendine yeterli yardımcılar (src.* import edilmez → hafif venv'de çalışır)
# --------------------------------------------------------------------------- #
def load_env_key(key: str) -> str:
    """frontend/.env veya kök .env'den anahtarı okur; os.environ'a da yazar."""
    if os.environ.get(key):
        return os.environ[key]
    for env_path in (FRONTEND_ENV, ROOT_ENV):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k == key and v:
                os.environ[k] = v
                return v
    return ""


def build_content_map() -> dict[str, str]:
    """chunk_id -> content haritası; data/processed/*.chunks.jsonl okur (saf stdlib)."""
    content_map: dict[str, str] = {}
    if not CHUNKS_DIR.exists():
        return content_map
    for path in sorted(CHUNKS_DIR.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("chunk_id")
            content = row.get("content")
            if cid and content is not None:
                content_map[cid] = content
    return content_map


# --------------------------------------------------------------------------- #
# custom_llm_api_func — litellm'i bypass eden, rate-limit'e dayanıklı çağrı
# --------------------------------------------------------------------------- #
def _strip_provider(model: str) -> str:
    """'openai/meta/llama-3.3-70b-instruct' -> 'meta/llama-3.3-70b-instruct'."""
    return model.split("/", 1)[1] if "/" in model else model


def _is_retryable(exc: Exception) -> bool:
    """Yalnızca geçici hatalar retry edilir; auth/bad-request çökertilir."""
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return getattr(exc, "status_code", 0) >= 500
    return False


def make_custom_llm_api_func(
    model: str,
    api_base: str,
    api_key: str,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    concurrency: int = 2,
    max_retries: int = 15,
    base_delay: float = 5.0,
) -> Callable[[list[str]], list[str]]:
    """RAGChecker'a verilecek ``custom_llm_api_func`` üretir.

    SÖZLEŞME (refchecker.utils.get_model_batch_response'tan doğrulandı):
        custom_llm_api_func(prompts: list[str]) -> list[str]
    - Yalnızca ``prompts`` verilir (temperature/max_tokens/model VERİLMEZ) — burada sabit.
    - Çıktı, girdiyle AYNI uzunluk ve SIRADA, her eleman modelin ham metin çıktısı.
    - Asla None dönmez; her istek yalnızca başarısızsa exponential backoff ile tekrarlanır
      (başarılılar korunur). litellm'in batch_completion'ı (tüm batch'i tekrar etme) KULLANILMAZ.
    """
    client = OpenAI(api_key=api_key, base_url=api_base)
    model_name = _strip_provider(model)

    def _call_one(prompt: str) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                if not _is_retryable(exc):
                    raise
                delay = base_delay * (2 ** min(attempt, 8))
                print(
                    f"  [retry {attempt + 1}/{max_retries}] {type(exc).__name__} "
                    f"→ {delay}s bekleniyor...",
                    flush=True,
                )
                time.sleep(delay)
        print("  [UYARI] max_retries doldu, boş cevap dönülüyor.", flush=True)
        return ""

    def call(prompts: list[str]) -> list[str]:
        if concurrency <= 1:
            return [_call_one(p) for p in prompts]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(_call_one, prompts))

    return call


# --------------------------------------------------------------------------- #
# FAZ 3 — Dönüştürücü
# --------------------------------------------------------------------------- #
def load_and_filter(pred_path: Path) -> list[dict[str, Any]]:
    """predictions.jsonl oku; RAG satırlarını döndür (guardrail/skipped dışı)."""
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.jsonl yok: {pred_path}")
    rows: list[dict[str, Any]] = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return [r for r in rows if not r.get("skipped_rag") and not r.get("is_guardrail")]


def cost_controlled_sample(
    rag_rows: list[dict[str, Any]], target_total: int, seed: int | None = 42
) -> list[dict[str, Any]]:
    """safety_critical satırlar HER ZAMAN dahil; geri kalanı hedefe kadar örneklenir."""
    if target_total <= 0 or target_total >= len(rag_rows):
        return rag_rows
    rng = random.Random(seed)
    safety = [r for r in rag_rows if r.get("safety_critical")]
    other = [r for r in rag_rows if not r.get("safety_critical")]
    remaining = max(0, target_total - len(safety))
    sampled_other = rng.sample(other, min(remaining, len(other)))
    result = safety + sampled_other
    print(
        f"Örnekleme: {len(safety)} safety_critical + {len(sampled_other)} rastgele = "
        f"{len(result)} toplam (havuz: {len(rag_rows)} RAG satırı)"
    )
    return result


def build_checking_input(
    rows: list[dict[str, Any]],
    *,
    use_top_k: bool,
    content_map: dict[str, str] | None,
) -> dict[str, Any]:
    """RAGChecker'ın beklediği ``{results: [...]}`` yapısını üretir.

    - use_top_k=True  → ranked_chunk_ids'in TAMAMI + content_map'ten metin.
    - use_top_k=False → yalnızca retrieved_contexts (top-3, LLM'in gördüğü).
    """
    results: list[dict[str, Any]] = []
    for r in rows:
        if use_top_k:
            ctx = [
                {"doc_id": cid, "text": (content_map or {}).get(cid, "")}
                for cid in r.get("ranked_chunk_ids", [])
            ]
        else:
            ctx = [
                {"doc_id": cid, "text": txt}
                for cid, txt in zip(
                    r.get("ranked_chunk_ids", [])[:3], r.get("retrieved_contexts", [])
                )
            ]
        results.append(
            {
                "query_id": r["gold_id"],
                "query": r["question"],
                "gt_answer": r["reference"],
                "response": r["answer"],
                "retrieved_context": ctx,
            }
        )
    return {"results": results}


# --------------------------------------------------------------------------- #
# FAZ 4 — iki-pas + birleştirme
# --------------------------------------------------------------------------- #
def _copy_gt_answer_claims(src: RAGResults, dst: RAGResults) -> None:
    """PAS 1'de çıkarılan gt_answer claim'lerini PAS 2'ye taşır (tekrar çıkarma yok)."""
    for s, d in zip(src.results, dst.results):
        d.gt_answer_claims = s.gt_answer_claims


def _pct(v: Any) -> float:
    """RAGChecker'ın 0–1 metrik değerini yüzdeye (0–100) çevirir."""
    if isinstance(v, (int, float)):
        return round(float(v) * 100, 4)
    return float("nan")


def merge_per_query_flat(
    topk_results: RAGResults,
    top3_results: RAGResults,
    sampled: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Soru başına metrikleri TEK DÜZ satırda birleştirir (yüzde, 0–100)."""
    meta = {r["gold_id"]: r for r in sampled}
    topk_by_id = {r.query_id: r.metrics for r in topk_results.results}
    top3_by_id = {r.query_id: r.metrics for r in top3_results.results}
    out: list[dict[str, Any]] = []
    for r in sampled:
        gid = r["gold_id"]
        row: dict[str, Any] = {
            "gold_id": gid,
            "category": r.get("category"),
            "safety_critical": bool(r.get("safety_critical")),
        }
        for m in top3_by_id.get(gid, {}):
            row[m] = _pct(top3_by_id[gid][m])
        for m in topk_by_id.get(gid, {}):
            row[m] = _pct(topk_by_id[gid][m])
        out.append(row)
    return out


def _safe_mean(vals: list[Any]) -> float:
    nums = [
        v
        for v in vals
        if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)
    ]
    return float(sum(nums) / len(nums)) if nums else float("nan")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-query (0–100) satırlarından grup ortalamalarını üretir (RAGChecker'ın
    ``round(mean*100, 1)`` davranışına birebir denk)."""
    def group_mean(keys: list[str]) -> dict[str, float]:
        return {k: round(_safe_mean([r.get(k) for r in rows]), 1) for k in keys}

    return {
        "overall": group_mean(OVERALL_METRICS),
        "retriever": group_mean(RETRIEVER_METRICS),
        "generator": group_mean(GENERATOR_METRICS),
    }


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #
def load_checkpoint(out_dir: Path) -> tuple[set[str], list[dict[str, Any]]]:
    path = out_dir / CHECKPOINT_NAME
    if not path.exists():
        return set(), []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("done", [])), list(data.get("per_query", []))
    except Exception as exc:  # noqa: BLE001
        print(f"Checkpoint okunamadı ({exc}); baştan başlanıyor.")
        return set(), []


def save_checkpoint(
    out_dir: Path, done: set[str], per_query: list[dict[str, Any]]
) -> None:
    path = out_dir / CHECKPOINT_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"done": sorted(done), "per_query": per_query},
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)  # atomik


# --------------------------------------------------------------------------- #
# FAZ 5 — Rapor
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "nan"
    try:
        if isinstance(v, float) and v != v:  # NaN
            return "nan"
        return f"{v:.1f}"
    except (TypeError, ValueError):
        return str(v)


def write_ragchecker_report(
    out_dir: Path,
    *,
    merged: dict[str, Any],
    per_query: list[dict[str, Any]],
    n_total: int,
    n_sampled: int,
    meta: dict[str, Any],
) -> None:
    """ragchecker_scores.json + ragchecker_summary.md + per_query.jsonl yazar."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "ragchecker_scores.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "ragchecker_per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    overall = merged.get("overall", {})
    retr = merged.get("retriever", {})
    gen = merged.get("generator", {})
    lines = [
        "# RAGChecker Sonuçları",
        "",
        f"Toplam RAG satırı: {n_total} | Örneklenen: {n_sampled}",
        "",
        "## Meta",
        "",
    ]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines += [
        "",
        "## Overall metrikler (claim-level doğruluk/tamlık)",
        "",
        f"- precision: **{_fmt(overall.get('precision'))}**",
        f"- recall: **{_fmt(overall.get('recall'))}**",
        f"- f1: **{_fmt(overall.get('f1'))}**",
        "",
        "## Retriever metrikleri (top-k context ile)",
        "",
        f"- claim_recall: **{_fmt(retr.get('claim_recall'))}**",
        f"- context_precision: **{_fmt(retr.get('context_precision'))}**",
        "",
        "## Generator metrikleri (top-3 context ile, LLM'in gerçekten gördüğü)",
        "",
        f"- context_utilization: **{_fmt(gen.get('context_utilization'))}**",
        f"- faithfulness: **{_fmt(gen.get('faithfulness'))}**",
        f"- hallucination: **{_fmt(gen.get('hallucination'))}** (düşük = iyi)",
        f"- noise_sensitivity_in_relevant: **{_fmt(gen.get('noise_sensitivity_in_relevant'))}** (düşük = iyi)",
        f"- noise_sensitivity_in_irrelevant: **{_fmt(gen.get('noise_sensitivity_in_irrelevant'))}** (düşük = iyi)",
        f"- self_knowledge: **{_fmt(gen.get('self_knowledge'))}**",
        "",
        "## Yorum notları",
        "",
        "- Claim extraction/checking promptları İngilizce; Türkçe claim kalitesi "
        "spot-check ile doğrulanmıştır (bkz. FAZ 2).",
        "- hallucination / noise_sensitivity / self_knowledge düşük olması daha iyidir.",
        "- Per-query değerler yüzde (0–100); aggregate = satır ortalaması, 1 ondalık.",
        "",
    ]
    (out_dir / "ragchecker_summary.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def build_evaluator_from(
    judge_model: str,
    api_base: str,
    *,
    batch_size: int,
    joint_check_num: int,
    max_tokens: int,
    concurrency: int,
) -> RAGChecker:
    api_key = load_env_key("NVIDIA_API_KEY").strip()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY yok. frontend/.env içine yazın.")

    custom_func = make_custom_llm_api_func(
        judge_model,
        api_base,
        api_key,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )
    return RAGChecker(
        extractor_name=judge_model,
        checker_name=judge_model,
        batch_size_extractor=batch_size,
        batch_size_checker=batch_size,
        joint_check_num=joint_check_num,
        custom_llm_api_func=custom_func,
    )


def run_ragchecker(
    pred_path: Path,
    out_dir: Path,
    *,
    sample_size: int,
    judge_model: str,
    api_base: str,
    batch_size: int,
    seed: int | None,
    joint_check_num: int,
    chunk_size: int,
    max_tokens: int,
    concurrency: int,
) -> dict[str, Any]:
    rag_rows = load_and_filter(pred_path)
    if not rag_rows:
        raise SystemExit("RAG satırı yok (hepsi guardrail/skipped). RAGChecker uygulanamaz.")

    sampled = cost_controlled_sample(rag_rows, target_total=sample_size, seed=seed)
    content_map = build_content_map()  # lokal lookup, ücretsiz (LLM yok)

    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator = build_evaluator_from(
        judge_model,
        api_base,
        batch_size=batch_size,
        joint_check_num=joint_check_num,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )

    done_ids, per_query_acc = load_checkpoint(out_dir)
    remaining = [r for r in sampled if r["gold_id"] not in done_ids]
    if done_ids:
        print(f"Checkpoint: {len(done_ids)} satır hazır, {len(remaining)} kaldı.")
    if not remaining:
        print("Tüm satırlar tamamlanmış; yalnızca rapor yazılıyor.")

    for i in range(0, len(remaining), chunk_size):
        chunk = remaining[i : i + chunk_size]
        gids = [r["gold_id"] for r in chunk]
        total_done = len(done_ids) + len(chunk)
        print(f"\n=== Chunk [{total_done}/{len(sampled)}] {gids[0]}..{gids[-1]} ===")

        # PAS 1 — retriever metrikleri, TOP-K context ile
        topk_input = build_checking_input(chunk, use_top_k=True, content_map=content_map)
        topk_results = RAGResults.from_dict(topk_input)
        evaluator.evaluate(topk_results, RETRIEVER_METRICS)

        # PAS 2 — generator + overall, TOP-3 context ile
        top3_input = build_checking_input(chunk, use_top_k=False, content_map=None)
        top3_results = RAGResults.from_dict(top3_input)
        _copy_gt_answer_claims(topk_results, top3_results)
        evaluator.evaluate(top3_results, GENERATOR_METRICS + OVERALL_METRICS)

        per_query_acc.extend(merge_per_query_flat(topk_results, top3_results, chunk))
        done_ids.update(gids)
        save_checkpoint(out_dir, done_ids, per_query_acc)
        print(f"Chunk bitti, checkpoint kaydedildi ({len(done_ids)}/{len(sampled)}).")

    merged = aggregate(per_query_acc)

    meta = {
        "judge_model": judge_model,
        "api_base": api_base,
        "batch_size": batch_size,
        "joint_check_num": joint_check_num,
        "chunk_size": chunk_size,
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "seed": seed,
        "n_total_rag": len(rag_rows),
        "n_sampled": len(sampled),
        "top_k_retriever_context": True,
        "top3_generator_context": True,
        "checkpoint": CHECKPOINT_NAME,
    }
    write_ragchecker_report(
        out_dir,
        merged=merged,
        per_query=per_query_acc,
        n_total=len(rag_rows),
        n_sampled=len(sampled),
        meta=meta,
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAGChecker — RAGAS'tan bağımsız değerlendirme (litellm bypass + resume)."
    )
    parser.add_argument(
        "--predictions", type=Path, required=True,
        help="Var olan bir predictions.jsonl yolu.",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Çıktı klasörü (verilmezse predictions ile aynı klasör).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=80,
        help="safety_critical + rastgele örnek hedefi (0 = tümü).",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help="Extractor/checker modeli (litellm notasyonu).",
    )
    parser.add_argument(
        "--api-base", type=str, default=DEFAULT_API_BASE,
        help="OpenAI-uyumlu LLM endpoint'i.",
    )
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Refchecker'ın prompt gruplama boyutu (concurrency'i etkilemez).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joint-check-num", type=int, default=30,
                        help="Bir checker prompt'unda kontrol edilen claim sayısı (büyük = az çağrı).")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="Checkpoint granülerliği (satır). Çökerse en fazla bu kadar kaybedilir.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Paralel istek sayısı (rate-limit'i tetiklememek için düşük tutun).")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"predictions.jsonl yok: {args.predictions}")

    out_dir = args.out or args.predictions.parent
    merged = run_ragchecker(
        args.predictions,
        out_dir,
        sample_size=args.sample_size,
        judge_model=args.judge_model,
        api_base=args.api_base,
        batch_size=args.batch_size,
        seed=args.seed,
        joint_check_num=args.joint_check_num,
        chunk_size=args.chunk_size,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )
    print("\n=== RAGChecker sonuçları ===")
    print(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"\nRapor: {out_dir}")


if __name__ == "__main__":
    main()
