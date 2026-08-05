#!/usr/bin/env python
"""Ragas aşama 2: predictions.jsonl → skorlar + summary.md.

ROL AYIRIMI:
  - Yanıt üreten chat LLM = Nemotron (src/api/llm.py) — bu dosya ona DOKUNMAZ.
  - Hakem (judge) LLM     = GPT (OpenAI, sabit) — yalnızca skorlama için.

Embedding gerektiren metrikler (answer_relevancy vb.) bilinçli olarak dışarıda.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dump import RESULTS_DIR  # noqa: E402
from metrics import KS, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402

from src.api.env import load_project_env  # noqa: E402
from src.api.triage import _norm  # noqa: E402

# Ragas 0.4 hâlâ evaluate + legacy metrikleri destekler; collections API
# Instructor LLM ister. NVIDIA OpenAI-uyumlu uç için LangchainLLMWrapper daha sağlam.
warnings.filterwarnings(
    "ignore",
    message=".*deprecated.*ragas.metrics.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="evaluate\\(\\) is deprecated.*",
    category=DeprecationWarning,
)

# Sabit hakem — chat Nemotron'dan ayrı (self-judge yok)
# NVIDIA free uçta Kimi modelleri hesapta kapalı (404/410) → judge = OpenAI GPT.
# Kaliteyi artırmak istersen "gpt-4o" yap (maliyet yükselir).
JUDGE_MODEL = "gpt-4o-mini"


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """predictions.jsonl dosyasını satır satır okuyup dict listesi döner.

    Boş satırlar atlanır. Dosya yoksa net bir FileNotFoundError fırlatır ki
    kullanıcı önce ragas_predict çalıştırması gerektiğini görsün.
    """
    if not path.exists():
        raise FileNotFoundError(f"predictions.jsonl yok: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_judge_config() -> dict[str, str]:
    """GPT hakem ayarını döner; API key OPENAI_API_KEY'den alınır.

    Chat/Nemotron (NVIDIA) ile tamamen ayrı: farklı sağlayıcı, farklı key.
    Key, frontend/.env içindeki OPENAI_API_KEY satırından okunur.
    """
    load_project_env()
    import os

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY yok. frontend/.env içine yazın "
            "(yalnızca Ragas judge için; chat Nemotron NVIDIA key kullanır)."
        )
    return {"model": JUDGE_MODEL, "api_key": api_key}


def build_judge_llm():
    """Yalnızca Ragas skorlaması için GPT hakem (llm_factory) üretir.

    Nemotron (yanıt LLM, NVIDIA) burada hiç kullanılmaz. max_tokens=4096:
    Ragas metrikleri uzun JSON üretir; varsayılan 1024 IncompleteOutput
    hatasına yol açar (faithfulness statement/NLI çıktıları kesilir).
    """
    from openai import OpenAI
    from ragas.llms import llm_factory

    cfg = resolve_judge_config()
    client = OpenAI(api_key=cfg["api_key"])
    print(f"Hakem LLM (GPT, chat değil): {cfg['model']} (max_tokens=4096)")
    return llm_factory(
        cfg["model"],
        provider="openai",
        client=client,
        max_tokens=4096,
    )


def build_ragas_metrics(*, with_lexical: bool = False) -> list[Any]:
    """Embedding istemeyen Ragas LLM metrik listesini oluşturur.

    Faithfulness, context precision/recall, factual correctness, entity recall
    ve noise sensitivity dahil. --with-lexical ile Rouge/Bleu eklenir.
    """
    from ragas.metrics import (
        ContextEntityRecall,
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
    )

    metrics: list[Any] = [
        Faithfulness(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        FactualCorrectness(mode="f1"),
        ContextEntityRecall(),
        NoiseSensitivity(mode="relevant"),
    ]
    if with_lexical:
        try:
            from ragas.metrics import BleuScore, RougeScore

            metrics.extend([RougeScore(), BleuScore()])
        except Exception as exc:  # noqa: BLE001
            print(f"Uyarı: lexical metrikler yüklenemedi ({exc}). Atlandı.")
    return metrics


def phrase_coverage(answer: str, phrases: list[str], *, mode: str) -> float:
    """must_include / must_not_include listelerine göre 0–1 kapsama skoru.

    Türkçe karakterler _norm ile sadeleştirilir. mode='include' ise kaçının
    cevapta geçtiği; mode='exclude' ise kaçının GEÇMEDİĞİ oranlanır.
    """
    if not phrases:
        return float("nan")
    ans = _norm(answer or "")
    hits = sum(1 for p in phrases if _norm(p) in ans)
    if mode == "include":
        return hits / len(phrases)
    if mode == "exclude":
        # Hiçbiri geçmemeli → başarı = (toplam - geçen) / toplam
        return (len(phrases) - hits) / len(phrases)
    raise ValueError(f"Bilinmeyen mode: {mode}")


def retrieval_row_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Tek prediction satırı için chunk-id tabanlı Hit/Recall/nDCG/MRR hesaplar.

    LLM çağrısı yapmaz; expected_chunk_ids ile ranked_chunk_ids karşılaştırır.
    Guardrail / boş retrieval satırlarında NaN döner.
    """
    expected = set(row.get("expected_chunk_ids") or [])
    ranked = list(row.get("ranked_chunk_ids") or [])
    if not expected or not ranked:
        return {f"hit@{k}": float("nan") for k in KS} | {
            f"recall@{k}": float("nan") for k in KS
        } | {f"ndcg@{k}": float("nan") for k in KS} | {"mrr@10": float("nan")}

    out: dict[str, float] = {}
    for k in KS:
        out[f"hit@{k}"] = hit_at_k(expected, ranked, k)
        out[f"recall@{k}"] = recall_at_k(expected, ranked, k)
        out[f"ndcg@{k}"] = ndcg_at_k(expected, ranked, k)
    out["mrr@10"] = mrr_at_k(expected, ranked, 10)
    return out


def split_rag_vs_guardrail(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Satırları RAG skorlanacaklar ve guardrail-only diye ikiye ayırır.

    skipped_rag=True, boş answer/error veya boş context'ler Ragas LLM
    metriklerine girmez; guardrail raporunda triage doğruluğu ölçülür.
    """
    rag: list[dict[str, Any]] = []
    guard: list[dict[str, Any]] = []
    for r in rows:
        if r.get("skipped_rag") or r.get("error") or not (r.get("answer") or "").strip():
            guard.append(r)
        elif not (r.get("retrieved_contexts") or []):
            guard.append(r)
        else:
            rag.append(r)
    return rag, guard


def rows_to_evaluation_dataset(rows: list[dict[str, Any]]):
    """Prediction dict'lerini Ragas EvaluationDataset / SingleTurnSample'a çevirir.

    Alan eşlemesi: question→user_input, answer→response,
    retrieved_contexts→retrieved_contexts, reference→reference.
    """
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r.get("question") or "",
            response=r.get("answer") or "",
            retrieved_contexts=list(r.get("retrieved_contexts") or []),
            reference=r.get("reference") or "",
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def safe_mean(values: list[float]) -> float:
    """NaN'ları atlayarak ortalama alır; hiç sayı yoksa NaN döner."""
    nums = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return float("nan")
    return float(sum(nums) / len(nums))


def compute_free_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tüm satırlar için LLM'siz (ücretsiz) metrik dict listesi üretir.

    must_include/exclude, triage eşleşmesi ve chunk-id retrieval metriklerini
    birleştirir; Ragas skorlarıyla sonradan merge edilir.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        free = {
            "gold_id": r.get("gold_id"),
            "must_include_coverage": phrase_coverage(
                r.get("answer") or "", list(r.get("must_include") or []), mode="include"
            ),
            "must_not_include_ok": phrase_coverage(
                r.get("answer") or "",
                list(r.get("must_not_include") or []),
                mode="exclude",
            ),
            "triage_match": 1.0
            if str(r.get("detected_triage")) == str(r.get("expected_triage"))
            else 0.0,
            "expected_triage": r.get("expected_triage"),
            "detected_triage": r.get("detected_triage"),
            "safety_critical": bool(r.get("safety_critical")),
            "is_guardrail": bool(r.get("skipped_rag")),
            "category": r.get("category"),
            "question": r.get("question"),
        }
        free.update(retrieval_row_metrics(r))
        out.append(free)
    return out


def run_ragas_llm_metrics(
    rag_rows: list[dict[str, Any]],
    *,
    with_lexical: bool = False,
    max_workers: int = 4,
    timeout: int = 180,
) -> list[dict[str, float]]:
    """RAG satırları üzerinde Ragas evaluate() çağırır; skor dict listesi döner.

    raise_exceptions=False ile tek satır hatası tüm koşuyu düşürmez (NaN).
    Sonuç pandas'a çevrilmez; result.scores doğrudan kullanılır.
    """
    if not rag_rows:
        return []

    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = rows_to_evaluation_dataset(rag_rows)
    metrics = build_ragas_metrics(with_lexical=with_lexical)
    llm = build_judge_llm()
    run_config = RunConfig(max_workers=max_workers, timeout=timeout)

    print(f"Ragas evaluate: {len(rag_rows)} satır, {len(metrics)} metrik...")
    t0 = time.perf_counter()
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )
    print(f"Ragas bitti ({time.perf_counter() - t0:.1f}s): {result}")
    # scores: list[dict[str, float]]
    return list(result.scores)


def merge_scores(
    all_rows: list[dict[str, Any]],
    free_rows: list[dict[str, Any]],
    rag_rows: list[dict[str, Any]],
    ragas_scores: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """Free metrikler + Ragas skorlarını gold_id üzerinden tek satırda birleştirir.

    Guardrail satırlarında Ragas alanları NaN kalır; rapor bunu ayırt eder.
    """
    by_id = {f["gold_id"]: dict(f) for f in free_rows}
    ragas_by_id: dict[str, dict[str, float]] = {}
    for row, sc in zip(rag_rows, ragas_scores):
        ragas_by_id[str(row.get("gold_id"))] = dict(sc)

    merged: list[dict[str, Any]] = []
    for r in all_rows:
        gid = r.get("gold_id")
        item = dict(by_id.get(gid, {"gold_id": gid}))
        item.update(ragas_by_id.get(str(gid), {}))
        item["answer_preview"] = ((r.get("answer") or "")[:240]).replace("\n", " ")
        merged.append(item)
    return merged


def triage_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """expected×detected triage karışıklık matrisi ve accuracy üretir.

    safety_critical alt kümesi ayrıca raporlanır; acil kaçırmalar kritik sinyaldir.
    """
    pairs = [
        (str(r.get("expected_triage")), str(r.get("detected_triage"))) for r in rows
    ]
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    matrix: dict[str, dict[str, int]] = {
        e: {d: 0 for d in labels} for e in labels
    }
    for e, d in pairs:
        matrix[e][d] += 1
    accuracy = safe_mean([1.0 if e == d else 0.0 for e, d in pairs])
    safety = [r for r in rows if r.get("safety_critical")]
    safety_acc = safe_mean(
        [
            1.0
            if str(r.get("expected_triage")) == str(r.get("detected_triage"))
            else 0.0
            for r in safety
        ]
    )
    return {
        "labels": labels,
        "matrix": matrix,
        "accuracy": accuracy,
        "safety_critical_n": len(safety),
        "safety_critical_accuracy": safety_acc,
    }


def write_scores_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Soru başına birleşik skor satırlarını JSONL olarak yazar."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_summary_md(
    path: Path,
    *,
    merged: list[dict[str, Any]],
    confusion: dict[str, Any],
    meta: dict[str, Any],
    worst_n: int = 10,
) -> None:
    """İnsan okunaklı summary.md üretir: ortalamalar, triage matrisi, kötü örnekler.

    Faithfulness ortalamasına göre en düşük N RAG sorusu listelenir; guardrail
    satırları bu sıralamaya girmez.
    """
    lines: list[str] = [
        "# Ragas Değerlendirme Özeti",
        "",
        f"Oluşturulma: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Meta",
        "",
    ]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Ortalama metrikler
    skip_keys = {
        "gold_id",
        "expected_triage",
        "detected_triage",
        "safety_critical",
        "is_guardrail",
        "category",
        "question",
        "answer_preview",
    }
    numeric_keys: set[str] = set()
    for row in merged:
        for k, v in row.items():
            if k in skip_keys:
                continue
            if isinstance(v, (int, float)):
                numeric_keys.add(k)

    lines.append("## Ortalama metrikler")
    lines.append("")
    lines.append("| metrik | mean | n |")
    lines.append("|---|---:|---:|")
    for key in sorted(numeric_keys):
        vals = [float(r[key]) for r in merged if key in r and r[key] is not None]
        vals = [v for v in vals if not math.isnan(v)]
        mean = safe_mean(vals)
        mean_s = f"{mean:.4f}" if not math.isnan(mean) else "nan"
        lines.append(f"| `{key}` | {mean_s} | {len(vals)} |")
    lines.append("")

    # Triage
    lines.append("## Triage doğruluğu")
    lines.append("")
    lines.append(f"- Genel accuracy: **{confusion['accuracy']:.3f}**")
    lines.append(
        f"- safety_critical accuracy: **{confusion['safety_critical_accuracy']:.3f}** "
        f"(n={confusion['safety_critical_n']})"
    )
    lines.append("")
    labels = confusion["labels"]
    header = "| expected \\ detected | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join(["---:" for _ in labels]) + "|"
    lines.append(header)
    lines.append(sep)
    for e in labels:
        cells = [str(confusion["matrix"][e].get(d, 0)) for d in labels]
        lines.append(f"| **{e}** | " + " | ".join(cells) + " |")
    lines.append("")

    # Worst faithfulness
    rag_scored = [
        r
        for r in merged
        if not r.get("is_guardrail")
        and isinstance(r.get("faithfulness"), (int, float))
        and not math.isnan(float(r["faithfulness"]))
    ]
    rag_scored.sort(key=lambda r: float(r.get("faithfulness", 1.0)))
    lines.append(f"## En düşük faithfulness (top {worst_n})")
    lines.append("")
    for r in rag_scored[:worst_n]:
        lines.append(
            f"- `{r.get('gold_id')}` faithfulness={float(r['faithfulness']):.3f} — "
            f"{r.get('question')}"
        )
        lines.append(f"  - preview: {r.get('answer_preview')}")
    if not rag_scored:
        lines.append("_RAG skorlu satır yok._")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_score(
    run_dir: Path,
    *,
    with_lexical: bool = False,
    max_workers: int = 4,
    timeout: int = 180,
) -> Path:
    """Tek eval klasörü üzerinde free + Ragas skorlarını çalıştırıp yazar.

    Beklenen girdi: run_dir/predictions.jsonl. Çıktılar: scores.jsonl,
    summary.md, score_meta.json.
    """
    pred_path = run_dir / "predictions.jsonl"
    rows = load_predictions(pred_path)
    rag_rows, guard_rows = split_rag_vs_guardrail(rows)
    print(
        f"Toplam={len(rows)} | RAG skorlanacak={len(rag_rows)} | "
        f"guardrail/atlanan={len(guard_rows)}"
    )

    free = compute_free_metrics(rows)
    ragas_scores = run_ragas_llm_metrics(
        rag_rows,
        with_lexical=with_lexical,
        max_workers=max_workers,
        timeout=timeout,
    )
    merged = merge_scores(rows, free, rag_rows, ragas_scores)
    confusion = triage_confusion(rows)

    scores_path = run_dir / "scores.jsonl"
    write_scores_jsonl(scores_path, merged)

    judge_cfg = resolve_judge_config()
    meta = {
        "n_total": len(rows),
        "n_rag": len(rag_rows),
        "n_guardrail": len(guard_rows),
        "judge_model": judge_cfg["model"],
        "judge_base_url": judge_cfg["base_url"],
        "with_lexical": with_lexical,
        "max_workers": max_workers,
        "timeout": timeout,
    }
    (run_dir / "score_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = run_dir / "summary.md"
    write_summary_md(summary_path, merged=merged, confusion=confusion, meta=meta)
    print(f"Skorlar : {scores_path}")
    print(f"Özet    : {summary_path}")
    return summary_path


def latest_ragas_run(results_dir: Path = RESULTS_DIR) -> Path | None:
    """data/eval_results altında en yeni ragas_* klasörünü bulur (yoksa None)."""
    if not results_dir.exists():
        return None
    dirs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("ragas_")],
        key=lambda p: p.name,
    )
    return dirs[-1] if dirs else None


def main() -> None:
    """CLI: python -m src.eval.ragas_score --run <klasör> | --latest."""
    parser = argparse.ArgumentParser(
        description="Ragas aşama 2: predictions.jsonl → scores + summary."
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Eval klasörü (predictions.jsonl içeren ragas_<stamp>).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="data/eval_results içindeki en yeni ragas_* klasörünü kullan.",
    )
    parser.add_argument(
        "--with-lexical",
        action="store_true",
        help="RougeScore + BleuScore ekle (ek paket gerekebilir).",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    run_dir = args.run
    if args.latest or run_dir is None:
        run_dir = latest_ragas_run()
        if run_dir is None:
            raise SystemExit(
                "ragas_* klasörü yok. Önce: python -m src.eval.ragas_predict --limit 5"
            )
        print(f"Kullanılan run: {run_dir}")

    if not run_dir.exists():
        raise SystemExit(f"Klasör yok: {run_dir}")

    run_score(
        run_dir,
        with_lexical=args.with_lexical,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
