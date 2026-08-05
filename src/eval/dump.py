"""Benchmark sonuç dökümü: per_query.jsonl + summary.md."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from metrics import KS, mean_metrics
from stats import bootstrap_ci, compare_systems

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "eval_results"


def new_run_dir(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{name}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_per_query(
    run_dir: Path,
    system: str,
    per_query: list[dict[str, Any]],
    sec: float | None = None,
) -> Path:
    path = run_dir / "per_query.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for q in per_query:
            row = {"system": system, **q}
            if sec is not None:
                row["system_sec"] = sec
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_per_query_by_system(run_dir: Path) -> dict[str, list[dict]]:
    path = run_dir / "per_query.jsonl"
    by_sys: dict[str, list[dict]] = {}
    if not path.exists():
        return by_sys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_sys.setdefault(row["system"], []).append(row)
    return by_sys


def miss_at_3(per_query: list[dict]) -> list[dict]:
    return [q for q in per_query if q.get("hit@3", 1) == 0.0]


def write_summary(
    run_dir: Path,
    title: str,
    system_rows: list[dict[str, Any]],
    per_query_by_system: dict[str, list[dict]],
    meta: dict[str, Any] | None = None,
    compare_top_n: int = 3,
    n_boot: int = 10_000,
) -> Path:
    """summary.md: metrik tablosu + CI + McNemar + Hit@3=0 hata listesi."""
    lines: list[str] = [
        f"# {title}",
        "",
        f"Oluşturulma (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if meta:
        lines.append("## Meta")
        lines.append("")
        for k, v in meta.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Metrikler (+ %95 bootstrap CI)")
    lines.append("")
    lines.append(
        "| system | sec | Hit@1 | Hit@1 CI | Hit@3 | MRR@10 | MRR@10 CI |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|")

    enriched: list[dict] = []
    for row in system_rows:
        name = row["system"]
        pq = per_query_by_system.get(name, [])
        if pq:
            questions = [q.get("question") for q in pq]
            paras = [q.get("paraphrase_of") for q in pq]
            from stats import _cluster_ids

            clusters = _cluster_ids(len(pq), paras, questions)
            hit_ci = bootstrap_ci(
                [q["hit@1"] for q in pq], n_boot=n_boot, clusters=clusters
            )
            mrr_ci = bootstrap_ci(
                [q["mrr@10"] for q in pq], n_boot=n_boot, clusters=clusters
            )
        else:
            hit_ci = {"mean": row.get("hit@1", 0), "ci_low": 0, "ci_high": 0}
            mrr_ci = {"mean": row.get("mrr@10", 0), "ci_low": 0, "ci_high": 0}
        enriched.append({**row, "hit_ci": hit_ci, "mrr_ci": mrr_ci})
        lines.append(
            f"| {name} | {row.get('sec', '-')} | "
            f"{row.get('hit@1', 0):.3f} | "
            f"[{hit_ci['ci_low']:.3f}, {hit_ci['ci_high']:.3f}] | "
            f"{row.get('hit@3', 0):.3f} | "
            f"{row.get('mrr@10', 0):.3f} | "
            f"[{mrr_ci['ci_low']:.3f}, {mrr_ci['ci_high']:.3f}] |"
        )
    lines.append("")

    # Top-N pairwise
    ranked = sorted(enriched, key=lambda r: (-r.get("mrr@10", 0), -r.get("hit@1", 0)))
    top = ranked[:compare_top_n]
    if len(top) >= 2:
        lines.append("## İkili karşılaştırma (McNemar Hit@1 + paired MRR delta)")
        lines.append("")
        lines.append(
            "Karar kuralı: kazanan = McNemar p<0.05 **veya** paired-delta CI 0'ı dışlar."
        )
        lines.append("")
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                cmp = compare_systems(
                    a["system"],
                    per_query_by_system[a["system"]],
                    b["system"],
                    per_query_by_system[b["system"]],
                    n_boot=n_boot,
                )
                mc = cmp["mcnemar"]
                d = cmp["mrr_delta"]
                lines.append(f"### {a['system']} vs {b['system']}")
                lines.append("")
                lines.append(
                    f"- McNemar: a_only={mc['a_only']}, b_only={mc['b_only']}, "
                    f"p={mc['p_value']}"
                )
                lines.append(
                    f"- MRR delta (A−B): {d['delta_mean']} "
                    f"CI [{d['ci_low']}, {d['ci_high']}] "
                    f"excludes_zero={d['excludes_zero']}"
                )
                lines.append(
                    f"- Anlamlı: **{cmp['significant']}** | "
                    f"Kazanan: {cmp['winner'] or 'beraberlik'}"
                )
                lines.append("")

    # Error analysis Hit@3=0 — best system
    if ranked:
        best_name = ranked[0]["system"]
        misses = miss_at_3(per_query_by_system.get(best_name, []))
        lines.append(f"## Hata analizi — {best_name} Hit@3=0")
        lines.append("")
        lines.append(f"Toplam miss: **{len(misses)}**")
        lines.append("")
        by_cat: dict[str, list] = {}
        for m in misses:
            by_cat.setdefault(m.get("category") or "?", []).append(m)
        for cat in sorted(by_cat):
            lines.append(f"### {cat} ({len(by_cat[cat])})")
            lines.append("")
            for m in by_cat[cat]:
                lines.append(f"- `{m.get('gold_id')}`: {m.get('question')}")
                lines.append(f"  - beklenen: `{m.get('expected_chunk_ids')}`")
                lines.append(f"  - top3: `{m.get('ranked_top10', [])[:3]}`")
            lines.append("")

    lines.append("## Karar kuralı (sabit)")
    lines.append("")
    lines.append(
        "1. Kazanan: McNemar p < 0.05 **veya** paired-delta %95 CI 0'ı dışlar."
    )
    lines.append(
        "2. Beraberlikte: düşük gecikme/VRAM → mimari sadelik "
        "(tek reranker > füzyon) → küçük model."
    )
    lines.append("")

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    # also dump machine-readable summary
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "title": title,
                "meta": meta or {},
                "systems": enriched,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def attach_ci_to_row(row: dict, per_query: list[dict], n_boot: int = 10_000) -> dict:
    """Ortalama satırına CI alanları ekle (terminal tablosu için)."""
    from stats import _cluster_ids

    questions = [q.get("question") for q in per_query]
    paras = [q.get("paraphrase_of") for q in per_query]
    clusters = _cluster_ids(len(per_query), paras, questions)
    hit_ci = bootstrap_ci([q["hit@1"] for q in per_query], n_boot=n_boot, clusters=clusters)
    mrr_ci = bootstrap_ci([q["mrr@10"] for q in per_query], n_boot=n_boot, clusters=clusters)
    return {
        **row,
        "hit@1_ci": f"[{hit_ci['ci_low']:.3f},{hit_ci['ci_high']:.3f}]",
        "mrr@10_ci": f"[{mrr_ci['ci_low']:.3f},{mrr_ci['ci_high']:.3f}]",
    }


# re-export for convenience
__all__ = [
    "KS",
    "RESULTS_DIR",
    "attach_ci_to_row",
    "load_per_query_by_system",
    "mean_metrics",
    "miss_at_3",
    "new_run_dir",
    "write_per_query",
    "write_summary",
]
