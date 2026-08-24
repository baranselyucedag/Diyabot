#!/usr/bin/env python
"""Korpus envanteri — manuel gold eşleme için dizin.

Çıktı: data/gold/authoring/corpus_inventory.md
Her chunk: chunk_id | bölüm/başlık | ilk ~200 karakter
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
OUT_PATH = ROOT / "data" / "gold" / "authoring" / "corpus_inventory.md"

HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.MULTILINE)


def preview(text: str, n: int = 200) -> str:
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if len(flat) <= n:
        return flat
    return flat[: n - 1] + "…"


def heading_of(content: str, section: str | None, chapter: str | None) -> str:
    match = HEADING_RE.search(content or "")
    if match:
        return match.group(1).strip()
    if section:
        return section
    if chapter:
        return chapter
    lines = (content or "").splitlines()
    return lines[0].strip()[:80] if lines else "(başlıksız)"


def load_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(PROCESSED.glob("*.chunks.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def build_markdown(chunks: list[dict]) -> str:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c.get("document_id") or c.get("source") or "?"].append(c)

    lines: list[str] = [
        "# Korpus Envanteri",
        "",
        "Manuel gold `chunk_ids` eşlemesi için referans dizin.",
        f"Toplam chunk: **{len(chunks)}** | Doküman: **{len(by_doc)}**",
        "",
        "Her satır: `chunk_id` — başlık — önizleme (ilk ~200 karakter).",
        "",
    ]

    for doc_id in sorted(by_doc):
        group = by_doc[doc_id]
        group.sort(key=lambda c: c.get("chunk_id", ""))
        src = group[0].get("source", "")
        lines.append(f"## `{doc_id}`")
        lines.append("")
        lines.append(f"Kaynak: {src} — {len(group)} chunk")
        lines.append("")
        for c in group:
            hid = c["chunk_id"]
            title = heading_of(
                c.get("content", ""),
                c.get("section"),
                c.get("chapter"),
            )
            prev = preview(c.get("content", ""))
            lines.append(f"- `{hid}`")
            lines.append(f"  - **{title}**")
            lines.append(f"  - {prev}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Korpus envanter dökümü.")
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    chunks = load_chunks()
    if not chunks:
        raise SystemExit(f"Chunk yok: {PROCESSED}")

    md = build_markdown(chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Yazıldı: {args.output} ({len(chunks)} chunk)")


if __name__ == "__main__":
    main()
