"""Export Cursor agent transcripts (20.07.2026–26.07.2026) to docs/exports/."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

TRANSCRIPTS = Path(
    r"C:\Users\baran\.cursor\projects\c-Users-baran-Desktop-Staj-Type-2-Diabet-Chatbot\agent-transcripts"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "exports"

START = datetime(2026, 7, 20, 0, 0, 0)
END = datetime(2026, 7, 26, 23, 59, 59)

TS_RE = re.compile(r"<timestamp>([^<]+)</timestamp>", re.I)
TS_FMT = "%A, %b %d, %Y, %I:%M %p (UTC+3)"

CHAT_META = {
    "0c176bf8-b2c1-4673-8138-fcedaca6d1a6": "Ana RAG / Gold / Eval sohbeti",
    "336b2c62-dc6f-4ade-bf6e-47e4b5515730": "Frontend sohbeti",
}


def parse_ts(s: str) -> datetime | None:
    s = " ".join(s.strip().split())
    try:
        return datetime.strptime(s, TS_FMT)
    except ValueError:
        pass
    m = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),\s+(\d{4}),\s+(\d{1,2}):(\d{2})\s+(AM|PM)",
        s,
    )
    if m:
        try:
            return datetime.strptime(m.group(0), "%b %d, %Y, %I:%M %p")
        except ValueError:
            return None
    return None


def iter_parts(content):
    if content is None:
        return
    if isinstance(content, str):
        yield ("text", content)
        return
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            if isinstance(part, str):
                yield ("text", part)
            continue
        t = part.get("type")
        if t == "text" or ("text" in part and t not in ("tool_use", "tool_result")):
            yield ("text", part.get("text", ""))
        elif t == "tool_use":
            name = part.get("name") or part.get("toolName") or "tool"
            yield ("tool", name)
        elif t == "tool_result":
            yield ("tool", "tool_result")
        elif t == "image":
            yield ("text", "[görsel]")


def clean_user(text: str) -> str:
    text = TS_RE.sub("", text)
    text = re.sub(r"<user_query>\s*", "", text)
    text = re.sub(r"\s*</user_query>", "", text)
    text = re.sub(r"<image_files>[\s\S]*?</image_files>", "[görsel eklendi]", text)
    text = re.sub(r"\[Image\]\s*", "", text)
    text = re.sub(
        r"<attached_files>[\s\S]*?</attached_files>", "[ek dosya/plan eklendi]", text
    )
    text = re.sub(
        r"<code_selection[\s\S]*?</code_selection>", "[kod seçimi]", text
    )
    text = re.sub(r"<system_reminder>[\s\S]*?</system_reminder>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def clean_assistant(text: str) -> str:
    text = text.replace("[REDACTED]", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def chat_id_from_path(path: Path) -> str:
    for p in path.parts:
        if re.fullmatch(r"[0-9a-f-]{36}", p):
            return p
    return path.stem


def process_file(path: Path):
    is_sub = "subagents" in path.parts
    cid = chat_id_from_path(path)
    title = CHAT_META.get(cid, cid)
    if is_sub:
        title = f"{title} — alt ajan ({path.stem[:8]})"

    entries = []
    last_ts: datetime | None = None

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = row.get("role") or "unknown"
            msg = row.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            texts: list[str] = []
            tools: list[str] = []
            for kind, val in iter_parts(content):
                if kind == "text" and val:
                    texts.append(val)
                elif kind == "tool":
                    tools.append(val)

            raw = "\n".join(texts)
            m = TS_RE.search(raw)
            if m:
                dt = parse_ts(m.group(1))
                if dt:
                    last_ts = dt
            ts = last_ts
            if ts is None or ts < START or ts > END:
                continue

            if role == "user":
                body = clean_user(raw)
                label = "USER"
            elif role == "assistant":
                body = clean_assistant(raw)
                label = "ASSISTANT"
            else:
                body = clean_assistant(raw)
                label = role.upper()

            tool_note = ""
            if tools:
                seen: list[str] = []
                for t in tools:
                    if t not in seen:
                        seen.append(t)
                extra = f" +{len(seen) - 12} more" if len(seen) > 12 else ""
                tool_note = f"\n\n_[tool: {', '.join(seen[:12])}{extra}]_"

            if not body and not tool_note:
                continue
            if not body and tool_note:
                body = "_(yalnızca araç çağrıları — kısaltıldı)_"

            entries.append((ts, label, body + tool_note))

    return title, cid, is_sub, entries


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mains = []
    subs = []
    for path in sorted(TRANSCRIPTS.rglob("*.jsonl")):
        title, cid, is_sub, entries = process_file(path)
        if not entries:
            print(f"SKIP (no msgs in window): {path.relative_to(TRANSCRIPTS)}")
            continue
        item = (title, cid, path, entries)
        (subs if is_sub else mains).append(item)
        print(f"OK {len(entries):4d} msgs | {path.relative_to(TRANSCRIPTS)}")

    all_items = mains + subs
    index_lines = [
        "# Sohbet geçmişi export",
        "",
        "**Aralık:** 2026-07-20 → 2026-07-26 (UTC+3 zaman damgaları)",
        f"**Oluşturulma:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Not: Tool çağrıları kısaltıldı; yalnızca kullanıcı/asistan metinleri tutuldu.",
        "",
        "## Dosyalar",
        "",
    ]

    written_names: set[str] = set()
    for title, cid, path, entries in all_items:
        safe = re.sub(r"[^\w\-]+", "_", title)[:60].strip("_")
        d0 = entries[0][0].strftime("%Y-%m-%d")
        d1 = entries[-1][0].strftime("%Y-%m-%d")
        fname = f"{d0}_{safe}.md" if d0 == d1 else f"{d0}_to_{d1}_{safe}.md"
        out = OUT_DIR / fname
        n = 2
        while out.name in written_names:
            out = OUT_DIR / f"{Path(fname).stem}_{n}.md"
            n += 1
        written_names.add(out.name)

        lines = [
            f"# {title}",
            "",
            f"- **Chat ID:** `{cid}`",
            f"- **Kaynak:** `{path}`",
            f"- **Mesaj sayısı (filtreli):** {len(entries)}",
            f"- **İlk:** {entries[0][0].strftime('%Y-%m-%d %H:%M')}",
            f"- **Son:** {entries[-1][0].strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            "",
        ]
        for ts, label, body in entries:
            stamp = ts.strftime("%Y-%m-%d %H:%M")
            lines.append(f"### [{stamp}] {label}")
            lines.append("")
            lines.append(body)
            lines.append("")
            lines.append("---")
            lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        index_lines.append(
            f"- [{title}]({out.name}) — {len(entries)} mesaj ({d0} → {d1})"
        )
        print(f"WROTE {out.name} ({len(entries)} msgs, {out.stat().st_size // 1024} KB)")

    combined = OUT_DIR / "sohbet_gecmisi_2026-07-20_2026-07-26.md"
    comb = [
        "# Tüm sohbetler (20.07.2026 – 26.07.2026)",
        "",
        "Birleşik export. Ayrı dosyalar için bkz. [README](README.md).",
        "",
    ]
    for title, cid, path, entries in all_items:
        comb.append(f"\n\n# {title}\n")
        comb.append(f"Chat ID: `{cid}`\n\n---\n")
        for ts, label, body in entries:
            stamp = ts.strftime("%Y-%m-%d %H:%M")
            comb.append(f"### [{stamp}] {label}\n\n{body}\n\n---\n")

    combined.write_text("\n".join(comb), encoding="utf-8")
    print(f"WROTE combined {combined.name} ({combined.stat().st_size // 1024} KB)")

    index_lines += [
        "",
        f"- **Birleşik:** [{combined.name}]({combined.name})",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    print("DONE", OUT_DIR)


if __name__ == "__main__":
    main()
