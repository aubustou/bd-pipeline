"""Aggregate per-book sidecar JSONs into a library-wide markdown index."""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from bd_pipeline.models import BookAnalysis


def _fold(name: str) -> str:
    return (
        "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
        .lower()
        .strip()
    )


def _richer(a: str, b: str) -> bool:
    diac_a = sum(1 for c in unicodedata.normalize("NFKD", a) if unicodedata.combining(c))
    diac_b = sum(1 for c in unicodedata.normalize("NFKD", b) if unicodedata.combining(c))
    if diac_a != diac_b:
        return diac_a > diac_b
    if len(a) != len(b):
        return len(a) > len(b)
    return a < b


def _load_sidecars(root: Path) -> list[BookAnalysis]:
    books: list[BookAnalysis] = []
    for f in sorted(root.rglob("*.json")):
        # Only JSON files that sit next to a matching CBZ.
        if not f.with_suffix(".cbz").exists():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            books.append(BookAnalysis.model_validate(data))
        except Exception:
            continue
    return books


def _collect_names(books: list[BookAnalysis], attr: str) -> dict[str, list[str]]:
    """Return {display_name: [sorted unique book titles]} for the given attribute."""
    by_key: dict[str, dict] = {}
    for b in books:
        for raw in getattr(b, attr):
            key = _fold(raw)
            if not key:
                continue
            entry = by_key.setdefault(key, {"display": raw, "books": set()})
            if _richer(raw, entry["display"]):
                entry["display"] = raw
            entry["books"].add(b.title)
    return {entry["display"]: sorted(entry["books"]) for entry in by_key.values()}


def _render_name_section(title: str, mapping: dict[str, list[str]]) -> list[str]:
    if not mapping:
        return []
    lines = [f"### {title}", ""]
    for name in sorted(mapping, key=lambda s: (_fold(s), s)):
        books = ", ".join(mapping[name])
        lines.append(f"- **{name}** — {books}")
    lines.append("")
    return lines


def build_index(root: Path) -> Path:
    """Write `<root>/INDEX.md` and return its path."""
    root = Path(root)
    books = _load_sidecars(root)
    lines: list[str] = ["# Bibliothèque BD", ""]
    lines.append(f"*{len(books)} album(s) indexé(s).*")
    lines.append("")

    # Books table
    lines.append("## Albums")
    lines.append("")
    if not books:
        lines.append("_Aucun album trouvé._")
        lines.append("")
    else:
        lines.append("| Titre | Pages | Tags |")
        lines.append("| --- | --- | --- |")
        for b in sorted(books, key=lambda x: x.title.lower()):
            tags = ", ".join(b.tags) if b.tags else "—"
            lines.append(f"| {b.title} | {b.page_count} | {tags} |")
        lines.append("")

    # Tag cloud
    by_tag: dict[str, list[str]] = defaultdict(list)
    for b in books:
        for t in b.tags:
            by_tag[t].append(b.title)
    if by_tag:
        lines.append("## Tags")
        lines.append("")
        for tag in sorted(by_tag, key=str.lower):
            titles = ", ".join(sorted(by_tag[tag]))
            lines.append(f"- **{tag}** ({len(by_tag[tag])}) — {titles}")
        lines.append("")

    # Name index
    lines.append("## Index des noms")
    lines.append("")
    lines += _render_name_section("Personnages", _collect_names(books, "characters"))
    lines += _render_name_section("Lieux", _collect_names(books, "locations"))
    lines += _render_name_section("Personnalités réelles", _collect_names(books, "notable_people"))

    out = root / "INDEX.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out
