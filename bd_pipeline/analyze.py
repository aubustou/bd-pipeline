"""Text LLM analysis: summary, tags, and named-entity extraction via Ollama."""
from __future__ import annotations

import json
import os
import unicodedata
from typing import Protocol

from bd_pipeline.prompts import (
    ANALYZE_SYSTEM,
    REDUCE_SYSTEM,
    analyze_user_prompt,
    reduce_user_prompt,
)

DEFAULT_LLM_MODEL = "qwen2.5:7b-instruct"
# Rough cap on characters per map chunk. Avoids truly long contexts on small local models.
MAP_CHUNK_CHARS = 12000


class AnalysisError(RuntimeError):
    """Raised when the LLM returns something we cannot parse."""


class OllamaChatClient(Protocol):
    def chat(self, **kwargs): ...  # noqa: ANN003


def default_llm_model() -> str:
    return os.environ.get("BD_LLM_MODEL", DEFAULT_LLM_MODEL)


def _extract_content(resp) -> str:
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    msg = getattr(resp, "message", None)
    if msg is None:
        return ""
    return getattr(msg, "content", "") or ""


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"LLM did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AnalysisError(f"LLM returned {type(data).__name__}, expected object")
    return data


def _coerce_analysis(data: dict) -> dict:
    """Normalise the raw LLM dict into the shape expected by BookAnalysis."""
    def strlist(key: str) -> list[str]:
        val = data.get(key, [])
        if not isinstance(val, list):
            return []
        out: list[str] = []
        for x in val:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    return {
        "summary": str(data.get("summary", "")).strip(),
        "tags": [t.lower() for t in strlist("tags")],
        "characters": strlist("characters"),
        "locations": strlist("locations"),
        "notable_people": strlist("notable_people"),
    }


def _fold(name: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    ).lower().strip()


def _merge_names(*lists: list[str]) -> list[str]:
    """Deduplicate name lists accent- and case-insensitively; prefer the richest spelling."""
    best: dict[str, str] = {}
    for lst in lists:
        for name in lst:
            key = _fold(name)
            if not key:
                continue
            current = best.get(key)
            if current is None or _richer(name, current):
                best[key] = name
    return sorted(best.values(), key=str.lower)


def _richer(a: str, b: str) -> bool:
    """Prefer the form with more diacritics; tie-break by length, then lexically."""
    diac_a = sum(1 for c in unicodedata.normalize("NFKD", a) if unicodedata.combining(c))
    diac_b = sum(1 for c in unicodedata.normalize("NFKD", b) if unicodedata.combining(c))
    if diac_a != diac_b:
        return diac_a > diac_b
    if len(a) != len(b):
        return len(a) > len(b)
    return a < b


def _format_pages(pages_text: list[str], start_index: int = 0) -> str:
    blocks = []
    for i, text in enumerate(pages_text, start=start_index + 1):
        blocks.append(f"--- PAGE {i} ---\n{text}".rstrip())
    return "\n\n".join(blocks)


def _chunk_pages(pages_text: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for page in pages_text:
        plen = len(page) + 20  # account for "--- PAGE N ---" framing
        if current and current_len + plen > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(page)
        current_len += plen
    if current:
        chunks.append(current)
    return chunks


def _call_json(
    client: OllamaChatClient,
    model: str,
    system: str,
    user: str,
) -> dict:
    resp = client.chat(
        model=model,
        format="json",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0},
    )
    return _parse_json(_extract_content(resp))


def analyze_book(
    title: str,
    pages_text: list[str],
    *,
    client: OllamaChatClient,
    model: str | None = None,
    map_chunk_chars: int = MAP_CHUNK_CHARS,
) -> dict:
    """Return a dict with keys: summary, tags, characters, locations, notable_people."""
    model = model or default_llm_model()
    pages_text = [p for p in pages_text]  # shallow copy
    if not any(p.strip() for p in pages_text):
        return {
            "summary": "",
            "tags": [],
            "characters": [],
            "locations": [],
            "notable_people": [],
        }

    total_chars = sum(len(p) for p in pages_text)
    if total_chars <= map_chunk_chars:
        data = _call_json(
            client, model, ANALYZE_SYSTEM, analyze_user_prompt(title, _format_pages(pages_text))
        )
        return _coerce_analysis(data)

    # Map-reduce for long books.
    chunks = _chunk_pages(pages_text, map_chunk_chars)
    partials: list[dict] = []
    offset = 0
    for chunk in chunks:
        raw = _call_json(
            client,
            model,
            ANALYZE_SYSTEM,
            analyze_user_prompt(title, _format_pages(chunk, start_index=offset)),
        )
        partials.append(_coerce_analysis(raw))
        offset += len(chunk)

    reduced = _call_json(
        client,
        model,
        REDUCE_SYSTEM,
        reduce_user_prompt(title, json.dumps(partials, ensure_ascii=False, indent=2)),
    )
    merged = _coerce_analysis(reduced)
    # Defensive merge in case the reducer drops names the map step found.
    merged["characters"] = _merge_names(
        merged["characters"], *[p["characters"] for p in partials]
    )
    merged["locations"] = _merge_names(
        merged["locations"], *[p["locations"] for p in partials]
    )
    merged["notable_people"] = _merge_names(
        merged["notable_people"], *[p["notable_people"] for p in partials]
    )
    seen: set[str] = set()
    merged_tags: list[str] = []
    for t in merged["tags"] + [t for p in partials for t in p["tags"]]:
        if t not in seen:
            seen.add(t)
            merged_tags.append(t)
    merged["tags"] = merged_tags
    return merged
