"""Visual search across CBZ pages using a VLM."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from bd_pipeline import cbz
from bd_pipeline.ocr import _maybe_downscale
from bd_pipeline.pipeline import iter_cbz
from bd_pipeline.prompts import SEARCH_SYSTEM, search_user_prompt

DEFAULT_VLM_MODEL = "qwen3.5:9b"


class OllamaChatClient(Protocol):
    def chat(self, **kwargs): ...  # noqa: ANN003


def default_vlm_model() -> str:
    return os.environ.get("BD_VLM_MODEL", DEFAULT_VLM_MODEL)


def _extract_content(resp) -> str:
    """Ollama returns either a dict-like response or an object; handle both."""
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    msg = getattr(resp, "message", None)
    if msg is None:
        return ""
    return getattr(msg, "content", "") or ""


def search_page(image_bytes: bytes, query: str, *, client: OllamaChatClient, model: str) -> bool:
    """Return True if the VLM judges that `query` is visually present on this page."""
    payload = _maybe_downscale(image_bytes)
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SEARCH_SYSTEM},
            {"role": "user", "content": search_user_prompt(query), "images": [payload]},
        ],
        options={"temperature": 0},
    )
    raw = _extract_content(resp).strip().lower()
    return raw.startswith("oui") or raw.startswith("yes") or raw in {"1", "true", "vrai"}


def search_cbz(
    cbz_path: Path,
    query: str,
    *,
    client: OllamaChatClient,
    model: str | None = None,
) -> list[int]:
    """Return 1-indexed page numbers where `query` is visually present."""
    model = model or default_vlm_model()
    hits: list[int] = []
    for page_num, (_, image_bytes) in enumerate(cbz.iter_pages(cbz_path), start=1):
        if search_page(image_bytes, query, client=client, model=model):
            hits.append(page_num)
    return hits


def search_library(
    library_path: Path,
    query: str,
    *,
    client: OllamaChatClient,
    model: str | None = None,
) -> dict[str, list[int]]:
    """Search all CBZs under `library_path`; return {book_stem: [page_numbers]}."""
    model = model or default_vlm_model()
    results: dict[str, list[int]] = {}
    for cbz_path in iter_cbz(library_path):
        hits = search_cbz(cbz_path, query, client=client, model=model)
        if hits:
            results[cbz_path.stem] = hits
    return results
