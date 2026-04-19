"""VLM-based OCR of comic pages via Ollama."""

from __future__ import annotations

import io
import os
from typing import Protocol

from PIL import Image

from bd_pipeline.prompts import OCR_SYSTEM, OCR_USER, OCR_USER_RETRY

DEFAULT_VLM_MODEL = "qwen3.5:9b"
MAX_EDGE_PX = 2048


class OllamaChatClient(Protocol):
    def chat(self, **kwargs): ...  # noqa: ANN003


def default_vlm_model() -> str:
    return os.environ.get("BD_VLM_MODEL", DEFAULT_VLM_MODEL)


def _maybe_downscale(image_bytes: bytes) -> bytes:
    """Downscale to MAX_EDGE_PX long edge; return original bytes if already small."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return image_bytes
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= MAX_EDGE_PX:
        return image_bytes
    scale = MAX_EDGE_PX / long_edge
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    img = img.convert("RGB") if img.mode not in ("RGB", "L") else img
    resized = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def ocr_page(
    image_bytes: bytes,
    *,
    client: OllamaChatClient,
    model: str | None = None,
) -> str:
    """Run OCR on a single page image. Returns extracted text (may be empty)."""
    model = model or default_vlm_model()
    payload = _maybe_downscale(image_bytes)
    text = _call_vlm(client, model, payload, OCR_USER)
    if not text:
        text = _call_vlm(client, model, image_bytes, OCR_USER_RETRY)
    return text


def _call_vlm(client: OllamaChatClient, model: str, payload: bytes, user_msg: str) -> str:
    resp = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": OCR_SYSTEM},
            {"role": "user", "content": user_msg, "images": [payload]},
        ],
        options={"temperature": 0},
    )
    text = _extract_content(resp).strip()
    return "" if text == "(aucun texte)" else text


def _extract_content(resp) -> str:
    """Ollama returns either a dict-like response or an object; handle both."""
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    msg = getattr(resp, "message", None)
    if msg is None:
        return ""
    return getattr(msg, "content", "") or ""
