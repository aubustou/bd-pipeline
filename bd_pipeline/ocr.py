"""ChandraOCR 2 page OCR for comic pages."""

from __future__ import annotations

import io
import os
from typing import Protocol

from PIL import Image

MAX_EDGE_PX = 2048
DEFAULT_CHANDRA_URL = "http://localhost:8000"


class ChandraClient(Protocol):
    def parse_image(self, image: Image.Image, **kwargs) -> dict: ...


def default_chandra_url() -> str:
    return os.environ.get("BD_CHANDRA_URL", DEFAULT_CHANDRA_URL)


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


def ocr_page(image_bytes: bytes, *, client: ChandraClient) -> str:
    """Run OCR on a single page image via ChandraOCR 2. Returns extracted text."""
    image_bytes = _maybe_downscale(image_bytes)
    img = Image.open(io.BytesIO(image_bytes))
    result = client.parse_image(img)
    return result.get("md_content", "").strip()
