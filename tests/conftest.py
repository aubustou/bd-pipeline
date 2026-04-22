from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image


def _make_png(w: int = 64, h: int = 64, color=(200, 200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_cbz(tmp_path: Path) -> Callable[..., Path]:
    """Create a synthetic CBZ on disk. Returns the path."""

    def _factory(
        name: str = "Sample.cbz",
        pages: int = 3,
        with_comicinfo: bool = False,
        extras: dict[str, bytes] | None = None,
        page_sizes: list[tuple[int, int]] | None = None,
    ) -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(1, pages + 1):
                size = page_sizes[i - 1] if page_sizes and i - 1 < len(page_sizes) else (64, 64)
                zf.writestr(f"{i:02d}.png", _make_png(*size))
            if with_comicinfo:
                zf.writestr(
                    "ComicInfo.xml",
                    b'<?xml version="1.0" encoding="utf-8"?>\n<ComicInfo><Title>Old</Title></ComicInfo>',
                )
            for entry_name, payload in (extras or {}).items():
                zf.writestr(entry_name, payload)
        return path

    return _factory


class FakeChandraClient:
    """Stand-in for `ChandraOCRClient` used in tests.

    - `responses`: list of strings served in order per parse_image() call.
    """

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[Image.Image] = []

    def parse_image(self, image: Image.Image, **kwargs) -> dict:
        self.calls.append(image)
        text = self.responses.pop(0) if self.responses else ""
        return {"md_content": text}


class FakeOllamaClient:
    """Stand-in for `ollama.Client` used in tests.

    - `analyze_response`: dict returned (as JSON) whenever format='json' is set.
    """

    def __init__(
        self,
        ocr_responses: list[str] | None = None,
        analyze_response: dict | None = None,
    ):
        self.ocr_responses = list(ocr_responses or [])
        self.analyze_response = analyze_response or {
            "summary": "Un résumé de test.",
            "tags": ["test", "aventure"],
            "characters": ["Tintin"],
            "locations": ["Bruxelles"],
            "notable_people": [],
        }
        self.calls: list[dict] = []
        # Per-chunk overrides for map-reduce analyse tests.
        self.analyze_response_queue: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("format") == "json":
            if self.analyze_response_queue:
                payload = self.analyze_response_queue.pop(0)
            else:
                payload = self.analyze_response
            return {"message": {"content": json.dumps(payload, ensure_ascii=False)}}
        # VLM/search path.
        if self.ocr_responses:
            text = self.ocr_responses.pop(0)
        else:
            text = ""
        return {"message": {"content": text}}


@pytest.fixture
def fake_ocr_client() -> FakeChandraClient:
    return FakeChandraClient()


@pytest.fixture
def fake_client() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
def png_bytes() -> bytes:
    return _make_png()


@pytest.fixture
def png_bytes_large() -> bytes:
    return _make_png(w=3000, h=2000, color=(10, 20, 30))
