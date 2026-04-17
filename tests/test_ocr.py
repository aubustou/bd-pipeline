from __future__ import annotations

import io

from PIL import Image

from bd_pipeline import ocr
from bd_pipeline.prompts import OCR_SYSTEM


def test_ocr_page_returns_model_text(png_bytes, fake_client):
    fake_client.ocr_responses = ["Bonjour Tintin !"]
    text = ocr.ocr_page(png_bytes, client=fake_client, model="test-vlm")
    assert text == "Bonjour Tintin !"


def test_ocr_page_uses_temperature_zero_and_french_prompt(png_bytes, fake_client):
    fake_client.ocr_responses = ["x"]
    ocr.ocr_page(png_bytes, client=fake_client, model="test-vlm")
    call = fake_client.calls[-1]
    assert call["options"]["temperature"] == 0
    assert call["messages"][0]["content"] == OCR_SYSTEM
    assert call["messages"][1]["images"][0] == png_bytes  # small image, no downscale
    assert call["model"] == "test-vlm"


def test_ocr_page_empty_marker_returns_empty_string(png_bytes, fake_client):
    fake_client.ocr_responses = ["(aucun texte)"]
    assert ocr.ocr_page(png_bytes, client=fake_client, model="m") == ""


def test_ocr_page_downscales_large_images(png_bytes_large, fake_client):
    fake_client.ocr_responses = ["ok"]
    ocr.ocr_page(png_bytes_large, client=fake_client, model="m")
    sent = fake_client.calls[-1]["messages"][1]["images"][0]
    assert sent != png_bytes_large  # bytes were re-encoded
    img = Image.open(io.BytesIO(sent))
    assert max(img.size) <= ocr.MAX_EDGE_PX


def test_default_vlm_model_env_override(monkeypatch):
    monkeypatch.setenv("BD_VLM_MODEL", "custom-vlm:latest")
    assert ocr.default_vlm_model() == "custom-vlm:latest"


def test_default_vlm_model_default(monkeypatch):
    monkeypatch.delenv("BD_VLM_MODEL", raising=False)
    assert ocr.default_vlm_model() == ocr.DEFAULT_VLM_MODEL
