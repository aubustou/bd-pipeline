from __future__ import annotations

from PIL import Image

from bd_pipeline import ocr


def test_ocr_page_returns_text(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = ["Bonjour Tintin !"]
    text = ocr.ocr_page(png_bytes, client=fake_ocr_client)
    assert text == "Bonjour Tintin !"


def test_ocr_page_passes_pil_image_to_client(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = ["x"]
    ocr.ocr_page(png_bytes, client=fake_ocr_client)
    assert len(fake_ocr_client.calls) == 1
    assert isinstance(fake_ocr_client.calls[0], Image.Image)


def test_ocr_page_returns_empty_when_no_content(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = [""]
    assert ocr.ocr_page(png_bytes, client=fake_ocr_client) == ""


def test_ocr_page_strips_whitespace(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = ["  Tintin arrive !  \n"]
    assert ocr.ocr_page(png_bytes, client=fake_ocr_client) == "Tintin arrive !"


def test_ocr_page_downscales_large_images(png_bytes_large, fake_ocr_client):
    fake_ocr_client.responses = ["ok"]
    ocr.ocr_page(png_bytes_large, client=fake_ocr_client)
    img = fake_ocr_client.calls[0]
    assert max(img.size) <= ocr.MAX_EDGE_PX
    assert ocr.MAX_EDGE_PX == 2048


def test_ocr_page_small_image_not_downscaled(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = ["ok"]
    ocr.ocr_page(png_bytes, client=fake_ocr_client)
    img = fake_ocr_client.calls[0]
    assert img.size == (64, 64)


def test_ocr_page_single_call_per_page(png_bytes, fake_ocr_client):
    fake_ocr_client.responses = ["text"]
    ocr.ocr_page(png_bytes, client=fake_ocr_client)
    assert len(fake_ocr_client.calls) == 1


def test_default_chandra_url_env_override(monkeypatch):
    monkeypatch.setenv("BD_CHANDRA_URL", "http://my-server:9000")
    assert ocr.default_chandra_url() == "http://my-server:9000"


def test_default_chandra_url_default(monkeypatch):
    monkeypatch.delenv("BD_CHANDRA_URL", raising=False)
    assert ocr.default_chandra_url() == ocr.DEFAULT_CHANDRA_URL
