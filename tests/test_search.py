from __future__ import annotations

import pytest

from bd_pipeline import search
from bd_pipeline.prompts import SEARCH_SYSTEM
from tests.conftest import FakeOllamaClient


# ---------------------------------------------------------------------------
# search_page
# ---------------------------------------------------------------------------

def test_search_page_returns_true_on_oui(png_bytes, fake_client):
    fake_client.ocr_responses = ["oui"]
    assert search.search_page(png_bytes, "escalier", client=fake_client, model="m") is True


def test_search_page_returns_false_on_non(png_bytes, fake_client):
    fake_client.ocr_responses = ["non"]
    assert search.search_page(png_bytes, "escalier", client=fake_client, model="m") is False


def test_search_page_sends_correct_system_prompt(png_bytes, fake_client):
    fake_client.ocr_responses = ["non"]
    search.search_page(png_bytes, "château", client=fake_client, model="test-vlm")
    call = fake_client.calls[-1]
    assert call["messages"][0]["content"] == SEARCH_SYSTEM
    assert call["options"]["temperature"] == 0
    assert call["model"] == "test-vlm"


def test_search_page_query_appears_in_user_message(png_bytes, fake_client):
    fake_client.ocr_responses = ["non"]
    search.search_page(png_bytes, "cheval blanc", client=fake_client, model="m")
    user_msg = fake_client.calls[-1]["messages"][1]["content"]
    assert "cheval blanc" in user_msg


def test_search_page_tolerates_verbose_oui(png_bytes, fake_client):
    fake_client.ocr_responses = ["Oui, on voit clairement un escalier."]
    assert search.search_page(png_bytes, "escalier", client=fake_client, model="m") is True


def test_search_page_accepts_yes_english(png_bytes, fake_client):
    fake_client.ocr_responses = ["yes"]
    assert search.search_page(png_bytes, "car", client=fake_client, model="m") is True


def test_search_page_downscales_large_images(png_bytes_large, fake_client):
    fake_client.ocr_responses = ["non"]
    search.search_page(png_bytes_large, "bateau", client=fake_client, model="m")
    sent = fake_client.calls[-1]["messages"][1]["images"][0]
    assert sent != png_bytes_large


# ---------------------------------------------------------------------------
# search_cbz
# ---------------------------------------------------------------------------

def test_search_cbz_returns_matching_page_numbers(make_cbz, fake_client):
    path = make_cbz(pages=3)
    fake_client.ocr_responses = ["non", "oui", "non"]
    hits = search.search_cbz(path, "escalier", client=fake_client, model="m")
    assert hits == [2]


def test_search_cbz_returns_empty_list_when_no_match(make_cbz, fake_client):
    path = make_cbz(pages=2)
    fake_client.ocr_responses = ["non", "non"]
    assert search.search_cbz(path, "dragon", client=fake_client, model="m") == []


def test_search_cbz_returns_multiple_hits(make_cbz, fake_client):
    path = make_cbz(pages=4)
    fake_client.ocr_responses = ["oui", "non", "oui", "oui"]
    hits = search.search_cbz(path, "château", client=fake_client, model="m")
    assert hits == [1, 3, 4]


def test_search_cbz_uses_default_model_when_none(make_cbz, fake_client, monkeypatch):
    monkeypatch.setenv("BD_VLM_MODEL", "sentinel-model")
    path = make_cbz(pages=1)
    fake_client.ocr_responses = ["non"]
    search.search_cbz(path, "test", client=fake_client, model=None)
    assert fake_client.calls[-1]["model"] == "sentinel-model"


# ---------------------------------------------------------------------------
# search_library
# ---------------------------------------------------------------------------

def test_search_library_finds_hits_across_books(tmp_path, make_cbz):
    # iter_cbz yields files in alphabetical order: Asterix before Tintin
    make_cbz(name="Asterix.cbz", pages=2)
    make_cbz(name="Tintin.cbz", pages=2)
    # Asterix: p1=non, p2=oui → hit on page 2
    # Tintin:  p1=oui, p2=non → hit on page 1
    client = FakeOllamaClient(ocr_responses=["non", "oui", "oui", "non"])
    results = search.search_library(tmp_path, "escalier", client=client, model="m")
    assert results["Asterix"] == [2]
    assert results["Tintin"] == [1]


def test_search_library_omits_books_with_no_hits(tmp_path, make_cbz):
    make_cbz(name="Empty.cbz", pages=2)
    client = FakeOllamaClient(ocr_responses=["non", "non"])
    results = search.search_library(tmp_path, "licorne", client=client, model="m")
    assert results == {}


def test_search_library_skips_cache_dir(tmp_path, make_cbz):
    """CBZs inside .bd_pipeline_cache must not be searched."""
    import io
    import zipfile

    from PIL import Image

    from bd_pipeline.pipeline import CACHE_DIRNAME

    make_cbz(name="Real.cbz", pages=1)
    cache_dir = tmp_path / CACHE_DIRNAME / "Real"
    cache_dir.mkdir(parents=True)
    fake_cbz = cache_dir / "cached.cbz"
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    with zipfile.ZipFile(fake_cbz, "w") as zf:
        zf.writestr("01.png", buf.getvalue())

    client = FakeOllamaClient(ocr_responses=["oui"])
    results = search.search_library(tmp_path, "maison", client=client, model="m")
    assert "cached" not in results
    assert len(client.ocr_responses) == 0  # exactly 1 response consumed (Real.cbz page 1)
