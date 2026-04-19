from __future__ import annotations

import json

import pytest

from bd_pipeline import cbz
from bd_pipeline.pipeline import CACHE_DIRNAME, process_cbz, process_library
from tests.conftest import FakeOllamaClient


def _fresh_client():
    return FakeOllamaClient(
        ocr_responses=["page 1 text", "page 2 text", "page 3 text"],
        analyze_response={
            "summary": "Un album d'aventure.",
            "tags": ["aventure"],
            "characters": ["Tintin", "Milou"],
            "locations": ["Bruxelles"],
            "notable_people": [],
        },
    )


def test_process_cbz_writes_sidecar_and_comicinfo(make_cbz):
    path = make_cbz(pages=3)
    client = _fresh_client()
    process_cbz(path, vlm_client=client, llm_client=client)

    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["title"] == path.stem
    assert data["page_count"] == 3
    assert data["summary"] == "Un album d'aventure."

    tree = cbz.read_comicinfo(path)
    fields = {c.tag: c.text for c in tree.getroot()}
    assert fields["Summary"] == "Un album d'aventure."
    assert fields["Tags"] == "aventure"
    assert "Tintin" in fields["Characters"]
    assert fields["LanguageISO"] == "fr"
    # 3 OCR calls + 1 analyse call.
    assert sum(1 for c in client.calls if c.get("format") == "json") == 1
    assert sum(1 for c in client.calls if c.get("format") != "json") == 3


def test_process_cbz_resumable_skips_all_calls(make_cbz):
    path = make_cbz(pages=2)
    first = _fresh_client()
    process_cbz(path, vlm_client=first, llm_client=first)

    second = FakeOllamaClient()  # empty — must NOT be called
    again = process_cbz(path, vlm_client=second, llm_client=second)
    assert second.calls == []
    assert again.title == path.stem


def test_process_cbz_force_reruns(make_cbz):
    path = make_cbz(pages=2)
    first = _fresh_client()
    process_cbz(path, vlm_client=first, llm_client=first)

    # Delete the page cache so force=True actually re-OCRs.
    cache = path.parent / CACHE_DIRNAME / path.stem / "pages.json"
    cache.unlink()

    second = FakeOllamaClient(
        ocr_responses=["new page 1", "new page 2"],
        analyze_response={
            "summary": "Nouveau résumé.",
            "tags": ["fantastique"],
            "characters": ["Obélix"],
            "locations": [],
            "notable_people": [],
        },
    )
    analysis = process_cbz(path, vlm_client=second, llm_client=second, force=True)
    assert analysis.summary == "Nouveau résumé."
    assert "Obélix" in analysis.characters


def test_page_cache_hits_without_vlm_calls(make_cbz):
    path = make_cbz(pages=2)
    first = _fresh_client()
    process_cbz(path, vlm_client=first, llm_client=first)
    # Remove the sidecar but keep the page cache.
    path.with_suffix(".json").unlink()

    second = FakeOllamaClient(
        ocr_responses=[],  # must stay empty — cache should serve the pages
        analyze_response={
            "summary": "Réanalyse.",
            "tags": [],
            "characters": [],
            "locations": [],
            "notable_people": [],
        },
    )
    process_cbz(path, vlm_client=second, llm_client=second)
    # Only one call and it is the analysis call.
    assert len(second.calls) == 1
    assert second.calls[0].get("format") == "json"


def test_process_cbz_corrupt_zip_raises(tmp_path):
    bogus = tmp_path / "broken.cbz"
    bogus.write_bytes(b"this is not a zip file")
    client = FakeOllamaClient()
    from bd_pipeline.pipeline import PipelineError

    with pytest.raises(PipelineError):
        process_cbz(bogus, vlm_client=client, llm_client=client)
    # No sidecar should have been written.
    assert not bogus.with_suffix(".json").exists()


def test_process_library_builds_index(make_cbz, tmp_path):
    path_a = make_cbz(name="A.cbz", pages=1)
    path_b = make_cbz(name="B.cbz", pages=1)
    client = _fresh_client()
    # Provide enough OCR responses for both books.
    client.ocr_responses = ["a1", "b1"]
    process_library(tmp_path, vlm_client=client, llm_client=client)

    index_md = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert path_a.stem in index_md
    assert path_b.stem in index_md
    assert "aventure" in index_md
