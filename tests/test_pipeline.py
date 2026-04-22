from __future__ import annotations

import json

import pytest

from bd_pipeline import cbz
from bd_pipeline.pipeline import CACHE_DIRNAME, process_cbz, process_library
from tests.conftest import FakeChandraClient, FakeOllamaClient

_DEFAULT_ANALYZE = {
    "summary": "Un album d'aventure.",
    "tags": ["aventure"],
    "characters": ["Tintin", "Milou"],
    "locations": ["Bruxelles"],
    "notable_people": [],
}


def _fresh_ocr_client(pages: int = 3) -> FakeChandraClient:
    return FakeChandraClient([f"page {i} text" for i in range(1, pages + 1)])


def _fresh_llm_client(analyze_response: dict | None = None) -> FakeOllamaClient:
    return FakeOllamaClient(analyze_response=analyze_response or _DEFAULT_ANALYZE)


def test_process_cbz_writes_sidecar_and_comicinfo(make_cbz):
    path = make_cbz(pages=3)
    ocr = _fresh_ocr_client(3)
    llm = _fresh_llm_client()
    process_cbz(path, ocr_client=ocr, llm_client=llm)

    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["title"] == path.stem
    assert data["page_count"] == 3
    assert data["summary"] == "Un album d'aventure."
    assert data["vlm_model"] == "chandra-ocr-2"

    tree = cbz.read_comicinfo(path)
    fields = {c.tag: c.text for c in tree.getroot()}
    assert fields["Summary"] == "Un album d'aventure."
    assert fields["Tags"] == "aventure"
    assert "Tintin" in fields["Characters"]
    assert fields["LanguageISO"] == "fr"
    # 3 OCR calls (parse_image) + 1 analyse call (chat with format=json)
    assert len(ocr.calls) == 3
    assert sum(1 for c in llm.calls if c.get("format") == "json") == 1


def test_process_cbz_resumable_skips_all_calls(make_cbz):
    path = make_cbz(pages=2)
    process_cbz(path, ocr_client=_fresh_ocr_client(2), llm_client=_fresh_llm_client())

    ocr2 = FakeChandraClient()  # empty — must NOT be called
    llm2 = FakeOllamaClient()
    again = process_cbz(path, ocr_client=ocr2, llm_client=llm2)
    assert ocr2.calls == []
    assert llm2.calls == []
    assert again.title == path.stem


def test_process_cbz_force_reruns(make_cbz):
    path = make_cbz(pages=2)
    process_cbz(path, ocr_client=_fresh_ocr_client(2), llm_client=_fresh_llm_client())

    # Delete the page cache so force=True actually re-OCRs.
    cache = path.parent / CACHE_DIRNAME / path.stem / "pages.json"
    cache.unlink()

    ocr2 = FakeChandraClient(["new page 1", "new page 2"])
    llm2 = _fresh_llm_client(
        {
            "summary": "Nouveau résumé.",
            "tags": ["fantastique"],
            "characters": ["Obélix"],
            "locations": [],
            "notable_people": [],
        }
    )
    analysis = process_cbz(path, ocr_client=ocr2, llm_client=llm2, force=True)
    assert analysis.summary == "Nouveau résumé."
    assert "Obélix" in analysis.characters


def test_page_cache_hits_without_ocr_calls(make_cbz):
    path = make_cbz(pages=2)
    process_cbz(path, ocr_client=_fresh_ocr_client(2), llm_client=_fresh_llm_client())
    # Remove the sidecar but keep the page cache.
    path.with_suffix(".json").unlink()

    ocr2 = FakeChandraClient()  # must stay empty — cache should serve the pages
    llm2 = _fresh_llm_client(
        {
            "summary": "Réanalyse.",
            "tags": [],
            "characters": [],
            "locations": [],
            "notable_people": [],
        }
    )
    process_cbz(path, ocr_client=ocr2, llm_client=llm2)
    assert ocr2.calls == []
    assert len(llm2.calls) == 1
    assert llm2.calls[0].get("format") == "json"


def test_process_cbz_corrupt_zip_raises(tmp_path):
    bogus = tmp_path / "broken.cbz"
    bogus.write_bytes(b"this is not a zip file")
    from bd_pipeline.pipeline import PipelineError

    with pytest.raises(PipelineError):
        process_cbz(bogus, ocr_client=FakeChandraClient(), llm_client=FakeOllamaClient())
    assert not bogus.with_suffix(".json").exists()


def test_process_library_builds_index(make_cbz, tmp_path):
    make_cbz(name="A.cbz", pages=1)
    make_cbz(name="B.cbz", pages=1)
    ocr = FakeChandraClient(["a1", "b1"])
    llm = _fresh_llm_client()
    process_library(tmp_path, ocr_client=ocr, llm_client=llm)

    index_md = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "A" in index_md
    assert "B" in index_md
    assert "aventure" in index_md
