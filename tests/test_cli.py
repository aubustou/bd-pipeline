from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from bd_pipeline import cli
from tests.conftest import FakeChandraClient, FakeOllamaClient


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def patch_clients(monkeypatch):
    """Replace the client factories with fakes and expose them."""
    fake_ocr = FakeChandraClient(["page text 1", "page text 2", "page text 3"])
    fake_llm = FakeOllamaClient(
        analyze_response={
            "summary": "résumé CLI",
            "tags": ["test"],
            "characters": ["Tintin"],
            "locations": [],
            "notable_people": [],
        }
    )
    monkeypatch.setattr(cli, "_make_ocr_client", lambda: fake_ocr)
    monkeypatch.setattr(cli, "_make_llm_client", lambda: fake_llm)
    return fake_ocr, fake_llm


def test_cli_process_writes_sidecar(runner, patch_clients, make_cbz):
    path = make_cbz(name="CLI.cbz", pages=2)
    result = runner.invoke(cli.app, ["process", str(path)])
    assert result.exit_code == 0, result.stdout
    sidecar = path.with_suffix(".json")
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["summary"] == "résumé CLI"


def test_cli_index_does_not_call_models(runner, patch_clients, make_cbz):
    path = make_cbz(name="X.cbz", pages=1)
    # Pre-create a sidecar so build_index has something to ingest.
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "title": "X",
                "path": str(path),
                "page_count": 1,
                "summary": "s",
                "tags": ["t"],
                "characters": [],
                "locations": [],
                "notable_people": [],
            }
        ),
        encoding="utf-8",
    )

    fake_ocr, fake_llm = patch_clients
    result = runner.invoke(cli.app, ["index", str(path.parent)])
    assert result.exit_code == 0, result.stdout
    assert (path.parent / "INDEX.md").exists()
    assert fake_ocr.calls == []
    assert fake_llm.calls == []


def test_cli_show_prints_sidecar(runner, make_cbz):
    path = make_cbz(name="Show.cbz", pages=1)
    sidecar = path.with_suffix(".json")
    sidecar.write_text('{"hello": "world"}', encoding="utf-8")
    result = runner.invoke(cli.app, ["show", str(path)])
    assert result.exit_code == 0
    assert "world" in result.stdout


def test_cli_show_missing_sidecar_errors(runner, make_cbz):
    path = make_cbz(name="NoSidecar.cbz", pages=1)
    result = runner.invoke(cli.app, ["show", str(path)])
    assert result.exit_code != 0


def test_cli_ocr_prints_pages(runner, monkeypatch, make_cbz):
    path = make_cbz(name="OCR.cbz", pages=2)
    fake_ocr = FakeChandraClient(["alpha", "beta"])
    monkeypatch.setattr(cli, "_make_ocr_client", lambda: fake_ocr)
    result = runner.invoke(cli.app, ["ocr", str(path)])
    assert result.exit_code == 0, result.stdout
    assert "alpha" in result.stdout
    assert "beta" in result.stdout


def test_cli_unknown_path_exits_nonzero(runner):
    result = runner.invoke(cli.app, ["process", "/does/not/exist.cbz"])
    assert result.exit_code != 0


def test_cli_search_single_cbz_prints_hits(runner, make_cbz, monkeypatch):
    path = make_cbz(name="Srch.cbz", pages=2)
    fake_vlm = FakeOllamaClient(ocr_responses=["non", "oui"])
    monkeypatch.setattr(cli, "_make_vlm_client", lambda: fake_vlm)
    result = runner.invoke(cli.app, ["search", "escalier", str(path)])
    assert result.exit_code == 0, result.stdout
    assert "Srch" in result.stdout
    assert "2" in result.stdout


def test_cli_search_no_hits_prints_message(runner, make_cbz, monkeypatch):
    path = make_cbz(name="Srch.cbz", pages=1)
    fake_vlm = FakeOllamaClient(ocr_responses=["non"])
    monkeypatch.setattr(cli, "_make_vlm_client", lambda: fake_vlm)
    result = runner.invoke(cli.app, ["search", "dragon", str(path)])
    assert result.exit_code == 0, result.stdout
    assert "aucun résultat" in result.stdout
