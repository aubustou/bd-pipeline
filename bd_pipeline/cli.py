"""Thin Typer CLI wrapper around the pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from bd_pipeline import cbz, ocr
from bd_pipeline.index import build_index
from bd_pipeline.pipeline import process_cbz, process_library
from bd_pipeline.search import default_vlm_model, search_cbz, search_library

app = typer.Typer(add_completion=False, help="OCR + analyse a CBZ library using local models.")


def _make_ocr_client():
    """Lazily create a ChandraOCR 2 client."""
    from chandra import ChandraOCRClient

    url = ocr.default_chandra_url()
    return ChandraOCRClient(base_url=url)


def _make_llm_client():
    """Lazily create an Ollama client for LLM analysis."""
    import ollama

    host = os.environ.get("OLLAMA_HOST")
    kwargs = {"host": host} if host else {}
    return ollama.Client(**kwargs)


def _make_vlm_client():
    """Lazily create an Ollama client for VLM visual search."""
    import ollama

    host = os.environ.get("OLLAMA_HOST")
    kwargs = {"host": host} if host else {}
    return ollama.Client(**kwargs)


@app.command()
def process(
    path: Path = typer.Argument(
        ..., exists=True, readable=True, help="CBZ file or library folder."
    ),
    force: bool = typer.Option(False, "--force", help="Reprocess even if a sidecar exists."),
    llm: Optional[str] = typer.Option(None, help="Override the Ollama text model."),
) -> None:
    """OCR + analyse one CBZ or all CBZs in a folder, and rebuild INDEX.md."""
    ocr_client = _make_ocr_client()
    llm_client = _make_llm_client()
    if path.is_file():
        analysis = process_cbz(
            path,
            ocr_client=ocr_client,
            llm_client=llm_client,
            llm_model=llm,
            force=force,
        )
        build_index(path.parent)
        typer.echo(f"Wrote {analysis.sidecar_path()}")
    else:
        process_library(
            path,
            ocr_client=ocr_client,
            llm_client=llm_client,
            llm_model=llm,
            force=force,
            progress=lambda xs: tqdm(xs, desc="Albums", unit="cbz"),
        )
        typer.echo(f"Wrote {path / 'INDEX.md'}")


@app.command("index")
def index_cmd(
    path: Path = typer.Argument(..., exists=True, file_okay=False, help="Library folder."),
) -> None:
    """Rebuild INDEX.md from existing sidecars (no model calls)."""
    out = build_index(path)
    typer.echo(f"Wrote {out}")


@app.command()
def show(
    cbz_path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Print the JSON sidecar for a single CBZ."""
    sidecar = cbz_path.with_suffix(".json")
    if not sidecar.exists():
        typer.echo(f"No sidecar at {sidecar}", err=True)
        raise typer.Exit(code=1)
    typer.echo(sidecar.read_text(encoding="utf-8"))


@app.command("ocr")
def ocr_cmd(
    cbz_path: Path = typer.Argument(..., exists=True, dir_okay=False),
) -> None:
    """Debug: print per-page OCR text for a single CBZ."""
    ocr_client = _make_ocr_client()
    for i, (name, image_bytes) in enumerate(cbz.iter_pages(cbz_path), start=1):
        typer.echo(f"--- PAGE {i} ({name}) ---")
        try:
            text = ocr.ocr_page(image_bytes, client=ocr_client)
            typer.echo(text or "(aucun texte)")
        except Exception as exc:
            typer.echo(f"[ERROR: {type(exc).__name__}: {exc}]", err=True)
        typer.echo("")


@app.command()
def search(
    query: str = typer.Argument(..., help="Visual subject to search for (e.g. 'staircase')."),
    path: Path = typer.Argument(
        ..., exists=True, readable=True, help="CBZ file or library folder."
    ),
    vlm: Optional[str] = typer.Option(None, help="Override the Ollama vision model."),
) -> None:
    """Search pages visually matching a subject across one CBZ or a library."""
    vlm_client = _make_vlm_client()
    model = vlm or default_vlm_model()
    if path.is_file():
        hits = search_cbz(path, query, client=vlm_client, model=model)
        pages_str = ", ".join(str(p) for p in hits)
        typer.echo(
            f"{path.stem} :: pages {pages_str}" if hits else f"{path.stem} :: aucun résultat"
        )
    else:
        results = search_library(path, query, client=vlm_client, model=model)
        if not results:
            typer.echo("Aucun résultat.")
        else:
            for title, pages in sorted(results.items()):
                typer.echo(f"{title} :: pages {', '.join(str(p) for p in pages)}")


def main() -> None:  # pragma: no cover - thin wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
