"""End-to-end orchestration: OCR + analyze + sidecar + ComicInfo + library index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from bd_pipeline import analyze, cbz, ocr
from bd_pipeline.index import build_index
from bd_pipeline.models import BookAnalysis

CACHE_DIRNAME = ".bd_pipeline_cache"


class PipelineError(RuntimeError):
    pass


def _load_sidecar(path: Path) -> Optional[BookAnalysis]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return BookAnalysis.model_validate(data)
    except Exception:
        return None


def _save_sidecar(analysis: BookAnalysis) -> Path:
    sidecar = analysis.sidecar_path()
    sidecar.write_text(
        analysis.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    return sidecar


def _cache_dir(cbz_path: Path) -> Path:
    return cbz_path.parent / CACHE_DIRNAME / cbz_path.stem


def _load_page_cache(cbz_path: Path) -> Optional[list[str]]:
    f = _cache_dir(cbz_path) / "pages.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        return None
    return None


def _save_page_cache(cbz_path: Path, pages: list[str]) -> None:
    d = _cache_dir(cbz_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")


def _ocr_all_pages(cbz_path: Path, *, vlm_client, vlm_model: str) -> list[str]:
    cached = _load_page_cache(cbz_path)
    if cached is not None:
        return cached
    pages: list[str] = []
    try:
        for _, image_bytes in cbz.iter_pages(cbz_path):
            pages.append(ocr.ocr_page(image_bytes, client=vlm_client, model=vlm_model))
    except Exception as exc:
        raise PipelineError(f"Failed to read pages from {cbz_path}: {exc}") from exc
    _save_page_cache(cbz_path, pages)
    return pages


def process_cbz(
    cbz_path: Path,
    *,
    vlm_client,
    llm_client,
    vlm_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    force: bool = False,
) -> BookAnalysis:
    """OCR + analyse a single CBZ. Returns the analysis and writes all artefacts."""
    cbz_path = Path(cbz_path)
    if not force:
        cached = _load_sidecar(cbz_path)
        if cached is not None:
            return cached

    vlm_model = vlm_model or ocr.default_vlm_model()
    llm_model = llm_model or analyze.default_llm_model()

    pages = _ocr_all_pages(cbz_path, vlm_client=vlm_client, vlm_model=vlm_model)
    count = cbz.page_count(cbz_path)

    raw = analyze.analyze_book(cbz_path.stem, pages, client=llm_client, model=llm_model)
    analysis = BookAnalysis(
        title=cbz_path.stem,
        path=str(cbz_path),
        page_count=count,
        summary=raw["summary"],
        tags=raw["tags"],
        characters=raw["characters"],
        locations=raw["locations"],
        notable_people=raw["notable_people"],
        vlm_model=vlm_model,
        llm_model=llm_model,
    )
    _save_sidecar(analysis)
    _write_comicinfo(cbz_path, analysis)
    return analysis


def _write_comicinfo(cbz_path: Path, a: BookAnalysis) -> None:
    fields = {
        "Title": a.title,
        "Summary": a.summary,
        "Tags": ", ".join(a.tags),
        "Characters": ", ".join(a.characters),
        "Locations": ", ".join(a.locations),
        "LanguageISO": "fr",
    }
    cbz.write_comicinfo(cbz_path, fields)


def iter_cbz(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.cbz")):
        if CACHE_DIRNAME in path.parts:
            continue
        yield path


def process_library(
    root: Path,
    *,
    vlm_client,
    llm_client,
    vlm_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    force: bool = False,
    progress=None,
) -> list[BookAnalysis]:
    root = Path(root)
    results: list[BookAnalysis] = []
    paths = list(iter_cbz(root))
    iterator = progress(paths) if progress else paths
    for cbz_path in iterator:
        results.append(
            process_cbz(
                cbz_path,
                vlm_client=vlm_client,
                llm_client=llm_client,
                vlm_model=vlm_model,
                llm_model=llm_model,
                force=force,
            )
        )
    build_index(root)
    return results
