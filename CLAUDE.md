# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e ".[dev]"   # install with dev dependencies

pytest                                           # run all tests
pytest tests/test_pipeline.py                   # run one file
pytest tests/test_pipeline.py::test_name        # run one test

ruff check                                       # lint
ruff format                                      # format (line-length=100, target py3.10)
```

## Architecture

**bd-pipeline** is a local-only OCR + analysis pipeline for Franco-Belgian CBZ comic files, using Ollama vision/text models. No cloud calls.

### Data flow

```
CBZ (zip of images)
  → ocr.py       VLM per page → text transcripts  (cached in .bd_pipeline_cache/<stem>/pages.json)
  → analyze.py   LLM on all text → BookAnalysis JSON  (map-reduce for long books)
  → cbz.py       writes ComicInfo.xml back into CBZ
  → sidecar .json next to CBZ
  → index.py     aggregates all sidecars → library INDEX.md
```

### Key modules

| Module | Role |
|--------|------|
| `cli.py` | Typer entry point: `process`, `index`, `show`, `ocr`, `search` |
| `pipeline.py` | Orchestration, caching, `process_cbz()`, `process_library()` |
| `models.py` | `BookAnalysis` Pydantic schema (single source of truth for output shape) |
| `cbz.py` | ZIP I/O, `iter_pages()`, `read_comicinfo()`, `write_comicinfo()` (atomic rename) |
| `ocr.py` | VLM-based page OCR, image downscaling |
| `analyze.py` | LLM analysis, map-reduce chunking, accent/case-insensitive name deduplication |
| `index.py` | Library-wide markdown index from sidecars |
| `search.py` | Visual page search via VLM |
| `prompts.py` | All prompts (French-language) |

### Design notes

- **Resumability**: sidecar JSON present → skip reprocessing; page cache present → skip OCR.
- **Map-reduce**: books with many pages are split into ~12 KB chunks, each analyzed separately, then reduced into one `BookAnalysis`.
- **Name deduplication**: NFKD normalization + accent stripping + case-insensitive merge; prefer the spelling with more diacritics.
- **Ollama models**: default `qwen3.5:9b` for both VLM and LLM; overridable via `BD_VLM_MODEL`/`BD_LLM_MODEL` env vars or `--vlm`/`--llm` flags.
- **Testing**: `FakeOllamaClient` (conftest.py) and `make_cbz()` fixture — no real Ollama needed.
- All user-facing text, prompts, and CLI help strings are in **French**.
