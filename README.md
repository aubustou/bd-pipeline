# bd-pipeline

Local pipeline to OCR Franco-Belgian CBZ comics (bande dessinée), generate a
résumé, extract tags, and build a cross-library index of names (characters,
places, notable people).

Everything runs locally through [Ollama](https://ollama.com): a vision model
transcribes pages and a text model produces the structured analysis.

## Install

```bash
pip install -e ".[dev]"
```

Pull the model (defaults can be overridden via env or CLI):

```bash
ollama pull qwen3.5:9b   # multimodal: OCR + summary / tags / NER
```

Qwen 3.5 is multimodal, so a single pull covers both the VLM (page OCR) and
LLM (analysis) roles. The 9B size targets a 12 GB GPU (e.g. RTX 5070); for
tighter VRAM use `qwen3.5:4b`, for more headroom `qwen3.5:27b`. You can still
split the two roles to different models via `BD_VLM_MODEL` / `BD_LLM_MODEL`
or `--vlm` / `--llm` — the previous defaults (`qwen2.5vl:7b`,
`qwen2.5:7b-instruct`) remain usable that way.

## Usage

Process a single album or a whole library:

```bash
bd-pipeline process /path/to/library/
```

Each CBZ gets a `<name>.json` sidecar next to it. Tags, summary, characters,
and locations are also written into `ComicInfo.xml` inside the CBZ so readers
like Komga/Kavita pick them up. A global `INDEX.md` is built at the library
root with a tag cloud and a name index.

Other commands:

```bash
bd-pipeline ocr sample.cbz     # debug: dump per-page OCR text
bd-pipeline show sample.cbz    # print the sidecar JSON
bd-pipeline index /path/to/library/   # rebuild INDEX.md only
```

Env overrides: `BD_VLM_MODEL`, `BD_LLM_MODEL`, `OLLAMA_HOST`.

## Layout

```
bd_pipeline/
  cbz.py        # zip I/O + ComicInfo.xml read/write
  ocr.py        # VLM page OCR via Ollama
  analyze.py    # text LLM: summary + tags + names (map-reduce for long books)
  index.py      # library-wide markdown index
  pipeline.py   # orchestration with resumable cache
  cli.py        # Typer CLI
  models.py     # pydantic schemas
  prompts.py    # French prompts

tests/          # offline tests using a FakeOllamaClient
```

## Tests

```bash
pytest
```
