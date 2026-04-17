from __future__ import annotations

import json
import zipfile
from pathlib import Path

from bd_pipeline.index import build_index
from bd_pipeline.models import BookAnalysis


def _write_book(
    folder: Path,
    title: str,
    *,
    tags=(),
    characters=(),
    locations=(),
    notable_people=(),
):
    cbz_path = folder / f"{title}.cbz"
    # Empty zip is enough; build_index only needs the file to exist.
    with zipfile.ZipFile(cbz_path, "w") as zf:
        zf.writestr("01.png", b"x")
    analysis = BookAnalysis(
        title=title,
        path=str(cbz_path),
        page_count=1,
        summary=f"Résumé de {title}.",
        tags=list(tags),
        characters=list(characters),
        locations=list(locations),
        notable_people=list(notable_people),
    )
    cbz_path.with_suffix(".json").write_text(
        analysis.model_dump_json(indent=2), encoding="utf-8"
    )


def test_build_index_lists_all_books(tmp_path):
    _write_book(tmp_path, "AlbumA", tags=["aventure"], characters=["Tintin"])
    _write_book(tmp_path, "AlbumB", tags=["humour"], characters=["Astérix"])
    out = build_index(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "AlbumA" in text
    assert "AlbumB" in text
    assert "aventure" in text
    assert "humour" in text


def test_build_index_dedups_names_case_insensitively(tmp_path):
    _write_book(tmp_path, "Book1", characters=["Tintin"])
    _write_book(tmp_path, "Book2", characters=["tintin"])
    text = build_index(tmp_path).read_text(encoding="utf-8")
    # Both books are listed under a single Tintin entry.
    lines = [l for l in text.splitlines() if "Tintin" in l or "tintin" in l]
    name_lines = [l for l in lines if l.startswith("- **")]
    assert len(name_lines) == 1
    assert "Book1" in name_lines[0] and "Book2" in name_lines[0]


def test_build_index_prefers_accented_form(tmp_path):
    _write_book(tmp_path, "B1", notable_people=["Herge"])
    _write_book(tmp_path, "B2", notable_people=["Hergé"])
    text = build_index(tmp_path).read_text(encoding="utf-8")
    assert "**Hergé**" in text
    # And the unaccented spelling should not appear as its own entry.
    assert "**Herge**" not in text


def test_build_index_distinguishes_truly_different_names(tmp_path):
    _write_book(tmp_path, "B1", locations=["Brussels"])
    _write_book(tmp_path, "B2", locations=["Bruxelles"])
    text = build_index(tmp_path).read_text(encoding="utf-8")
    assert "**Brussels**" in text
    assert "**Bruxelles**" in text


def test_build_index_empty_dir(tmp_path):
    out = build_index(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "0 album" in text
    assert "Aucun album trouvé" in text
