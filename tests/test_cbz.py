from __future__ import annotations

import os
import zipfile

import pytest

from bd_pipeline import cbz


def test_iter_pages_natural_order(make_cbz):
    # Build a CBZ where alphabetical != natural ordering.
    path = make_cbz(pages=0)
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("10.png", b"ten")
        zf.writestr("2.png", b"two")
        zf.writestr("01.png", b"one")

    names = [n for n, _ in cbz.iter_pages(path)]
    assert names == ["01.png", "2.png", "10.png"]


def test_iter_pages_skips_junk(make_cbz):
    path = make_cbz(pages=1, extras={"__MACOSX/._01.png": b"junk", "notes.txt": b"hello"})
    names = [n for n, _ in cbz.iter_pages(path)]
    assert names == ["01.png"]


def test_read_comicinfo_missing_returns_none(make_cbz):
    path = make_cbz(pages=1, with_comicinfo=False)
    assert cbz.read_comicinfo(path) is None


def test_write_comicinfo_roundtrip(make_cbz):
    path = make_cbz(pages=2)
    cbz.write_comicinfo(
        path,
        {
            "Title": "Tintin en Amérique",
            "Summary": "Résumé.",
            "Tags": "aventure, jeunesse",
            "Characters": "Tintin, Milou",
        },
    )
    tree = cbz.read_comicinfo(path)
    assert tree is not None
    fields = {c.tag: c.text for c in tree.getroot()}
    assert fields["Title"] == "Tintin en Amérique"
    assert fields["Summary"] == "Résumé."
    assert fields["Tags"] == "aventure, jeunesse"
    assert fields["Characters"] == "Tintin, Milou"


def test_write_comicinfo_preserves_pages(make_cbz):
    path = make_cbz(pages=2)
    with zipfile.ZipFile(path) as zf:
        pages_before = {n: zf.read(n) for n in zf.namelist() if n.endswith(".png")}
    cbz.write_comicinfo(path, {"Title": "X"})
    with zipfile.ZipFile(path) as zf:
        pages_after = {n: zf.read(n) for n in zf.namelist() if n.endswith(".png")}
    assert pages_before == pages_after


def test_write_comicinfo_merges_existing_fields(make_cbz):
    path = make_cbz(pages=1, with_comicinfo=True)  # starts with <Title>Old</Title>
    cbz.write_comicinfo(path, {"Summary": "New summary"})
    tree = cbz.read_comicinfo(path)
    fields = {c.tag: c.text for c in tree.getroot()}
    assert fields["Title"] == "Old"  # preserved
    assert fields["Summary"] == "New summary"  # added


def test_write_comicinfo_atomic_on_failure(make_cbz, monkeypatch):
    path = make_cbz(pages=1)
    original = path.read_bytes()

    def boom(*a, **kw):  # noqa: ANN001
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cbz.write_comicinfo(path, {"Title": "Will not stick"})

    # Original file untouched; no stray tmp files left in the directory.
    assert path.read_bytes() == original
    leftovers = [p for p in path.parent.iterdir() if p.name.endswith(".cbz.tmp")]
    assert leftovers == []


def test_page_count(make_cbz):
    path = make_cbz(pages=5)
    assert cbz.page_count(path) == 5
