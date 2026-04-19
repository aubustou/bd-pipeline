"""CBZ (zip-of-images) read/write and ComicInfo.xml handling."""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
COMICINFO_NAME = "ComicInfo.xml"

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(name: str) -> list:
    return [int(p) if p.isdigit() else p.lower() for p in _NUM_RE.split(name)]


def _is_page(name: str) -> bool:
    if name.endswith("/"):
        return False
    base = name.rsplit("/", 1)[-1]
    if not base or base.startswith("."):
        return False
    if "__MACOSX" in name:
        return False
    return Path(name).suffix.lower() in IMAGE_EXTS


def iter_pages(cbz_path: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for each image page in natural sort order."""
    with zipfile.ZipFile(cbz_path) as zf:
        names = [n for n in zf.namelist() if _is_page(n)]
        names.sort(key=_natural_key)
        for name in names:
            yield name, zf.read(name)


def page_count(cbz_path: Path) -> int:
    with zipfile.ZipFile(cbz_path) as zf:
        return sum(1 for n in zf.namelist() if _is_page(n))


def read_comicinfo(cbz_path: Path) -> Optional[ET.ElementTree]:
    """Return the parsed ComicInfo.xml tree, or None if absent."""
    with zipfile.ZipFile(cbz_path) as zf:
        for name in zf.namelist():
            if name.rsplit("/", 1)[-1] == COMICINFO_NAME:
                data = zf.read(name)
                return ET.ElementTree(ET.fromstring(data))
    return None


def _build_comicinfo_xml(fields: dict[str, str]) -> bytes:
    root = ET.Element("ComicInfo")
    # Deterministic ordering keeps diffs readable.
    for key in sorted(fields):
        value = fields[key]
        if value is None or value == "":
            continue
        el = ET.SubElement(root, key)
        el.text = value
    ET.indent(root, space="  ")
    return b'<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="utf-8")


def write_comicinfo(cbz_path: Path, fields: dict[str, str]) -> None:
    """Merge `fields` into ComicInfo.xml inside the CBZ, atomically.

    Existing entries not listed in `fields` are preserved.
    """
    existing: dict[str, str] = {}
    tree = read_comicinfo(cbz_path)
    if tree is not None:
        for child in tree.getroot():
            if child.text is not None:
                existing[child.tag] = child.text
    existing.update({k: v for k, v in fields.items() if v not in (None, "")})
    new_xml = _build_comicinfo_xml(existing)

    # Write a new zip alongside the original, copy non-ComicInfo entries,
    # then atomically replace the source.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=cbz_path.stem + ".",
        suffix=".cbz.tmp",
        dir=cbz_path.parent,
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with (
            zipfile.ZipFile(cbz_path, "r") as src,
            zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as dst,
        ):
            written_comicinfo = False
            for item in src.infolist():
                base = item.filename.rsplit("/", 1)[-1]
                if base == COMICINFO_NAME:
                    dst.writestr(item.filename, new_xml)
                    written_comicinfo = True
                else:
                    dst.writestr(item, src.read(item.filename))
            if not written_comicinfo:
                dst.writestr(COMICINFO_NAME, new_xml)
        os.replace(tmp_path, cbz_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
