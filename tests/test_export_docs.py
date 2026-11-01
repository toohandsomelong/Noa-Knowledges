"""Tests for document export."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from kgrag.export_docs import (
    _build_parser,
    _extract_pdf_text,
    _output_suffix,
    export_documents,
)


def test_export_cli_has_zero_argument_defaults():
    args = _build_parser().parse_args([])
    assert args.input_root.parts[-2:] == ("data", "docs")
    assert args.output_root.parts[-2:] == ("processed", "docs")
    assert args.manifest_path.name == "docs_manifest.jsonl"


def test_output_suffix():
    assert _output_suffix(Path("a.pdf")) == ".txt"
    assert _output_suffix(Path("a.docx")) == ".md"
    assert _output_suffix(Path("a.txt")) == ".md"


def test_passthrough_export(tmp_path: Path):
    input_root = tmp_path / "in"
    input_root.mkdir()
    source = input_root / "a.txt"
    source.write_text("Điều 1\n", encoding="utf-8")

    output_root = tmp_path / "out"
    manifest = tmp_path / "manifest.jsonl"
    entries = export_documents(input_root, output_root, manifest)

    assert entries[0]["status"] == "copied"
    assert entries[0]["converter"] == "passthrough"
    assert entries[0]["output_path"] == "a.md"
    assert (output_root / "a.md").read_text(encoding="utf-8") == "Điều 1\n"
    assert manifest.exists()


def test_pdf_export_uses_pypdfium2(tmp_path: Path):
    pdfs = sorted(Path("data/docs").glob("*.pdf"))
    if not pdfs:
        pytest.skip("no PDF fixture available")

    input_root = tmp_path / "in"
    input_root.mkdir()
    source = pdfs[0]
    shutil.copy2(source, input_root / source.name)

    output_root = tmp_path / "out"
    manifest = tmp_path / "manifest.jsonl"
    entries = export_documents(input_root, output_root, manifest)

    entry = entries[0]
    assert entry["converter"] == "pypdfium2"
    assert entry["output_path"].endswith(".txt")
    assert entry["page_count"] > 0
    assert entry["status"] in {"exported", "empty_output"}
    assert (output_root / entry["output_path"]).exists()


def test_pdf_text_extraction_starts_with_page_marker():
    pdfs = sorted(Path("data/docs").glob("*.pdf"))
    if not pdfs:
        pytest.skip("no PDF fixture available")

    text, page_count, empty_pages = _extract_pdf_text(pdfs[0])

    assert page_count > 0
    assert text.startswith("===== PAGE 1 =====")
    assert 0 <= empty_pages <= page_count
