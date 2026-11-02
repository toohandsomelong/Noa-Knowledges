"""Tests for document export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kgrag.export_docs import _build_parser, export_documents


def test_export_cli_has_zero_argument_defaults():
    args = _build_parser().parse_args([])
    assert args.input_root.parts[-2:] == ("data", "docs")
    assert args.output_root.parts[-2:] == ("processed", "docs")
    assert args.manifest_path.name == "docs_manifest.jsonl"


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
    assert entries[0]["output_path"] == "md/a.md"
    assert entries[0]["md_path"] == "md/a.md"
    assert entries[0]["json_path"] == ""
    assert (output_root / "md" / "a.md").read_text(encoding="utf-8") == "Điều 1\n"
    assert manifest.exists()


def test_pdf_export_uses_docling(tmp_path: Path):
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
    assert entry["converter"] == "docling"
    assert entry["output_path"].endswith(".json")
    assert entry["json_path"] == entry["output_path"]
    assert entry["md_path"].endswith(".md")
    assert entry["page_count"] > 0
    assert entry["character_count"] > 0
    assert entry["status"] == "exported"

    json_path = output_root / entry["json_path"]
    md_path = output_root / entry["md_path"]
    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["texts"]
    assert "body" in data
    assert "pages" in data
    assert len(data["pages"]) == entry["page_count"]
    assert md_path.read_text(encoding="utf-8").strip()
