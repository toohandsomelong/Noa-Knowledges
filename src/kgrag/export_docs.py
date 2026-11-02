"""Export documents from a source directory to structured JSON or Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO_ROOT / "data" / "docs"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "processed" / "docs"
_DEFAULT_MANIFEST = _REPO_ROOT / "data" / "processed" / "docs_manifest.jsonl"

_PASSTHROUGH_SUFFIXES = {".txt", ".md"}
_PDF_SUFFIX = ".pdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def _make_docling_converter() -> Any:
    try:
        import torch
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError("docling is required to export PDF files") from exc

    device = (
        AcceleratorDevice.CUDA
        if torch.cuda.is_available()
        else AcceleratorDevice.CPU
    )
    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
    )
    options.do_ocr = False
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
        }
    )


def _extract_pdf_docling(
    source: Path,
    converter: Any,
) -> tuple[dict[str, Any], str, int, int, int]:
    result = converter.convert(str(source))
    document = result.document
    data = document.export_to_dict()
    markdown = document.export_to_markdown()

    page_count = len(data.get("pages", []))
    texts = data.get("texts", [])
    tables = data.get("tables", [])
    character_count = sum(len(item.get("text", "")) for item in texts)

    pages_with_content: set[int] = set()
    for item in texts + tables:
        for prov in item.get("prov", []):
            page_number = prov.get("page_no")
            if page_number is not None:
                pages_with_content.add(page_number)

    empty_pages = max(0, page_count - len(pages_with_content))
    return data, markdown, page_count, empty_pages, character_count


def _output_paths(
    source: Path,
    input_root: Path,
    output_root: Path,
) -> tuple[Path, Path | None]:
    relative = source.relative_to(input_root)
    markdown = (output_root / "md" / relative).with_suffix(".md")
    if source.suffix.lower() == _PDF_SUFFIX:
        json_path = (output_root / "json" / relative).with_suffix(".json")
        return markdown, json_path
    return markdown, None


def _converter_name(source: Path) -> str:
    suffix = source.suffix.lower()
    if suffix == _PDF_SUFFIX:
        return "docling"
    if suffix in _PASSTHROUGH_SUFFIXES:
        return "passthrough"
    return ""


def _export_one(
    source: Path,
    input_root: Path,
    output_root: Path,
    docling_converter: Any,
) -> dict[str, Any]:
    relative = source.relative_to(input_root)
    markdown_output, json_output = _output_paths(source, input_root, output_root)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "source_path": relative.as_posix(),
        "source_sha256": _sha256(source),
        "output_path": "",
        "md_path": markdown_output.relative_to(output_root).as_posix(),
        "json_path": (
            json_output.relative_to(output_root).as_posix()
            if json_output is not None
            else ""
        ),
        "character_count": 0,
        "page_count": 0,
        "empty_pages": 0,
        "converter": _converter_name(source),
        "converter_version": "",
        "warning": "",
        "error": "",
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    suffix = source.suffix.lower()
    if suffix in _PASSTHROUGH_SUFFIXES:
        content = source.read_text(encoding="utf-8")
        markdown_output.write_text(content, encoding="utf-8")
        entry["output_path"] = entry["md_path"]
        entry["status"] = "copied"
        entry["character_count"] = len(content)
    elif suffix == _PDF_SUFFIX:
        if docling_converter is None:
            raise RuntimeError("docling converter was not initialized")
        data, markdown, page_count, empty_pages, character_count = _extract_pdf_docling(
            source,
            docling_converter,
        )
        assert json_output is not None
        json_output.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        markdown_output.write_text(markdown, encoding="utf-8")
        entry["output_path"] = entry["json_path"]
        entry["page_count"] = page_count
        entry["empty_pages"] = empty_pages
        entry["character_count"] = character_count
        entry["converter_version"] = _package_version("docling")
        entry["status"] = (
            "exported"
            if data.get("texts") or data.get("tables") or data.get("pictures")
            else "empty_output"
        )
        if empty_pages:
            entry["warning"] = f"{empty_pages} page(s) have no text or table content"
    else:
        entry["status"] = "unsupported"
        entry["warning"] = f"unsupported extension: {suffix or '(none)'}"
        return entry

    return entry


def export_documents(
    input_root: Path,
    output_root: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    suffix_set = _PASSTHROUGH_SUFFIXES | {_PDF_SUFFIX}
    needs_docling = any(
        path.is_file() and path.suffix.lower() == _PDF_SUFFIX
        for path in input_root.rglob("*")
    )
    docling_converter = _make_docling_converter() if needs_docling else None

    entries: list[dict[str, Any]] = []

    for source in sorted(path for path in input_root.rglob("*") if path.is_file()):
        if source.suffix.lower() not in suffix_set:
            relative = source.relative_to(input_root)
            entries.append(
                {
                    "source_path": relative.as_posix(),
                    "source_sha256": _sha256(source),
                    "output_path": "",
                    "md_path": "",
                    "json_path": "",
                    "character_count": 0,
                    "page_count": 0,
                    "empty_pages": 0,
                    "converter": "",
                    "converter_version": "",
                    "status": "unsupported",
                    "warning": f"unsupported extension: {source.suffix or '(none)'}",
                    "error": "",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        try:
            entries.append(
                _export_one(source, input_root, output_root, docling_converter)
            )
        except Exception as exc:
            relative = source.relative_to(input_root)
            entries.append(
                {
                    "source_path": relative.as_posix(),
                    "source_sha256": _sha256(source),
                    "output_path": "",
                    "md_path": "",
                    "json_path": "",
                    "character_count": 0,
                    "page_count": 0,
                    "empty_pages": 0,
                    "converter": _converter_name(source),
                    "converter_version": "",
                    "status": "failed",
                    "warning": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    return entries


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, dest="input_root", default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, dest="output_root", default=_DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, dest="manifest_path", default=_DEFAULT_MANIFEST)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    entries = export_documents(args.input_root, args.output_root, args.manifest_path)

    for entry in entries:
        print(
            f"{entry['status']:<12} {entry['source_path']} "
            f"-> {entry['output_path'] or '-'}"
        )
        if entry["error"]:
            print(f"  error: {entry['error']}")
        elif entry["warning"]:
            print(f"  warning: {entry['warning']}")

    return int(any(entry["status"] == "failed" for entry in entries))


if __name__ == "__main__":
    raise SystemExit(main())
