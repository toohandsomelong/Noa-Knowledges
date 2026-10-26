"""Export documents from a source directory to Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_PASSTHROUGH_SUFFIXES = {".txt", ".md"}
_MARKITDOWN_SUFFIXES = {
    ".csv",
    ".docx",
    ".htm",
    ".html",
    ".json",
    ".pdf",
    ".pptx",
    ".xlsx",
    ".xml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _markitdown_version() -> str:
    try:
        return version("markitdown")
    except PackageNotFoundError:
        return "unavailable"


def _extract_markdown(result: Any) -> str:
    for attribute in ("markdown", "text_content"):
        value = getattr(result, attribute, None)
        if isinstance(value, str):
            return value
    raise ValueError("MarkItDown result does not contain Markdown text")


def _export_one(
    source: Path,
    input_root: Path,
    output_root: Path,
    converter: Any,
    converter_version: str,
) -> dict[str, Any]:
    relative = source.relative_to(input_root)
    output = (output_root / relative).with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "source_path": relative.as_posix(),
        "source_sha256": _sha256(source),
        "output_path": output.relative_to(output_root).as_posix(),
        "character_count": 0,
        "converter_version": converter_version,
        "warning": "",
        "error": "",
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    suffix = source.suffix.lower()
    if suffix in _PASSTHROUGH_SUFFIXES:
        content = source.read_text(encoding="utf-8")
        entry["converter"] = "passthrough"
        entry["status"] = "copied"
    elif suffix in _MARKITDOWN_SUFFIXES:
        if converter is None:
            raise RuntimeError("MarkItDown is not installed")
        result = converter.convert_local(str(source))
        content = _extract_markdown(result)
        entry["converter"] = f"markitdown{suffix or ''}"
        entry["status"] = "exported" if content.strip() else "empty_output"
        if not content.strip():
            entry["warning"] = "MarkItDown returned no text"
    else:
        entry["converter"] = ""
        entry["status"] = "unsupported"
        entry["warning"] = f"unsupported extension: {suffix or '(none)'}"
        return entry

    output.write_text(content, encoding="utf-8")
    entry["character_count"] = len(content)
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

    suffix_set = _PASSTHROUGH_SUFFIXES | _MARKITDOWN_SUFFIXES
    needs_markitdown = any(
        path.is_file() and path.suffix.lower() in _MARKITDOWN_SUFFIXES
        for path in input_root.rglob("*")
    )

    converter = None
    if needs_markitdown:
        from markitdown import MarkItDown

        converter = MarkItDown(enable_plugins=False)

    converter_version = _markitdown_version() if needs_markitdown else ""
    entries: list[dict[str, Any]] = []

    for source in sorted(path for path in input_root.rglob("*") if path.is_file()):
        if source.suffix.lower() not in suffix_set:
            relative = source.relative_to(input_root)
            entries.append(
                {
                    "source_path": relative.as_posix(),
                    "source_sha256": _sha256(source),
                    "output_path": "",
                    "character_count": 0,
                    "converter": "",
                    "converter_version": converter_version,
                    "status": "unsupported",
                    "warning": f"unsupported extension: {source.suffix or '(none)'}",
                    "error": "",
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue

        try:
            entries.append(
                _export_one(
                    source,
                    input_root,
                    output_root,
                    converter,
                    converter_version,
                )
            )
        except Exception as exc:
            relative = source.relative_to(input_root)
            entries.append(
                {
                    "source_path": relative.as_posix(),
                    "source_sha256": _sha256(source),
                    "output_path": "",
                    "character_count": 0,
                    "converter": f"markitdown{source.suffix.lower()}",
                    "converter_version": converter_version,
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
    parser.add_argument("--input", required=True, type=Path, dest="input_root")
    parser.add_argument("--output", required=True, type=Path, dest="output_root")
    parser.add_argument("--manifest", required=True, type=Path, dest="manifest_path")
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
