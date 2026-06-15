"""Load local documents (text, markdown, PDF) into plain text. No network access."""

from __future__ import annotations

from pathlib import Path

_TEXT_EXT = {".txt", ".md", ".markdown", ".rst", ".text"}
_SUPPORTED = _TEXT_EXT | {".pdf"}


class UnsupportedDocument(Exception):
    pass


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in _SUPPORTED


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in _TEXT_EXT:
        return path.read_text(errors="ignore")
    if ext == ".pdf":
        return _load_pdf(path)
    raise UnsupportedDocument(f"unsupported document type: {ext}")


def iter_files(path: str | Path) -> list[Path]:
    """Expand a file or directory into the list of supported document files."""
    p = Path(path).expanduser()
    if p.is_file():
        return [p] if is_supported(p) else []
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.is_file() and is_supported(f))
    return []


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
