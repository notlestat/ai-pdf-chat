"""PDF validation and upload to the Files API."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import anthropic
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from pdf_chat.client import FILES_BETA

# A single request tops out at 600 pages against a 1M-context model. Past that the
# document has to be split, which this app doesn't do — so refuse clearly rather
# than letting the API return a confusing error later.
MAX_PAGES = 600

# The Files API accepts far larger uploads, but a PDF this big will produce a very
# expensive first question. Warn rather than block: the caller may well mean it.
LARGE_DOC_PAGES = 150


@dataclass(frozen=True)
class Document:
    """An uploaded PDF, ready to be referenced in a conversation."""

    file_id: str
    filename: str
    pages: int
    size_bytes: int

    @property
    def is_large(self) -> bool:
        return self.pages >= LARGE_DOC_PAGES


class PDFError(ValueError):
    """The uploaded file isn't a PDF we can work with."""


def fingerprint(data: bytes) -> str:
    """Stable ID for a file's contents.

    Streamlit reruns the whole script on every interaction, so we key the
    already-uploaded document on this to avoid re-uploading the same PDF on
    every question the user asks.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def count_pages(data: bytes) -> int:
    """Page count, or a clear error if the file isn't readable as a PDF."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = len(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise PDFError(f"Couldn't read this as a PDF: {exc}") from exc

    if pages == 0:
        raise PDFError("This PDF has no pages.")
    if pages > MAX_PAGES:
        raise PDFError(
            f"This PDF is {pages} pages. The limit for a single request is "
            f"{MAX_PAGES}. Split it into smaller files first."
        )
    return pages


def upload(client: anthropic.Anthropic, data: bytes, filename: str) -> Document:
    """Validate a PDF and upload it once, returning a reusable reference.

    Uploading beats inlining base64 on every request: the bytes cross the wire a
    single time and each later question just names the file_id.
    """
    pages = count_pages(data)

    uploaded = client.beta.files.upload(
        file=(filename, io.BytesIO(data), "application/pdf"),
        betas=[FILES_BETA],
    )

    return Document(
        file_id=uploaded.id,
        filename=filename,
        pages=pages,
        size_bytes=len(data),
    )
