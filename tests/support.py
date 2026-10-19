"""Helpers for building documents in tests."""

from __future__ import annotations

from pathlib import Path


def write_text_pdf(path: Path, pages: list[str]) -> None:
    """Write a tiny text PDF that pypdf can extract, one line per page."""
    path.write_bytes(build_text_pdf(pages))


def build_text_pdf(pages: list[str]) -> bytes:
    font_id = 3 + 2 * len(pages)
    page_ids: list[int] = []
    content_ids: list[int] = []
    streams: list[bytes] = []
    next_id = 3
    for text in pages:
        content_ids.append(next_id)
        page_ids.append(next_id + 1)
        next_id += 2
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        streams.append(f"BT /F1 18 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))

    objects: list[bytes] = [b""] * (font_id + 1)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode()
    for page_id, content_id, stream in zip(page_ids, content_ids, streams, strict=True):
        objects[content_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
    objects[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number in range(1, font_id + 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {font_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for number in range(1, font_id + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += f"trailer << /Size {font_id + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
