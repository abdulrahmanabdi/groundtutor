"""Split normalized source text into chunks with stable character offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from groundtutor.types import Chunk

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_FENCE = re.compile(r"^```.*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Cue:
    """One timestamped transcript line."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _Section:
    path: tuple[str, ...]
    start: int
    end: int


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def chunk_text(
    text: str,
    *,
    doc_id: str,
    source: str,
    source_type: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    page: int | None = None,
    slide: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    heading_prefix: tuple[str, ...] = (),
    id_start: int = 0,
) -> list[Chunk]:
    """Chunk ``text`` without crossing Markdown headings.

    ``chunk.text`` is always ``text[chunk.char_start:chunk.char_end]`` after
    newline normalization.
    """
    _check_window(chunk_size, overlap)
    text = normalize_newlines(text)
    chunks: list[Chunk] = []
    for section in _sections(text, heading_prefix):
        for start, end in _windows(text, section.start, section.end, chunk_size, overlap):
            raw = text[start:end]
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())
            start += lead
            end -= trail
            if start >= end:
                continue
            piece = text[start:end]
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{id_start + len(chunks):04d}",
                    doc_id=doc_id,
                    text=piece,
                    source=source,
                    source_type=source_type,
                    char_start=start,
                    char_end=end,
                    page=page,
                    slide=slide,
                    start_time=start_time,
                    end_time=end_time,
                    heading_path=section.path,
                )
            )
    return chunks


def chunk_cues(
    cues: list[Cue],
    *,
    doc_id: str,
    source: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    id_start: int = 0,
    max_gap: float = 5.0,
) -> list[Chunk]:
    """Pack transcript cues into windows that keep the first and last timestamp.

    A silence longer than ``max_gap`` seconds starts a new window, so a later
    topic is not glued to the previous one just because both lines are short.
    """
    _check_window(chunk_size, overlap)
    ordered = sorted(cues, key=lambda cue: (cue.start, cue.end))
    if not ordered:
        return []
    full, spans = _join_cues(ordered)
    chunks: list[Chunk] = []
    for first, last in _cue_groups(ordered, spans, chunk_size, overlap, max_gap):
        start = spans[first][0]
        end = spans[last][1]
        piece = full[start:end]
        if not piece.strip():
            continue
        chunks.append(
            Chunk(
                id=f"{doc_id}#{id_start + len(chunks):04d}",
                doc_id=doc_id,
                text=piece,
                source=source,
                source_type="transcript",
                char_start=start,
                char_end=end,
                start_time=ordered[first].start,
                end_time=ordered[last].end,
            )
        )
    return chunks


def _check_window(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")


def _code_ranges(text: str) -> list[tuple[int, int]]:
    fences = list(_FENCE.finditer(text))
    ranges: list[tuple[int, int]] = []
    for index in range(0, len(fences) - 1, 2):
        ranges.append((fences[index].start(), fences[index + 1].end()))
    if len(fences) % 2 == 1:
        ranges.append((fences[-1].start(), len(text)))
    return ranges


def _inside(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _sections(text: str, prefix: tuple[str, ...]) -> list[_Section]:
    masked = _code_ranges(text)
    matches = [match for match in _HEADING.finditer(text) if not _inside(match.start(), masked)]
    if not matches:
        return [_Section(prefix, 0, len(text))]
    sections: list[_Section] = []
    if matches[0].start() > 0:
        sections.append(_Section(prefix, 0, matches[0].start()))
    stack: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        stack = [item for item in stack if item[0] < level]
        stack.append((level, title))
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(_Section(prefix + tuple(item[1] for item in stack), start, end))
    return sections


def _windows(
    text: str,
    start: int,
    end: int,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if not text[start:end].strip():
        return []
    if end - start <= size:
        return [(start, end)]
    windows: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        window_end = min(pos + size, end)
        if window_end < end:
            snap = text.rfind(" ", pos, window_end)
            if snap > pos:
                window_end = snap
        if window_end <= pos:
            window_end = min(pos + size, end)
        windows.append((pos, window_end))
        if window_end >= end:
            break
        next_pos = window_end - overlap
        if next_pos > pos and next_pos < end and not text[next_pos - 1].isspace():
            back = text.rfind(" ", pos + 1, next_pos + 1)
            if back != -1 and back + 1 > pos:
                next_pos = back + 1
        if next_pos <= pos:
            next_pos = window_end
        pos = next_pos
        if len(windows) > len(text) + 1:
            raise RuntimeError("chunker failed to advance")
    return windows


def _join_cues(cues: list[Cue]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for index, cue in enumerate(cues):
        if index:
            pos += 1
        piece = cue.text.strip()
        start = pos
        parts.append(piece)
        pos += len(piece)
        spans.append((start, pos))
    return "\n".join(parts), spans


def _cue_groups(
    cues: list[Cue],
    spans: list[tuple[int, int]],
    chunk_size: int,
    overlap: int,
    max_gap: float,
) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    index = 0
    count = len(spans)
    while index < count:
        last = index
        while last + 1 < count and spans[last + 1][1] - spans[index][0] <= chunk_size:
            if cues[last + 1].start - cues[last].end > max_gap:
                break
            last += 1
        groups.append((index, last))
        if last == count - 1:
            break
        if cues[last + 1].start - cues[last].end > max_gap:
            index = last + 1
            continue
        next_index = last + 1
        end_char = spans[last][1]
        cursor = last
        while cursor > index and end_char - spans[cursor][0] <= overlap:
            cursor -= 1
        if cursor < last:
            next_index = cursor + 1
        if next_index <= index:
            next_index = index + 1
        index = next_index
    return groups
