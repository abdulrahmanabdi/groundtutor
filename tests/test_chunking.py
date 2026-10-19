import pytest

from groundtutor.chunking import Cue, chunk_cues, chunk_text


def test_markdown_heading_path_and_exact_offsets():
    text = "# Neural nets\n\n## Backprop\n\nThe chain rule computes gradients.\n"
    chunks = chunk_text(
        text,
        doc_id="notes",
        source="notes.md",
        source_type="markdown",
        chunk_size=1000,
        overlap=50,
    )
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("Neural nets",)
    assert chunks[1].heading_path == ("Neural nets", "Backprop")
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


def test_code_fence_is_not_a_heading():
    text = "# Real\n\n```\n## Not a heading\n```\n\n## After\n\nBody.\n"
    chunks = chunk_text(
        text,
        doc_id="notes",
        source="notes.md",
        source_type="markdown",
        chunk_size=1000,
        overlap=20,
    )
    paths = [chunk.heading_path for chunk in chunks]
    assert ("Real", "Not a heading") not in paths
    assert ("Real", "After") in paths


def test_long_section_overlaps_and_covers_the_text():
    sentence = "Backpropagation computes gradients by the chain rule. "
    text = "# Lesson\n\n" + sentence * 40
    chunks = chunk_text(
        text,
        doc_id="notes",
        source="notes.md",
        source_type="markdown",
        chunk_size=120,
        overlap=30,
    )
    assert len(chunks) > 2
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.heading_path == ("Lesson",)
    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(starts)
    assert chunks[0].char_start < chunks[1].char_start < chunks[1].char_end
    assert chunks[1].char_start < chunks[0].char_end


def test_transcript_windows_keep_timestamps():
    cues = [
        Cue("Welcome to the lecture.", 0.0, 2.0),
        Cue("Backpropagation computes gradients.", 2.0, 6.0),
        Cue("The chain rule is applied at every layer.", 6.0, 11.0),
        Cue("That is the whole method.", 40.0, 44.0),
    ]
    chunks = chunk_cues(cues, doc_id="talk", source="talk.vtt", chunk_size=80, overlap=20)
    assert len(chunks) >= 2
    assert chunks[0].start_time == 0.0
    assert chunks[0].source_type == "transcript"
    assert "Backpropagation computes gradients." in chunks[0].text
    assert chunks[-1].text == "That is the whole method."
    assert chunks[-1].start_time == 40.0
    assert chunks[-1].end_time == 44.0
    for chunk in chunks:
        assert chunk.char_end - chunk.char_start == len(chunk.text)


def test_chunk_size_must_exceed_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", doc_id="d", source="s", source_type="text", chunk_size=10, overlap=10)
