from groundtutor import __version__
from groundtutor.types import Chunk


def _chunk(**overrides) -> Chunk:
    payload = {
        "id": "a#0000",
        "doc_id": "a",
        "text": "hello",
        "source": "notes.md",
        "source_type": "markdown",
        "char_start": 0,
        "char_end": 5,
    }
    payload.update(overrides)
    return Chunk(**payload)


def test_package_version():
    assert __version__ == "0.1.0"


def test_locator_formats_page_slide_and_timestamp():
    assert _chunk(page=3, heading_path=("Neural nets", "Backprop")).locator() == (
        "p. 3 · Neural nets › Backprop"
    )
    assert _chunk(slide=2).locator() == "slide 2"
    assert _chunk(start_time=65, end_time=70).locator() == "01:05–01:10"
    assert _chunk().locator() == "source"
