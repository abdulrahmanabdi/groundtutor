import json

import pytest

from groundtutor.embeddings import HashEmbedder
from groundtutor.grounding import NOT_FOUND, UNGROUNDED
from groundtutor.tutor import Tutor
from tests.fakes import ScriptedLLM
from tests.support import write_text_pdf

PASSAGE = "Backpropagation computes gradients of the loss by applying the chain rule."


def _answer(quote: str = PASSAGE) -> str:
    return json.dumps(
        {
            "answer": f"{quote} [1].",
            "claims": [
                {
                    "text": quote,
                    "citation_ids": ["1"],
                    "quotes": [{"citation_id": "1", "text": quote}],
                }
            ],
        }
    )


def _verdict(supported: bool = True, score: float = 0.9) -> str:
    return json.dumps(
        {"verdicts": [{"index": 0, "supported": supported, "score": score, "reason": "checked"}]}
    )


def _tutor(llm: ScriptedLLM, **kwargs) -> Tutor:
    options = {"verifier": "llm"}
    options.update(kwargs)
    return Tutor(llm=llm, embedder=HashEmbedder(), **options)


def test_ask_end_to_end_across_pdf_and_transcript(tmp_path):
    pdf = tmp_path / "lecture.pdf"
    write_text_pdf(pdf, [PASSAGE])
    vtt = tmp_path / "talk.vtt"
    vtt.write_text(
        "WEBVTT\n\n00:00:04.000 --> 00:00:09.000\nThe learning rate scales each gradient step.\n",
        encoding="utf-8",
    )
    llm = ScriptedLLM([_answer(), _verdict()])
    tutor = _tutor(llm, mode="strict")
    tutor.add(pdf)
    tutor.add_transcript(vtt)
    tutor.add_text("# Notes\n\nA bias is a learned offset.", source="notes.md")
    result = tutor.ask("What is backpropagation?")
    assert result.fully_grounded
    assert result.citations[0].source_type == "pdf"
    assert result.citations[0].page == 1
    assert "lecture.pdf" in result.citations[0].source
    assert len(tutor.documents) == 3


def test_refusal_and_modes(tmp_path):
    tutor = _tutor(ScriptedLLM([]), min_dense_score=1.1)
    tutor.add_text("Chop onions and simmer tomatoes with olive oil.", source="recipe.txt")
    missed = tutor.ask("What is backpropagation?", mode="strict")
    assert missed.answer == NOT_FOUND
    assert missed.suggestions
    assert tutor.llm.calls == []

    shared = ScriptedLLM([_answer(), _verdict(False, 0.1)])
    strict = _tutor(shared, mode="flag")
    strict.add_text(PASSAGE, source="lecture.txt")
    refused = strict.ask("What does backpropagation compute?", mode="strict")
    assert refused.answer == UNGROUNDED
    assert refused.ungrounded

    flagged = _tutor(ScriptedLLM([_answer(), _verdict(False, 0.1)]))
    flagged.add_text(PASSAGE, source="lecture.txt")
    kept = flagged.ask("What does backpropagation compute?", mode="flag")
    assert kept.refused is False
    assert kept.ungrounded

    permissive = _tutor(ScriptedLLM([_answer()]))
    permissive.add_text(PASSAGE, source="lecture.txt")
    loose = permissive.ask("What does backpropagation compute?", mode="permissive")
    assert loose.fully_grounded
    assert len(permissive.llm.calls) == 1


def test_save_and_load_round_trip(tmp_path):
    llm = ScriptedLLM([_answer(), _verdict()])
    tutor = _tutor(llm)
    tutor.add_text(PASSAGE, source="lecture.txt", doc_id="lecture")
    directory = tmp_path / "index"
    tutor.save(directory)
    restored = Tutor.load(
        directory, llm=ScriptedLLM([_answer(), _verdict()]), embedder=HashEmbedder()
    )
    assert restored.documents == ("lecture",)
    result = restored.ask("What is backpropagation?", mode="strict")
    assert result.fully_grounded
    assert result.citations[0].doc_id == "lecture"


def test_duplicate_document_and_empty_question():
    tutor = _tutor(ScriptedLLM([]), verifier="lexical")
    tutor.add_text(PASSAGE, source="lecture.txt", doc_id="lecture")
    with pytest.raises(ValueError):
        tutor.add_text("other", doc_id="lecture")
    with pytest.raises(ValueError):
        tutor.ask("   ")


def test_quiz_and_flashcards_drop_ungrounded_items():
    passage = PASSAGE
    quiz_payload = json.dumps(
        {
            "questions": [
                {
                    "question": "What does backpropagation compute?",
                    "choices": [
                        "Gradients of the loss by the chain rule",
                        "A cooking time",
                        "A font name",
                        "A page size",
                    ],
                    "answer": "Gradients of the loss by the chain rule",
                    "explanation": "It computes gradients of the loss by applying the chain rule.",
                    "citation_ids": [1],
                    "quote": passage,
                },
                {
                    "question": "Where is the energy stored?",
                    "choices": ["In glucose", "In the loss", "In the page", "In the slide"],
                    "answer": "In glucose",
                    "explanation": "Photosynthesis stores energy in glucose.",
                    "citation_ids": [1],
                    "quote": passage,
                },
            ]
        }
    )
    cards_payload = json.dumps(
        {
            "cards": [
                {
                    "front": "What does backpropagation compute?",
                    "back": "Gradients of the loss by applying the chain rule.",
                    "citation_ids": [1],
                    "quote": passage,
                },
                {
                    "front": "What stores sunlight?",
                    "back": "Photosynthesis stores energy in glucose.",
                    "citation_ids": [1],
                    "quote": "not in the source",
                },
            ]
        }
    )
    llm = ScriptedLLM([quiz_payload, cards_payload])
    tutor = Tutor(llm=llm, embedder=HashEmbedder(), verifier="lexical")
    tutor.add_text(passage, source="lecture.txt")
    quiz = tutor.quiz(5)
    assert len(quiz) == 1
    assert quiz[0].citations[0].source == "lecture.txt"
    assert quiz[0].quote == passage
    assert quiz[0].grounding_score > 0

    cards = tutor.flashcards(10)
    assert len(cards) == 1
    assert cards[0].back.startswith("Gradients")
    assert cards[0].citations[0].locator


def test_topic_with_no_support_skips_the_model():
    llm = ScriptedLLM([])
    tutor = Tutor(
        llm=llm,
        embedder=HashEmbedder(),
        verifier="lexical",
        min_dense_score=1.1,
    )
    tutor.add_text(PASSAGE, source="lecture.txt")
    assert len(tutor.quiz(3, topic="photosynthesis and glucose")) == 0
    assert llm.calls == []
