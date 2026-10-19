import json

import pytest

from groundtutor.embeddings import HashEmbedder
from groundtutor.errors import GroundingError
from groundtutor.grounding import (
    NOT_FOUND,
    UNGROUNDED,
    LexicalVerifier,
    LLMJudgeVerifier,
    answer_question,
    extract_json,
    find_span,
)
from groundtutor.retrieve import HybridRetriever
from groundtutor.stores import MemoryStore
from groundtutor.types import Chunk
from tests.fakes import ScriptedLLM

PASSAGE = "Backpropagation computes gradients of the loss by applying the chain rule."


def _chunk(text: str = PASSAGE, *, page: int | None = 2) -> Chunk:
    return Chunk(
        id="doc#0000",
        doc_id="doc",
        text=text,
        source="lecture.pdf",
        source_type="pdf",
        char_start=0,
        char_end=len(text),
        page=page,
        heading_path=("Backprop",),
    )


def _retrieval(*chunks: Chunk, query: str = "What is backpropagation?"):
    retriever = HybridRetriever(HashEmbedder(), MemoryStore())
    retriever.add(chunks)
    return retriever.retrieve(query, k=4)


def _answer(quote: str = PASSAGE) -> str:
    return json.dumps(
        {
            "answer": "Backpropagation computes gradients of the loss by applying the chain rule [1].",
            "claims": [
                {
                    "text": "Backpropagation computes gradients of the loss by applying the chain rule.",
                    "citation_ids": [1],
                    "quotes": [{"citation_id": 1, "text": quote}],
                }
            ],
        }
    )


def _verdict(supported: bool, score: float) -> str:
    return json.dumps(
        {"verdicts": [{"index": 0, "supported": supported, "score": score, "reason": "checked"}]}
    )


def test_find_span_collapses_whitespace_and_rejects_paraphrase():
    text = "Backpropagation computes\ngradients of the loss."
    start, end = find_span(text, "computes gradients of the loss")
    assert "computes" in text[start:end]
    assert "gradients" in text[start:end]
    assert find_span(text, "gradients of the error") is None


def test_extract_json_accepts_fences_and_rejects_prose():
    assert extract_json('```json\n{"answer": "x"}\n```') == {"answer": "x"}
    with pytest.raises(GroundingError):
        extract_json("I do not know.")


def test_strict_answer_keeps_a_supported_claim_and_its_citation():
    llm = ScriptedLLM([_answer(), _verdict(True, 0.95)])
    result = answer_question(
        "What is backpropagation?",
        _retrieval(_chunk()),
        llm=llm,
        verifier=LexicalVerifier(),
        mode="strict",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    # LexicalVerifier is passed but strict mode calls it. The scripted judge is not used
    # because the verifier is lexical. One model call: the answer.
    assert len(llm.calls) == 1
    assert result.refused is False
    assert result.fully_grounded
    assert result.claims[0].supported
    assert result.claims[0].quotes[0].text in PASSAGE
    assert result.citations[0].locator.startswith("p. 2")
    assert result.citations[0].page == 2
    assert "Backprop" in result.citations[0].locator


def test_llm_judge_scores_are_stored_and_strict_refuses_a_low_score():
    retrieval = _retrieval(_chunk())
    kept = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=ScriptedLLM([_answer(), _verdict(True, 0.91)]),
        verifier=None,  # replaced below
        mode="flag",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    # Build the judge through the same scripted client by calling with a real verifier.
    from groundtutor.grounding import LLMJudgeVerifier

    llm = ScriptedLLM([_answer(), _verdict(True, 0.91)])
    judged = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=llm,
        verifier=LLMJudgeVerifier(llm),
        mode="flag",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert len(llm.calls) == 2
    assert judged.fully_grounded
    assert judged.claims[0].grounding_score == pytest.approx(0.91)
    assert "entailed" in llm.calls[1][0]["content"].lower()

    refused = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=ScriptedLLM([_answer(), _verdict(False, 0.2)]),
        verifier=LLMJudgeVerifier(ScriptedLLM([_verdict(False, 0.2)])),
        mode="strict",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    # The verifier has its own client, so the answer client is separate.
    # Re-run with one shared client so the queue order is answer then verdict.
    shared = ScriptedLLM([_answer(), _verdict(False, 0.2)])
    refused = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=shared,
        verifier=LLMJudgeVerifier(shared),
        mode="strict",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert refused.refused
    assert refused.answer == UNGROUNDED
    assert refused.ungrounded[0].supported is False
    assert kept.answer  # the unused first call still produced the passage shape
    del kept


def test_flag_keeps_the_answer_and_permissive_skips_the_judge():
    retrieval = _retrieval(_chunk())
    flagged = ScriptedLLM([_answer(), _verdict(False, 0.1)])
    flagged_result = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=flagged,
        verifier=LLMJudgeVerifier(flagged),
        mode="flag",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert flagged_result.refused is False
    assert flagged_result.answer.startswith("Backpropagation")
    assert flagged_result.ungrounded

    permissive = ScriptedLLM([_answer("this quote is not in the source")])
    permissive_result = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=permissive,
        verifier=LLMJudgeVerifier(permissive),
        mode="permissive",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert len(permissive.calls) == 1
    assert permissive_result.ungrounded
    assert permissive_result.ungrounded[0].grounding_score == 0.0

    trusted = ScriptedLLM([_answer()])
    trusted_result = answer_question(
        "What is backpropagation?",
        retrieval,
        llm=trusted,
        verifier=LLMJudgeVerifier(trusted),
        mode="permissive",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert len(trusted.calls) == 1
    assert trusted_result.fully_grounded
    assert trusted_result.claims[0].grounding_score == 1.0


def test_low_confidence_refuses_without_calling_the_model():
    llm = ScriptedLLM([])
    result = answer_question(
        "What is photosynthesis?",
        _retrieval(_chunk(), query="What is photosynthesis?"),
        llm=llm,
        verifier=LexicalVerifier(),
        mode="strict",
        min_dense_score=1.1,
        min_term_coverage=0.45,
    )
    assert llm.calls == []
    assert result.refused
    assert result.answer == NOT_FOUND
    assert result.suggestions
    assert result.suggestions[0].page == 2
    assert "Backpropagation" in result.suggestions[0].snippet


def test_fabricated_quote_is_ungrounded_without_a_verdict():
    llm = ScriptedLLM([_answer("a sentence the source never wrote")])
    result = answer_question(
        "What is backpropagation?",
        _retrieval(_chunk()),
        llm=llm,
        verifier=LexicalVerifier(),
        mode="flag",
        min_dense_score=0.28,
        min_term_coverage=0.45,
    )
    assert len(llm.calls) == 1
    assert result.ungrounded
    assert "not found" in result.ungrounded[0].reason.lower()
