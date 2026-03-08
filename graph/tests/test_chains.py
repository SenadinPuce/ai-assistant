import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from graph.chains.retrieval_grader_chain import GradeDocument, retrieval_grader_chain
from graph.nodes.generation_node import _format_context, generate_answer
from graph.nodes.grade_documents_node import grade_documents
from graph.state import GraphState

# ---------------------------------------------------------------------------
# Helper functions for tests
# ---------------------------------------------------------------------------


def _make_state(question: str, documents: list) -> GraphState:
    return {
        "question": question,
        "documents": documents,
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
    }


def _grade(score: str) -> GradeDocument:
    return GradeDocument(binary_score=score)


# ---------------------------------------------------------------------------
# Unit tests for grade_documents node
# ---------------------------------------------------------------------------


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_filters_irrelevant_above_threshold(mock_grader: MagicMock) -> None:
    """2/3 relevant (0.67) is above the default 0.5 threshold — no web search."""
    mock_grader.invoke.side_effect = [_grade("yes"), _grade("no"), _grade("yes")]
    docs = [
        Document(page_content="relevant content A"),
        Document(page_content="completely unrelated content"),
        Document(page_content="relevant content B"),
    ]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["documents"][0].page_content == "relevant content A"
    assert result["documents"][1].page_content == "relevant content B"
    assert result["web_search"] is False
    assert result["relevance_ratio"] == pytest.approx(2 / 3)


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_all_relevant(mock_grader: MagicMock) -> None:
    """When all documents are relevant, web_search stays False and ratio is 1.0."""
    mock_grader.invoke.return_value = _grade("yes")
    docs = [Document(page_content="doc A"), Document(page_content="doc B")]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["web_search"] is False
    assert result["relevance_ratio"] == 1.0


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_all_irrelevant(mock_grader: MagicMock) -> None:
    """When all documents are irrelevant, result is empty and web_search is True."""
    mock_grader.invoke.return_value = _grade("no")
    docs = [Document(page_content="unrelated A"), Document(page_content="unrelated B")]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_empty_list(mock_grader: MagicMock) -> None:
    """Empty document list triggers web search directly — no LLM calls."""
    result = grade_documents(_make_state("poslovni sistemi", []))

    mock_grader.invoke.assert_not_called()
    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_below_threshold(mock_grader: MagicMock) -> None:
    """1/4 relevant (0.25) is below the 0.5 threshold — triggers web search."""
    mock_grader.invoke.side_effect = [
        _grade("yes"),
        _grade("no"),
        _grade("no"),
        _grade("no"),
    ]
    docs = [
        Document(page_content="relevant"),
        Document(page_content="unrelated A"),
        Document(page_content="unrelated B"),
        Document(page_content="unrelated C"),
    ]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 1
    assert result["web_search"] is True
    assert result["relevance_ratio"] == pytest.approx(0.25)


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_at_exact_threshold(mock_grader: MagicMock) -> None:
    """Exactly at the threshold (0.5) — no web search (>= comparison)."""
    mock_grader.invoke.side_effect = [_grade("yes"), _grade("no")]
    docs = [
        Document(page_content="relevant"),
        Document(page_content="unrelated"),
    ]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 1
    assert result["web_search"] is False
    assert result["relevance_ratio"] == 0.5

# ---------------------------------------------------------------------------
# Integration tests for retrieval_grader chain
# ---------------------------------------------------------------------------

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@_skip_integration
def test_retrieval_grader_relevant_document() -> None:
    """Integration: grader returns 'yes' for a clearly relevant document."""
    result: GradeDocument = retrieval_grader_chain.invoke(
        {
            "question": "Sta su poslovni sistemi?",
            "document": (
                "Poslovni sistem predstavlja skup proizvodnih, ekonomskih i društvenih podsistema i elemenata koji povezuju "
                "okolinu (tržište) sa proizvodnim sistemima."
            ),
        }
    )
    assert result.binary_score == "yes"


@_skip_integration
def test_retrieval_grader_irrelevant_document() -> None:
    """Integration: grader returns 'no' for a clearly unrelated document."""
    result: GradeDocument = retrieval_grader_chain.invoke(
        {
            "question": "Sta su poslovni sistemi?",
            "document": "Stari Most u Mostaru je proglašen najljepšim mostom na svijetu 2026. godine.",
        }
    )
    assert result.binary_score == "no"


# ---------------------------------------------------------------------------
# Unit tests — generation node
# ---------------------------------------------------------------------------


def test_format_context_joins_documents() -> None:
    """_format_context joins multiple Document page_contents with double newline."""
    docs = [
        Document(page_content="First chunk."),
        Document(page_content="Second chunk."),
    ]
    result = _format_context(docs)
    assert result == "First chunk.\n\nSecond chunk."


def test_format_context_empty_list() -> None:
    """_format_context returns empty string for no documents."""
    assert _format_context([]) == ""


@patch("graph.nodes.generation_node.generation_chain")
def test_generate_answer_calls_chain_with_formatted_context(mock_chain: MagicMock) -> None:
    """generate_answer formats documents and passes them as a string to the chain."""
    mock_chain.invoke.return_value = "Poslovni sistemi su..."
    docs = [
        Document(page_content="Dio A."),
        Document(page_content="Dio B."),
    ]
    state: GraphState = {
        "question": "Sta su poslovni sistemi?",
        "documents": docs,
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
    }

    result = generate_answer(state)

    mock_chain.invoke.assert_called_once_with(
        {"question": "Sta su poslovni sistemi?", "context": "Dio A.\n\nDio B."}
    )
    assert result["generation"] == "Poslovni sistemi su..."


@patch("graph.nodes.generation_node.generation_chain")
def test_generate_answer_returns_generation_in_state(mock_chain: MagicMock) -> None:
    """generate_answer stores the chain output under 'generation' key."""
    mock_chain.invoke.return_value = "Odgovor."
    state: GraphState = {
        "question": "Pitanje?",
        "documents": [Document(page_content="Kontekst.")],
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
    }

    result = generate_answer(state)

    assert result["generation"] == "Odgovor."


# ---------------------------------------------------------------------------
# Integration test — generation chain
# ---------------------------------------------------------------------------


@_skip_integration
def test_generation_chain_produces_answer() -> None:
    """Integration: generation chain returns a non-empty string answer."""
    from graph.chains.generation_chain import generation_chain

    result: str = generation_chain.invoke(
        {
            "question": "Sta su poslovni sistemi?",
            "context": (
                "Poslovni sistem predstavlja skup proizvodnih, ekonomskih i "
                "društvenih podsistema koji povezuju okolinu sa proizvodnim sistemima."
            ),
        }
    )

    assert isinstance(result, str)
    assert len(result) > 0

