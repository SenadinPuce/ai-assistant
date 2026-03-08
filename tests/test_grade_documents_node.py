from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.models import GradeDocument
from rag.nodes.grade_documents import grade_documents_node
from rag.state import GraphState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(question: str, documents: list) -> GraphState:
    return {
        "question": question,
        "documents": documents,
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
    }


def _grade(score: bool) -> GradeDocument:
    return GradeDocument(binary_score=score)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_filters_irrelevant_above_threshold(
    mock_grader: MagicMock,
) -> None:
    """2/3 relevant (0.67) is above the default 0.5 threshold — no web search."""
    mock_grader.batch.return_value = [_grade(True), _grade(False), _grade(True)]
    docs = [
        Document(page_content="relevant content A"),
        Document(page_content="completely unrelated content"),
        Document(page_content="relevant content B"),
    ]

    result = grade_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["documents"][0].page_content == "relevant content A"
    assert result["documents"][1].page_content == "relevant content B"
    assert result["web_search"] is False
    assert result["relevance_ratio"] == pytest.approx(2 / 3)


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_all_relevant(mock_grader: MagicMock) -> None:
    """When all documents are relevant, web_search stays False and ratio is 1.0."""
    mock_grader.batch.return_value = [_grade(True), _grade(True)]
    docs = [Document(page_content="doc A"), Document(page_content="doc B")]

    result = grade_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["web_search"] is False
    assert result["relevance_ratio"] == 1.0


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_all_irrelevant(mock_grader: MagicMock) -> None:
    """When all documents are irrelevant, result is empty and web_search is True."""
    mock_grader.batch.return_value = [_grade(False), _grade(False)]
    docs = [Document(page_content="unrelated A"), Document(page_content="unrelated B")]

    result = grade_documents_node(_make_state("poslovni sistemi", docs))

    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_empty_list(mock_grader: MagicMock) -> None:
    """Empty document list triggers web search directly — no LLM calls."""
    result = grade_documents_node(_make_state("poslovni sistemi", []))

    mock_grader.batch.assert_not_called()
    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_below_threshold(mock_grader: MagicMock) -> None:
    """1/4 relevant (0.25) is below the 0.5 threshold — triggers web search."""
    mock_grader.batch.return_value = [
        _grade(True),
        _grade(False),
        _grade(False),
        _grade(False),
    ]
    docs = [
        Document(page_content="relevant"),
        Document(page_content="unrelated A"),
        Document(page_content="unrelated B"),
        Document(page_content="unrelated C"),
    ]

    result = grade_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 1
    assert result["web_search"] is True
    assert result["relevance_ratio"] == pytest.approx(0.25)


@patch("rag.nodes.grade_documents.retrieval_grader_chain")
def test_grade_documents_at_exact_threshold(mock_grader: MagicMock) -> None:
    """Exactly at the threshold (0.5) — no web search (>= comparison)."""
    mock_grader.batch.return_value = [_grade(True), _grade(False)]
    docs = [
        Document(page_content="relevant"),
        Document(page_content="unrelated"),
    ]

    result = grade_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 1
    assert result["web_search"] is False
    assert result["relevance_ratio"] == 0.5
