import os
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from graph.chains.retrieval_grader_chain import GradeDocument, retrieval_grader_chain
from graph.nodes.grade_documents_node import grade_documents
from graph.state import GraphState


# Helper functions for tests


def _make_state(question: str, documents: list) -> GraphState:
    return {"question": question, "documents": documents, "generation": "", "web_search": False}


def _grade(score: str) -> GradeDocument:
    return GradeDocument(binary_score=score)



# Unit tests for grade_documents node


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_filters_irrelevant(mock_grader: MagicMock) -> None:
    """Irrelevant documents are removed and web_search is set to True."""
    mock_grader.invoke.side_effect = [_grade("yes"), _grade("no"), _grade("yes")]
    docs = [
        Document(page_content="relevant content A"),
        Document(page_content="completely unrelated content "),
        Document(page_content="relevant content B"),
    ]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["documents"][0].page_content == "relevant content A"
    assert result["documents"][1].page_content == "relevant content B"
    assert result["web_search"] is True


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_all_relevant(mock_grader: MagicMock) -> None:
    """When all documents are relevant, web_search stays False."""
    mock_grader.invoke.return_value = _grade("yes")
    docs = [Document(page_content="doc A"), Document(page_content="doc B")]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["web_search"] is False


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_all_irrelevant(mock_grader: MagicMock) -> None:
    """When all documents are irrelevant, result is empty and web_search is True."""
    mock_grader.invoke.return_value = _grade("no")
    docs = [Document(page_content="unrelated A"), Document(page_content="unrelated B")]

    result = grade_documents(_make_state("poslovni sistemi", docs))

    assert result["documents"] == []
    assert result["web_search"] is True


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_empty_list(mock_grader: MagicMock) -> None:
    """Empty document list returns empty result and web_search stays False."""
    result = grade_documents(_make_state("poslovni sistemi", []))

    mock_grader.invoke.assert_not_called()
    assert result["documents"] == []
    assert result["web_search"] is False


@patch("graph.nodes.grade_documents_node.retrieval_grader_chain")
def test_grade_documents_preserves_question(mock_grader: MagicMock) -> None:
    """The question is passed through unchanged in the returned state."""
    mock_grader.invoke.return_value = _grade("yes")
    docs = [Document(page_content="content")]

    result = grade_documents(_make_state("Sta su poslovni sistemi?", docs))

    assert result["question"] == "Sta su poslovni sistemi?"


# Integration tests for retrieval_grader chain

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
