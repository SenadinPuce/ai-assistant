from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.nodes.rerank_documents import rerank_documents_node
from rag.state import GraphState


def _make_state(question: str, documents: list) -> GraphState:
    return {
        "question": question,
        "documents": documents,
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
    }


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_keeps_top_relevant_documents(mock_rerank: MagicMock) -> None:
    """Documents above threshold are kept, sorted by score, limited to top_n."""
    docs = [
        Document(page_content="doc A"),
        Document(page_content="doc B"),
        Document(page_content="doc C"),
        Document(page_content="doc D"),
    ]
    mock_rerank.return_value = [
        (docs[2], 5.2),
        (docs[0], 3.1),
        (docs[3], 1.0),
        (docs[1], -2.5),
    ]

    result = rerank_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 3
    assert result["documents"][0].page_content == "doc C"
    assert result["documents"][1].page_content == "doc A"
    assert result["documents"][2].page_content == "doc D"
    assert result["web_search"] is False
    assert result["relevance_ratio"] == pytest.approx(3 / 4)


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_all_above_threshold(mock_rerank: MagicMock) -> None:
    """When all documents score above threshold, top_n limits the output."""
    docs = [Document(page_content=f"doc {i}") for i in range(5)]
    mock_rerank.return_value = [
        (docs[0], 8.0),
        (docs[1], 6.0),
        (docs[2], 4.0),
        (docs[3], 2.0),
        (docs[4], 1.0),
    ]

    result = rerank_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 3
    assert result["web_search"] is False


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_all_below_threshold(mock_rerank: MagicMock) -> None:
    """When all documents score below threshold, web search triggers."""
    docs = [Document(page_content="unrelated A"), Document(page_content="unrelated B")]
    mock_rerank.return_value = [
        (docs[0], -1.5),
        (docs[1], -3.0),
    ]

    result = rerank_documents_node(_make_state("poslovni sistemi", docs))

    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_empty_document_list(mock_rerank: MagicMock) -> None:
    """Empty document list triggers web search — reranker is not called."""
    result = rerank_documents_node(_make_state("poslovni sistemi", []))

    mock_rerank.assert_not_called()
    assert result["documents"] == []
    assert result["web_search"] is True
    assert result["relevance_ratio"] == 0.0


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_exactly_at_threshold(mock_rerank: MagicMock) -> None:
    """Score exactly at threshold (0.0) is kept (>= comparison)."""
    docs = [Document(page_content="borderline doc")]
    mock_rerank.return_value = [(docs[0], 0.0)]

    result = rerank_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 1
    assert result["web_search"] is False
    assert result["relevance_ratio"] == 1.0


@patch("rag.nodes.rerank_documents.rerank")
def test_rerank_fewer_than_top_n_above_threshold(mock_rerank: MagicMock) -> None:
    """When fewer docs than top_n pass threshold, all passing docs are kept."""
    docs = [Document(page_content=f"doc {i}") for i in range(6)]
    mock_rerank.return_value = [
        (docs[0], 4.0),
        (docs[1], 0.5),
        (docs[2], -0.1),
        (docs[3], -1.0),
        (docs[4], -2.0),
        (docs[5], -5.0),
    ]

    result = rerank_documents_node(_make_state("poslovni sistemi", docs))

    assert len(result["documents"]) == 2
    assert result["web_search"] is False
    assert result["relevance_ratio"] == pytest.approx(2 / 6)
