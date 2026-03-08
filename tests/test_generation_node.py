from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.nodes.generate import _format_context, generate_answer_node
from rag.state import GraphState


# ---------------------------------------------------------------------------
# Unit tests — _format_context
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


# ---------------------------------------------------------------------------
# Unit tests — generate_answer_node
# ---------------------------------------------------------------------------


@patch("rag.nodes.generate.generation_chain")
def test_generate_answer_calls_chain_with_formatted_context(
    mock_chain: MagicMock,
) -> None:
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

    result = generate_answer_node(state)

    mock_chain.invoke.assert_called_once_with(
        {"question": "Sta su poslovni sistemi?", "context": "Dio A.\n\nDio B."}
    )
    assert result["generation"] == "Poslovni sistemi su..."


@patch("rag.nodes.generate.generation_chain")
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

    result = generate_answer_node(state)

    assert result["generation"] == "Odgovor."
