from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from rag.nodes.web_search import web_search_node
from rag.state import GraphState


def _make_state(documents: list | None = None) -> GraphState:
    return {
        "question": "Sta je novo u AI industriji?",
        "documents": documents or [],
        "generation": "",
        "web_search": True,
        "relevance_ratio": 0.0,
    }


@patch("rag.nodes.web_search._web_search_tool")
def test_web_search_appends_results_as_documents(mock_tool: MagicMock) -> None:
    """Each Tavily result becomes its own Document, preserving prior documents."""
    mock_tool.invoke.return_value = {
        "results": [
            {"content": "Prvi rezultat.", "url": "https://example.com/1"},
            {"content": "Drugi rezultat.", "url": "https://example.com/2"},
        ]
    }
    existing = [Document(page_content="postojeći dokument")]

    result = web_search_node(_make_state(existing))

    assert len(result["documents"]) == 3
    assert result["documents"][0] is existing[0]
    assert result["documents"][1].page_content == "Prvi rezultat."
    assert result["documents"][1].metadata["source"] == "https://example.com/1"
    assert result["web_search"] is False


@patch("rag.nodes.web_search._web_search_tool")
def test_web_search_handles_tavily_error(mock_tool: MagicMock) -> None:
    """A Tavily API failure keeps existing documents and doesn't raise."""
    mock_tool.invoke.return_value = {"error": "rate limited"}
    existing = [Document(page_content="postojeći dokument")]

    result = web_search_node(_make_state(existing))

    assert result["documents"] == existing
    assert result["web_search"] is False
