from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

from rag.graph import _decide_to_generate, _route_question, build_graph
from rag.state import GraphState


def _make_state(**overrides) -> GraphState:
    base: GraphState = {
        "question": "Sta su poslovni sistemi?",
        "original_question": "Sta su poslovni sistemi?",
        "generation": "",
        "web_search": False,
        "documents": [],
        "relevance_ratio": 0.0,
        "sources": [],
        "chat_history": [],
    }
    base.update(overrides)
    return base


@patch("rag.graph.question_router_chain")
def test_route_question_to_vectorstore(mock_chain: MagicMock) -> None:
    """Router decision 'vectorstore' sends the question to query rewriting."""
    mock_chain.invoke.return_value = MagicMock(datasource="vectorstore")

    result = _route_question(_make_state())

    assert result == "rewrite_query"


@patch("rag.graph.question_router_chain")
def test_route_question_direct(mock_chain: MagicMock) -> None:
    """Router decision 'direct' skips retrieval entirely."""
    mock_chain.invoke.return_value = MagicMock(datasource="direct")

    result = _route_question(_make_state())

    assert result == "direct_answer"


def test_decide_to_generate_triggers_web_search() -> None:
    """web_search flag routes to the WEB_SEARCH node."""
    result = _decide_to_generate(_make_state(web_search=True, relevance_ratio=0.0))

    assert result == "web_search"


def test_decide_to_generate_skips_web_search() -> None:
    """Sufficient relevant documents route straight to answer generation."""
    result = _decide_to_generate(_make_state(web_search=False, relevance_ratio=1.0))

    assert result == "generate_answer"


def test_build_graph_compiles() -> None:
    """The workflow graph compiles into a runnable app without errors."""
    compiled = build_graph()

    assert compiled is not None
