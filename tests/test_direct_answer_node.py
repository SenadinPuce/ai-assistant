from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

load_dotenv()

from rag.nodes.direct_answer import direct_answer_node
from rag.state import GraphState


def _make_state(chat_history: list | None = None) -> GraphState:
    return {
        "question": "Zdravo!",
        "documents": [],
        "generation": "",
        "web_search": False,
        "relevance_ratio": 0.0,
        "chat_history": chat_history or [],
    }


@patch("rag.nodes.direct_answer.detect_language", return_value="Bosnian")
@patch("rag.nodes.direct_answer.generation_chain")
def test_direct_answer_streams_and_forwards_config(
    mock_chain: MagicMock, _mock_language: MagicMock
) -> None:
    """The node concatenates streamed chunks and forwards the LangGraph config."""
    mock_chain.stream.return_value = iter(["Zdravo", ", ", "svijete!"])
    config = {"run_id": "abc"}

    result = direct_answer_node(_make_state(), config)

    assert result["generation"] == "Zdravo, svijete!"
    assert result["documents"] == []
    _, kwargs = mock_chain.stream.call_args
    assert kwargs["config"] is config


@patch("rag.nodes.direct_answer.detect_language", return_value="Bosnian")
@patch("rag.nodes.direct_answer.generation_chain")
def test_direct_answer_strips_citations_from_prior_turns(
    mock_chain: MagicMock, _mock_language: MagicMock
) -> None:
    """Citation markers from earlier assistant turns aren't echoed back into the prompt."""
    mock_chain.stream.return_value = iter(["ok"])
    chat_history = [
        {"role": "user", "content": "Pitanje"},
        {"role": "assistant", "content": "Odgovor [1] i [2]."},
    ]

    direct_answer_node(_make_state(chat_history), {})

    args, _ = mock_chain.stream.call_args
    forwarded_history = args[0]["chat_history"]
    assert forwarded_history[1].content == "Odgovor i."
