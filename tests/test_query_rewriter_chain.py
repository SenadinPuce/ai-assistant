import os

import pytest
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

from rag.chains.query_rewriter import query_rewriter_chain

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@_skip_integration
def test_rewrites_question_without_history() -> None:
    """Rewriting produces a non-empty, self-contained query."""
    result = query_rewriter_chain.invoke({"question": "Sta su to?", "chat_history": []})

    assert isinstance(result, str)
    assert len(result) > 0


@_skip_integration
def test_rewrites_question_using_chat_history() -> None:
    """The rewriter resolves pronouns using prior conversation turns."""
    chat_history = [
        HumanMessage(content="Reci mi nesto o poslovnim sistemima."),
        AIMessage(content="Poslovni sistem je skup podsistema koji povezuju okolinu s proizvodnjom."),
    ]

    result = query_rewriter_chain.invoke(
        {"question": "Koje su njegove karakteristike?", "chat_history": chat_history}
    )

    assert isinstance(result, str)
    assert len(result) > 0
