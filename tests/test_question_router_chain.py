import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag.chains.question_router import question_router_chain

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@_skip_integration
def test_routes_greeting_directly() -> None:
    """A simple greeting doesn't need retrieval."""
    result = question_router_chain.invoke({"question": "Zdravo, kako si?"})

    assert result.datasource == "direct"


@_skip_integration
def test_routes_domain_question_to_vectorstore() -> None:
    """A domain-specific question requires retrieval from the knowledge base."""
    result = question_router_chain.invoke(
        {"question": "Objasni razliku između poslovnog i informacionog sistema."}
    )

    assert result.datasource == "vectorstore"
