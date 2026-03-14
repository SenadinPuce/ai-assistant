import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag.chains.generation import generation_chain

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@_skip_integration
def test_generation_chain_produces_answer() -> None:
    """Integration: generation chain returns a non-empty string answer."""
    result: str = generation_chain.invoke(
        {
            "question": "Sta su poslovni sistemi?",
            "language": "Bosnian",
            "context": (
                "Poslovni sistem predstavlja skup proizvodnih, ekonomskih i "
                "društvenih podsistema koji povezuju okolinu sa proizvodnim sistemima."
            ),
        }
    )

    assert isinstance(result, str)
    assert len(result) > 0
