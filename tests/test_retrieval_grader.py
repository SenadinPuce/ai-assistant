import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag.chains.retrieval_grader import retrieval_grader_chain
from rag.models import GradeDocument

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
    assert result.binary_score


@_skip_integration
def test_retrieval_grader_irrelevant_document() -> None:
    """Integration: grader returns 'no' for a clearly unrelated document."""
    result: GradeDocument = retrieval_grader_chain.invoke(
        {
            "question": "Sta su poslovni sistemi?",
            "document": "Stari Most u Mostaru je proglašen najljepšim mostom na svijetu 2026. godine.",
        }
    )
    assert not result.binary_score
