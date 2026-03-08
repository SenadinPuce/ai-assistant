import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from retrieval import search_with_scores

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


@_skip_integration
def test_search_with_scores_returns_scored_results() -> None:
    """Integration: search_with_scores returns documents with cosine similarity scores."""
    results = search_with_scores(
        "Šta su poslovni sistemi i koje su njihove ključne komponente?", k=4
    )

    assert len(results) == 4
    for _doc, score in results:
        assert 0.0 <= score <= 1.0
