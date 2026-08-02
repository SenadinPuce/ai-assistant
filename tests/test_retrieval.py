import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from retrieval import search_with_scores
from retrieval.vectorstore import get_vectorstore

_skip_integration = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def _index_is_empty() -> bool:
    stats = get_vectorstore().index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0


@_skip_integration
def test_search_with_scores_returns_scored_results() -> None:
    """Integration: search_with_scores returns documents with cosine similarity scores."""
    if _index_is_empty():
        pytest.skip("Pinecone index has no ingested documents to search against.")

    results = search_with_scores(
        "Šta su poslovni sistemi i koje su njihove ključne komponente?", k=4
    )

    assert len(results) == 4
    for _doc, score in results:
        assert 0.0 <= score <= 1.0
