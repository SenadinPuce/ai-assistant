import logging
from functools import lru_cache

from langchain_core.documents import Document

from rag.constants import RERANKER_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, documents: list[Document]) -> list[tuple[Document, float]]:
    """Score documents by relevance using a cross-encoder.

    Returns (document, score) pairs sorted by descending score.
    """
    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = _get_reranker().predict(pairs)

    scored_docs = sorted(
        zip(documents, scores), key=lambda x: x[1], reverse=True
    )

    for i, (doc, score) in enumerate(scored_docs, 1):
        logger.debug("[%d] reranker score=%.4f | %s", i, score, doc.page_content[:80])

    return scored_docs
