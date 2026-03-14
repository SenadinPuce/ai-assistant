import logging

from rag.constants import RERANKER_SCORE_THRESHOLD, RERANKER_TOP_N
from rag.reranker import rerank
from rag.state import GraphState

logger = logging.getLogger(__name__)


def rerank_documents_node(state: GraphState) -> GraphState:
    """Rerank retrieved documents by relevance using a cross-encoder.

    Over-retrieved candidates are scored, filtered by threshold, and
    trimmed to the top N.  If no documents survive, ``web_search`` is
    set to ``True`` so downstream nodes supplement with web results.
    """
    logger.info("Reranking documents for relevance to question.")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.info("No documents retrieved — triggering web search.")
        return {
            "documents": [],
            "web_search": True,
            "relevance_ratio": 0.0,
        }

    scored_docs = rerank(question, documents)

    filtered = [
        doc for doc, score in scored_docs if score >= RERANKER_SCORE_THRESHOLD
    ][:RERANKER_TOP_N]

    relevance_ratio = len(filtered) / len(documents)
    web_search = len(filtered) == 0

    logger.info(
        "Reranking complete: %d/%d above threshold (ratio %.2f). Web search: %s.",
        len(filtered),
        len(documents),
        relevance_ratio,
        web_search,
    )

    return {
        "documents": filtered,
        "web_search": web_search,
        "relevance_ratio": relevance_ratio,
    }
