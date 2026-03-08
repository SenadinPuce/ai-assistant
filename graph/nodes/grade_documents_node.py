import logging

from graph.chains.retrieval_grader_chain import retrieval_grader_chain
from graph.consts import MIN_RELEVANCE_RATIO
from graph.state import GraphState

logger = logging.getLogger(__name__)


def grade_documents_node(state: GraphState) -> GraphState:
    """Filter retrieved documents to those relevant to the question.

    Uses a ratio-based threshold (``MIN_RELEVANCE_RATIO``) to decide
    whether web search is needed, instead of triggering on any single
    irrelevant document.

    - If no documents were retrieved, ``web_search`` is set to ``True``
      and grading is skipped entirely (no LLM calls wasted).
    - If the fraction of relevant documents falls below the threshold,
      ``web_search`` is set to ``True`` so downstream nodes supplement
      with web results.
    """
    logger.info("Grading document relevance to question.")
    question = state["question"]
    documents = state["documents"]

    # Edge case: no documents retrieved at all — skip grading, go to web search
    if not documents:
        logger.info("No documents retrieved — triggering web search.")
        return {
            "documents": [],
            "web_search": True,
            "relevance_ratio": 0.0,
        }

    inputs = [{"question": question, "document": doc.page_content} for doc in documents]
    scores = retrieval_grader_chain.batch(inputs)

    filtered_docs = []
    for doc, score in zip(documents, scores):
        if score.binary_score == "yes":
            logger.info("Document graded as relevant.")
            filtered_docs.append(doc)
        else:
            logger.info("Document graded as not relevant.")

    relevance_ratio = len(filtered_docs) / len(documents)
    web_search = relevance_ratio < MIN_RELEVANCE_RATIO

    logger.info(
        "Relevance ratio: %.2f (%d/%d). Web search: %s.",
        relevance_ratio,
        len(filtered_docs),
        len(documents),
        web_search,
    )

    return {
        "documents": filtered_docs,
        "web_search": web_search,
        "relevance_ratio": relevance_ratio,
    }
