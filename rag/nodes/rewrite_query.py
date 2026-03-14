import logging

from rag.chains.query_rewriter import query_rewriter_chain
from rag.state import GraphState

logger = logging.getLogger(__name__)


def rewrite_query_node(state: GraphState) -> GraphState:
    """Rewrite the user question for better retrieval while preserving the original."""
    original = state["question"]

    rewritten = query_rewriter_chain.invoke({"question": original})
    logger.info("Query rewritten: '%s' -> '%s'", original, rewritten)

    return {"question": rewritten, "original_question": original}
