import logging

from langchain_core.messages import AIMessage, HumanMessage

from rag.chains.query_rewriter import query_rewriter_chain
from rag.state import GraphState

logger = logging.getLogger(__name__)


def rewrite_query_node(state: GraphState) -> GraphState:
    """Rewrite the user question for better retrieval while preserving the original."""
    original = state["question"]

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in state.get("chat_history") or []
    ]

    rewritten = query_rewriter_chain.invoke({
        "question": original,
        "chat_history": chat_history,
    })
    logger.info("Query rewritten: '%s' -> '%s'", original, rewritten)

    return {"question": rewritten, "original_question": original}
