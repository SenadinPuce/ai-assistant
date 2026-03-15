import logging

from langchain_core.messages import AIMessage, HumanMessage

from rag.chains.generation import generation_chain
from rag.language import detect_language
from rag.state import GraphState

logger = logging.getLogger(__name__)


def direct_answer_node(state: GraphState) -> GraphState:
    """Answer the question directly without retrieval — for greetings, simple queries, etc."""
    logger.info("Routing to direct answer (no retrieval needed).")

    question = state["question"]
    language = detect_language(question)

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in state.get("chat_history") or []
    ]

    generation = generation_chain.invoke({
        "question": question,
        "context": "",
        "language": language,
        "chat_history": chat_history,
    })

    return {"generation": generation, "documents": []}
