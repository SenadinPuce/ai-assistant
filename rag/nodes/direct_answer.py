import logging
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from rag.chains.generation import generation_chain
from rag.language import detect_language
from rag.state import GraphState

logger = logging.getLogger(__name__)

_CITATION_PATTERN = re.compile(r"\s?\[\d+\]")


def direct_answer_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """Answer the question directly without retrieval — for greetings, simple queries, etc."""
    logger.info("Routing to direct answer (no retrieval needed).")

    question = state["question"]
    language = detect_language(question)

    # Strip citation markers from prior assistant turns so the model has no
    # bracketed-number pattern to imitate when this answer has no real sources.
    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=_CITATION_PATTERN.sub("", m["content"]))
        for m in state.get("chat_history") or []
    ]

    # Stream (rather than invoke) so token deltas surface through astream_events.
    generation = "".join(
        generation_chain.stream(
            {
                "question": question,
                "context": "",
                "language": language,
                "chat_history": chat_history,
            },
            config=config,
        )
    )

    return {"generation": generation, "documents": []}
