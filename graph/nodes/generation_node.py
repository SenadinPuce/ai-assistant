import logging

from langchain_core.documents import Document

from graph.chains.generation_chain import generation_chain
from graph.state import GraphState

logger = logging.getLogger(__name__)


def _format_context(documents: list[Document]) -> str:
    """Join document page content into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer to the question based on retrieved documents."""
    logger.info("Generating answer for question: %s", state["question"])

    question = state["question"]
    context = _format_context(state["documents"])

    generation = generation_chain.invoke({"question": question, "context": context})

    return {"generation": generation, "question": question}
