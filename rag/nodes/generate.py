import logging

from langchain_core.documents import Document

from rag.chains.generation import generation_chain
from rag.language import detect_language
from rag.state import GraphState

logger = logging.getLogger(__name__)


def _format_context(documents: list[Document]) -> str:
    """Join document page content into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def generate_answer_node(state: GraphState) -> GraphState:
    """Generate an answer to the question based on retrieved documents."""
    question = state.get("original_question") or state["question"]
    logger.info("Generating answer for question: %s", question)

    context = _format_context(state["documents"])
    language = detect_language(question)

    generation = generation_chain.invoke({"question": question, "context": context, "language": language})

    return {"generation": generation}
