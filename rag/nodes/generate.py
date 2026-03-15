import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag.chains.generation import generation_chain
from rag.language import detect_language
from rag.state import GraphState

logger = logging.getLogger(__name__)


def _format_context(documents: list[Document]) -> str:
    """Join document page content into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def _source_label(path: str) -> str:
    """Extract a file name from a full path."""
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if path else "unknown"


def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """Build a deduplicated list of source references from document metadata."""
    seen: set[tuple] = set()
    sources: list[dict[str, Any]] = []

    for doc in documents:
        meta = doc.metadata or {}
        source = meta.get("source", "")

        if "page" in meta:
            key = (source, meta["page"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source": _source_label(source),
                "page": meta["page"] + 1,
            })
        elif source:
            key = (source,)
            if key in seen:
                continue
            seen.add(key)
            sources.append({"url": source})

    return sources


def generate_answer_node(state: GraphState) -> GraphState:
    """Generate an answer to the question based on retrieved documents."""
    question = state.get("original_question") or state["question"]
    logger.info("Generating answer for question: %s", question)

    context = _format_context(state["documents"])
    language = detect_language(question)

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in state.get("chat_history") or []
    ]

    generation = generation_chain.invoke({
        "question": question,
        "context": context,
        "language": language,
        "chat_history": chat_history,
    })
    sources = _extract_sources(state["documents"])

    return {"generation": generation, "sources": sources}
