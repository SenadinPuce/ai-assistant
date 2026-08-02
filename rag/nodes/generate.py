import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from rag.chains.generation import generation_chain
from rag.language import detect_language
from rag.state import GraphState

logger = logging.getLogger(__name__)

_SNIPPET_MAX_LENGTH = 280


def _document_key(doc: Document) -> tuple | None:
    """Return the dedup/citation key for a document (source+page, or url)."""
    meta = doc.metadata or {}
    source = meta.get("source", "")

    if "page" in meta:
        return (source, meta["page"])
    if source:
        return (source,)
    return None


def _make_snippet(text: str, max_length: int = _SNIPPET_MAX_LENGTH) -> str:
    """Collapse whitespace and trim a chunk's text to a short citation excerpt."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].rstrip() + "…"


def _format_context(documents: list[Document]) -> str:
    """Join document page content into blocks numbered to match citation markers."""
    if not documents:
        return ""

    numbers: dict[tuple, int] = {}
    blocks: list[str] = []

    for doc in documents:
        key = _document_key(doc)
        if key is None:
            blocks.append(doc.page_content)
            continue

        if key not in numbers:
            numbers[key] = len(numbers) + 1
        blocks.append(f"[{numbers[key]}] {doc.page_content}")

    return "\n\n".join(blocks)


def _source_label(path: str) -> str:
    """Extract a file name from a full path."""
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if path else "unknown"


def _extract_sources(documents: list[Document]) -> list[dict[str, Any]]:
    """Build a deduplicated list of source references, numbered to match citation markers."""
    seen: set[tuple] = set()
    sources: list[dict[str, Any]] = []

    for doc in documents:
        key = _document_key(doc)
        if key is None or key in seen:
            continue
        seen.add(key)

        meta = doc.metadata or {}
        snippet = _make_snippet(doc.page_content)

        # Ingested files always carry "file_type"; only true web-search results
        # (rag/nodes/web_search.py) lack it and should render as a clickable link.
        if "file_type" in meta:
            source = {
                "source": _source_label(meta.get("original_filename") or meta.get("source", "")),
                "snippet": snippet,
            }
            if "page" in meta:
                source["page"] = meta["page"] + 1
            sources.append(source)
        else:
            sources.append({"url": meta.get("source", ""), "snippet": snippet})

    return sources


def generate_answer_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """Generate an answer to the question based on retrieved documents."""
    question = state.get("original_question") or state["question"]
    logger.info("Generating answer for question: %s", question)

    context = _format_context(state["documents"])
    language = detect_language(question)

    chat_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in state.get("chat_history") or []
    ]

    # Stream (rather than invoke) so token deltas surface through astream_events.
    generation = "".join(
        generation_chain.stream(
            {
                "question": question,
                "context": context,
                "language": language,
                "chat_history": chat_history,
            },
            config=config,
        )
    )
    sources = _extract_sources(state["documents"])

    return {"generation": generation, "sources": sources}

