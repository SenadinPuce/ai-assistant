import logging

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.consts import WEB_SEARCH_MAX_RESULTS
from graph.state import GraphState

logger = logging.getLogger(__name__)

_web_search_tool = TavilySearch(max_results=WEB_SEARCH_MAX_RESULTS)


def web_search_node(state: GraphState) -> GraphState:
    """Perform a web search and append results as Documents to the state.

    Each search result becomes its own Document so downstream grading
    nodes can evaluate and filter them individually.
    """
    logger.info("Performing web search.")

    question = state["question"]
    existing_docs = state.get("documents") or []

    search_results = _web_search_tool.invoke({"query": question})
    new_docs = [
        Document(
            page_content=result["content"], metadata={"source": result.get("url", "")}
        )
        for result in search_results["results"]
    ]

    return {
        "documents": existing_docs + new_docs,
        "question": question,
        "web_search": False,
    }
