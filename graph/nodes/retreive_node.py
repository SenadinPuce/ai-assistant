from langchain_core.documents import Document

from graph.state import GraphState
from ingestion import get_retriever


_retriever = get_retriever()


def retrieve_documents(state: GraphState) -> GraphState:
    """Retrieve documents relevant to the question in the current graph state."""
    documents: list[Document] = _retriever.invoke(state["question"])
    return {"documents": documents}