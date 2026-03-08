from graph.state import GraphState
from ingestion import retrieve_with_score_filter


def retrieve_documents_node(state: GraphState) -> GraphState:
    """Retrieve documents relevant to the question in the current graph state."""
    documents = retrieve_with_score_filter(state["question"])
    return {"documents": documents}
