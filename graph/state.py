from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of the graph, including nodes and edges.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether web search is needed
        documents: list of retrieved documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]