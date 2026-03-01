from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question:   The user's input question.
        generation: The final LLM-generated answer.
        web_search: Flag indicating whether a web search step is needed.
        documents:  Documents retrieved from the vector store (or web search).
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]