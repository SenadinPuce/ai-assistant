from typing import Any, List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question:          The user's input question (may be rewritten for retrieval).
        original_question: The original user question before rewriting.
        generation:        The final LLM-generated answer.
        web_search:        Flag indicating whether a web search step is needed.
        documents:         Documents retrieved from the vector store (or web search).
        relevance_ratio:   Fraction of retrieved documents graded as relevant (0.0 – 1.0).
        sources:           Deduplicated source references extracted from retrieved documents.
    """

    question: str
    original_question: str
    generation: str
    web_search: bool
    documents: List[Document]
    relevance_ratio: float
    sources: List[dict[str, Any]]
