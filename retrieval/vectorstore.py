from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from retrieval.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME,
    SIMILARITY_THRESHOLD,
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)


def get_vectorstore() -> PineconeVectorStore:
    """Return the Pinecone vectorstore instance."""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
    )


_vectorstore = get_vectorstore()


def retrieve_with_score_filter(
    query: str,
    k: int = 4,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> list:
    """Return documents whose cosine similarity meets the threshold.

    Uses ``similarity_search_with_score`` and filters manually because
    PineconeVectorStore's ``score_threshold`` search type does not reliably
    enforce the threshold in all langchain-pinecone versions.
    """
    results = _vectorstore.similarity_search_with_score(query, k=k)
    return [doc for doc, score in results if score >= score_threshold]


def search_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return (document, score) pairs and print them for inspection.

    Useful for debugging — call this directly to inspect what the retriever
    sees before the score threshold is applied.
    """
    results = _vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"[{i}] score={score:.4f} | {doc.page_content[:120]!r}")
    return results
