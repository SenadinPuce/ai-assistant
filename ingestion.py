import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

from graph.consts import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    PDF_DIR,
    PINECONE_INDEX_NAME,
    SIMILARITY_THRESHOLD,
)


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)


def _ensure_index_exists() -> None:
    """Create the Pinecone index if it does not already exist."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def ingest_documents(pdf_dir: str = PDF_DIR) -> None:
    """Load PDFs, split into chunks, and upsert into Pinecone.

    Run this once (or whenever the document set changes).
    """
    _ensure_index_exists()

    docs = PyPDFDirectoryLoader(pdf_dir).load()
    doc_splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    ).split_documents(docs)

    PineconeVectorStore.from_documents(
        documents=doc_splits,
        embedding=_get_embeddings(),
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"Ingested {len(doc_splits)} chunks into '{PINECONE_INDEX_NAME}'.")


def get_vectorstore() -> PineconeVectorStore:
    """Return the Pinecone vectorstore instance."""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
    )


# Initialized once at import time — avoids a Pinecone network round-trip on every query.
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


def search_with_scores(
    query: str, k: int = 4
) -> list[tuple[Document, float]]:
    """Return (document, score) pairs and print them for inspection.

    Useful for debugging — call this directly to inspect what the retriever
    sees before the score threshold is applied.
    """
    results = _vectorstore.similarity_search_with_score(query, k=k)
    for i, (doc, score) in enumerate(results, start=1):
        print(f"[{i}] score={score:.4f} | {doc.page_content[:120]!r}")
    return results


if __name__ == "__main__":
    ingest_documents()
