import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.vectorstores import VectorStoreRetriever
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


def get_retriever(k: int = 4, score_threshold: float = SIMILARITY_THRESHOLD) -> VectorStoreRetriever:
    """Return a retriever backed by the existing Pinecone index.

    Args:
        k: Maximum number of documents to retrieve.
        score_threshold: Minimum cosine similarity score. Documents below
            this threshold are discarded before they reach the grading node,
            reducing unnecessary LLM calls.
    """
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=_get_embeddings(),
    )
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


if __name__ == "__main__":
    ingest_documents()