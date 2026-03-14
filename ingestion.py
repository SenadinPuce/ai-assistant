import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

from retrieval.config import (
    EMBEDDING_DIMENSIONS,
    PDF_DIR,
    PINECONE_INDEX_NAME,
)
from retrieval.vectorstore import _get_embeddings


def _ensure_index_exists() -> None:
    """Create the Pinecone index if missing and wait until it is ready."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    list_result = pc.list_indexes()
    if hasattr(list_result, "names"):
        index_names = set(list_result.names())
    else:
        index_names = {idx.name for idx in list_result}

    if PINECONE_INDEX_NAME not in index_names:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    max_wait_seconds = 90
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        description = pc.describe_index(PINECONE_INDEX_NAME)
        status = getattr(description, "status", {})
        if isinstance(status, dict) and status.get("ready"):
            return
        if hasattr(status, "ready") and status.ready:
            return
        time.sleep(2)

    raise RuntimeError(
        f"Pinecone index '{PINECONE_INDEX_NAME}' is not ready after {max_wait_seconds}s. "
        "Verify PINECONE_API_KEY/PINECONE_INDEX_NAME point to the same Pinecone project."
    )


def ingest_documents(pdf_dir: str = PDF_DIR) -> None:
    """Load PDFs, split into chunks, and upsert into Pinecone.

    Run this once (or whenever the document set changes).
    """
    _ensure_index_exists()

    docs = PyPDFDirectoryLoader(pdf_dir).load()
    doc_splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64
    ).split_documents(docs)

    PineconeVectorStore.from_documents(
        documents=doc_splits,
        embedding=_get_embeddings(),
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"Ingested {len(doc_splits)} chunks into '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    ingest_documents()
