import os

# ---------------------------------------------------------------------------
# Pinecone / embedding settings
# ---------------------------------------------------------------------------
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# ---------------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------------
PDF_DIR = "./docs"

# ---------------------------------------------------------------------------
# Retriever settings
# ---------------------------------------------------------------------------
# Minimum cosine similarity for a document to be considered a candidate.
# Documents below this threshold are discarded before LLM grading,
# saving API calls on obviously irrelevant chunks.
SIMILARITY_THRESHOLD = 0.45
