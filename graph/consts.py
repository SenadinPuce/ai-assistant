import os

# ---------------------------------------------------------------------------
# Graph node names
# ---------------------------------------------------------------------------
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
WEB_SEARCH = "web_search"
GENERATE_ANSWER = "generate_answer"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"

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
SIMILARITY_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Grading settings
# ---------------------------------------------------------------------------
# Minimum fraction of retrieved documents that must be graded as relevant
# for the pipeline to skip web search. For example, 0.5 means "at least
# half the documents must be relevant to avoid a web search fallback".
MIN_RELEVANCE_RATIO = 0.5

# ---------------------------------------------------------------------------
# Web search settings
# ---------------------------------------------------------------------------
WEB_SEARCH_MAX_RESULTS = 3