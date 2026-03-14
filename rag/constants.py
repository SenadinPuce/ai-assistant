# ---------------------------------------------------------------------------
# Graph node names
# ---------------------------------------------------------------------------
ROUTE_QUESTION = "route_question"
REWRITE_QUERY = "rewrite_query"
RETRIEVE = "retrieve"
RERANK_DOCUMENTS = "rerank_documents"
WEB_SEARCH = "web_search"
GENERATE_ANSWER = "generate_answer"
DIRECT_ANSWER = "direct_answer"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Reranker settings
# ---------------------------------------------------------------------------
RERANKER_MODEL = "corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1"
RERANKER_SCORE_THRESHOLD = 0.0
RERANKER_TOP_N = 3

# ---------------------------------------------------------------------------
# Web search settings
# ---------------------------------------------------------------------------
WEB_SEARCH_MAX_RESULTS = 3
