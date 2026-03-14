# ---------------------------------------------------------------------------
# Graph node names
# ---------------------------------------------------------------------------
ROUTE_QUESTION = "route_question"
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
WEB_SEARCH = "web_search"
GENERATE_ANSWER = "generate_answer"
DIRECT_ANSWER = "direct_answer"

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"

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
