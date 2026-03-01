import logging

from graph.chains.retrieval_grader_chain import retrieval_grader_chain
from graph.state import GraphState

logger = logging.getLogger(__name__)


def grade_documents(state: GraphState) -> GraphState:
    """Filter retrieved documents to those relevant to the question.

    Sets ``web_search=True`` if any document is graded as not relevant,
    signalling downstream nodes to supplement with a web search.
    """
    logger.info("Grading document relevance to question.")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score == "yes":
            logger.info("Document graded as relevant.")
            filtered_docs.append(doc)
        else:
            logger.info("Document graded as not relevant — web search flagged.")
            web_search = True

    return {"documents": filtered_docs, "question": question, "web_search": web_search}
