import logging

from langgraph.graph import END, StateGraph

from rag.chains.question_router import question_router_chain
from rag.constants import (
    DIRECT_ANSWER,
    GENERATE_ANSWER,
    GRADE_DOCUMENTS,
    RETRIEVE,
    ROUTE_QUESTION,
    WEB_SEARCH,
)
from rag.nodes import (
    direct_answer_node,
    generate_answer_node,
    grade_documents_node,
    retrieve_documents_node,
    web_search_node,
)
from rag.state import GraphState

logger = logging.getLogger(__name__)


def _route_question(state: GraphState) -> str:
    """Route the question: use retrieval or answer directly."""
    result = question_router_chain.invoke({"question": state["question"]})
    if result.datasource == "vectorstore":
        logger.info("Router decision: question requires retrieval.")
        return RETRIEVE
    logger.info("Router decision: answering directly (no retrieval needed).")
    return DIRECT_ANSWER


def _decide_to_generate(state: GraphState) -> str:
    """Route after grading: trigger web search if relevance ratio is too low."""
    if state["web_search"]:
        logger.info(
            "Decision: relevance ratio %.2f below threshold — running web search.",
            state.get("relevance_ratio", 0.0),
        )
        return WEB_SEARCH
    logger.info(
        "Decision: relevance ratio %.2f is sufficient — generating answer.",
        state.get("relevance_ratio", 1.0),
    )
    return GENERATE_ANSWER


def build_graph() -> StateGraph:
    """Construct and compile the CRAG workflow graph with question routing."""
    workflow = StateGraph(GraphState)

    workflow.add_node(ROUTE_QUESTION, lambda state: state)  # passthrough; routing is in the edge
    workflow.add_node(RETRIEVE, retrieve_documents_node)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents_node)
    workflow.add_node(WEB_SEARCH, web_search_node)
    workflow.add_node(GENERATE_ANSWER, generate_answer_node)
    workflow.add_node(DIRECT_ANSWER, direct_answer_node)

    workflow.set_entry_point(ROUTE_QUESTION)
    workflow.add_conditional_edges(
        ROUTE_QUESTION,
        _route_question,
        {
            RETRIEVE: RETRIEVE,
            DIRECT_ANSWER: DIRECT_ANSWER,
        },
    )
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        _decide_to_generate,
        {
            WEB_SEARCH: WEB_SEARCH,
            GENERATE_ANSWER: GENERATE_ANSWER,
        },
    )
    workflow.add_edge(WEB_SEARCH, GENERATE_ANSWER)
    workflow.add_edge(GENERATE_ANSWER, END)
    workflow.add_edge(DIRECT_ANSWER, END)

    return workflow.compile()


app = build_graph()
