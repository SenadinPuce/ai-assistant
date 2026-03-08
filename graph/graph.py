import logging

from langgraph.graph import END, StateGraph

from graph.consts import GENERATE_ANSWER, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from graph.nodes import generate_answer, grade_documents, retrieve_documents, web_search
from graph.state import GraphState

logger = logging.getLogger(__name__)


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
    """Construct and compile the CRAG workflow graph."""
    workflow = StateGraph(GraphState)

    workflow.add_node(RETRIEVE, retrieve_documents)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(WEB_SEARCH, web_search)
    workflow.add_node(GENERATE_ANSWER, generate_answer)

    workflow.set_entry_point(RETRIEVE)
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

    return workflow.compile()


app = build_graph()
