import logging


from rag.chains.generation import generation_chain
from rag.state import GraphState

logger = logging.getLogger(__name__)


def direct_answer_node(state: GraphState) -> GraphState:
    """Answer the question directly without retrieval — for greetings, simple queries, etc."""
    logger.info("Routing to direct answer (no retrieval needed).")

    question = state["question"]
    generation = generation_chain.invoke({"question": question, "context": ""})

    return {"generation": generation, "documents": []}
