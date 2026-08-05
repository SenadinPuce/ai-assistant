"""Manual evaluation entrypoint — runs the CRAG graph against a golden Q&A set
via LangSmith and prints the resulting experiment name for review in the UI.

Requires real OPENAI_API_KEY / PINECONE_API_KEY / TAVILY_API_KEY / LANGSMITH_API_KEY
in .env. Not part of the pytest suite or CI — run manually:

    uv run python -m evaluation.run_eval
"""

import logging

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

from langsmith import Client, evaluate

from evaluation.dataset import QA_EXAMPLES
from evaluation.evaluators import cites_sources_evaluator, correctness_evaluator
from rag.graph import app as rag_app
from rag.language import _get_detector
from rag.reranker import _get_reranker, rerank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "crag-ai-assistant-golden-set"


def _warmup_local_models() -> None:
    """Warm heavy local components so first evaluated question is not penalized."""
    _get_reranker()
    _get_detector()
    rerank("warmup", [Document(page_content="warmup", metadata={"source": "warmup"})])


def _ensure_dataset(client: Client) -> None:
    """Create and populate the golden dataset in LangSmith if it doesn't exist yet."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists — skipping upload.", DATASET_NAME)
        return

    client.create_dataset(
        DATASET_NAME,
        description="Golden Q&A set for evaluating the CRAG assistant's answer quality.",
    )
    client.create_examples(
        dataset_name=DATASET_NAME,
        examples=[
            {
                "inputs": {"question": item["question"]},
                "outputs": {"reference": item["reference"]},
                "metadata": {"expects_retrieval": item["expects_retrieval"]},
            }
            for item in QA_EXAMPLES
        ],
    )
    logger.info("Uploaded %d examples to dataset '%s'.", len(QA_EXAMPLES), DATASET_NAME)


def target(inputs: dict) -> dict:
    """Invoke the compiled CRAG graph for a single evaluation example."""
    result = rag_app.invoke({"question": inputs["question"], "chat_history": []})
    return {"answer": result.get("generation", ""), "sources": result.get("sources", [])}


def main() -> None:
    client = Client()
    _warmup_local_models()
    _ensure_dataset(client)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, cites_sources_evaluator],
        experiment_prefix="crag-eval",
        max_concurrency=2,
    )

    logger.info("Evaluation experiment: %s", results.experiment_name)


if __name__ == "__main__":
    main()
