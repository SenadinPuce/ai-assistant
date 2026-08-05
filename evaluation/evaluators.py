"""LangSmith evaluators for the CRAG assistant's answer quality.

Each evaluator follows the LangSmith `(run, example) -> dict` contract, where
the returned dict is `{"key": ..., "score": ..., "comment": ...}`.
"""

import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from rag.constants import LLM_MODEL

_judge_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


class _CorrectnessGrade(BaseModel):
    correct: bool = Field(
        description="Whether the generated answer is factually consistent with the reference answer."
    )
    reasoning: str = Field(description="Brief justification for the grade.")


_correctness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are grading whether a generated answer is factually consistent with a "
            "reference answer for a question-answering system. The generated answer "
            "doesn't need to match wording, only the factual content that matters.",
        ),
        (
            "human",
            "Question: {question}\n\nReference answer: {reference}\n\nGenerated answer: {answer}",
        ),
    ]
)

_correctness_chain = _correctness_prompt | _judge_llm.with_structured_output(_CorrectnessGrade)


def correctness_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """LLM-as-judge: does the generated answer match the reference answer's facts?"""
    question = example.inputs["question"]
    reference = example.outputs["reference"]
    answer = (run.outputs or {}).get("answer", "")

    grade = _correctness_chain.invoke(
        {"question": question, "reference": reference, "answer": answer}
    )

    return {"key": "correctness", "score": 1.0 if grade.correct else 0.0, "comment": grade.reasoning}


def cites_sources_evaluator(run: Run, example: Example) -> dict[str, Any]:
    """Checks that citation markers ([1], [2], ...) appear whenever sources were used."""
    outputs = run.outputs or {}
    answer = outputs.get("answer", "")
    sources = outputs.get("sources") or []

    if not sources:
        return {"key": "cites_sources", "score": 1.0}

    has_citation = bool(re.search(r"\[\d+\]", answer))
    return {"key": "cites_sources", "score": 1.0 if has_citation else 0.0}
