from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )


_system = (
    "You are a grader assessing the relevance of a retrieved document to a user question. "
    "If the document contains keywords or semantic meaning related to the question, grade it as relevant. "
    "Return 'yes' if relevant, 'no' otherwise."
)

_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
    ]
)

retrieval_grader_chain = _grade_prompt | _llm.with_structured_output(GradeDocument)
