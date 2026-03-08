from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.consts import LLM_MODEL
from graph.models.grade_document import GradeDocument

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)


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
