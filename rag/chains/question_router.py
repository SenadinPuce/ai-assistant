from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.constants import LLM_MODEL
from rag.models.route_question import RouteQuestion

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

_system = (
    "You are a router that decides whether a user question requires looking up "
    "information from a knowledge base about academic subjects (poslovni sistemi, "
    "informacioni sistemi, etc.) or can be answered directly.\n\n"
    "Route to 'vectorstore' if the question asks about domain-specific topics "
    "that require retrieved context.\n"
    "Route to 'direct' if the question is a greeting, a clarification, "
    "or a simple general-knowledge question that does not need retrieval."
)

_route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        ("human", "{question}"),
    ]
)

question_router_chain = _route_prompt | _llm.with_structured_output(RouteQuestion)
