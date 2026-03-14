from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from rag.constants import LLM_MODEL

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Equivalent to the rlm/rag-prompt from LangChain Hub — defined locally to
# avoid a network call at import time and remove the hub dependency.
_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            (
                "You are an assistant for question-answering tasks. "
                "Use the following retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Keep the answer concise — three sentences maximum. "
                "Always answer in the same language as the question.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )
    ]
)

generation_chain = _prompt | _llm | StrOutputParser()
