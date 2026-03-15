from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from rag.constants import LLM_MODEL

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Equivalent to the rlm/rag-prompt from LangChain Hub — defined locally to
# avoid a network call at import time and remove the hub dependency.
_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an assistant for question-answering tasks. "
                "Use the retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Keep the answer concise — three sentences maximum. "
                "You MUST respond in {language}. Do not use any other language."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            "Context: {context}\n\nQuestion: {question}",
        ),
    ]
)

generation_chain = _prompt | _llm | StrOutputParser()
