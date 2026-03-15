from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from rag.constants import LLM_MODEL

_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a query rewriter that optimizes a user question for "
                "semantic similarity search against an academic knowledge base. "
                "If the conversation history is provided, use it to resolve "
                "references and make the question self-contained (e.g. replace "
                "'that', 'it', 'this topic' with the actual subject). "
                "Rewrite the question to be more specific, self-contained, and "
                "keyword-rich so that it retrieves the most relevant documents. "
                "Preserve the original intent and language. "
                "Output ONLY the rewritten query, nothing else."
            ),
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            "{question}",
        ),
    ]
)

query_rewriter_chain = _prompt | _llm | StrOutputParser()
