from pydantic import BaseModel, Field


class RouteQuestion(BaseModel):
    """Routes a user question to either vectorstore retrieval or direct LLM answer."""

    datasource: str = Field(
        description=(
            "'vectorstore' if the question requires looking up information from "
            "the knowledge base, or 'direct' if it can be answered directly "
            "(e.g. greetings, clarifications, general knowledge)."
        )
    )
