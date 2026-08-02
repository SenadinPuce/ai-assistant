from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str
    content: str


class QuestionRequest(BaseModel):
    """Incoming user question."""

    question: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = []


class SourceReference(BaseModel):
    """A single source reference (PDF page or web URL)."""

    source: str | None = None
    page: int | None = None
    url: str | None = None
    snippet: str | None = None


class AnswerResponse(BaseModel):
    """Generated answer with optional source references."""

    answer: str
    sources: list[SourceReference] = []
