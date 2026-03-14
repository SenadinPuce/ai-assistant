from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Incoming user question."""

    question: str = Field(..., min_length=1, max_length=2000)


class SourceReference(BaseModel):
    """A single source reference (PDF page or web URL)."""

    source: str | None = None
    page: int | None = None
    url: str | None = None


class AnswerResponse(BaseModel):
    """Generated answer with optional source references."""

    answer: str
    sources: list[SourceReference] = []
