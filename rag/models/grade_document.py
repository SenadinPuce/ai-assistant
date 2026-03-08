from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: bool = Field(
        description="True if the document is relevant to the question, False otherwise."
    )
