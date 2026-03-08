from typing import Literal

from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """Binary relevance score for a retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' otherwise."
    )
