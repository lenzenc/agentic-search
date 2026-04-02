from pydantic import BaseModel, Field
from app.models.card import CardResult


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class TrajectoryStep(BaseModel):
    node: str
    timestamp: str
    detail: str
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    answer: str
    cards: list[CardResult]
    trajectory: list[TrajectoryStep]
    total_iterations: int
    pattern: str  # "self_correcting_loop" | "query_expansion" | "direct"
