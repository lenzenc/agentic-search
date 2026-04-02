from typing import Annotated, TypedDict, Literal
import operator


class SubQuery(TypedDict):
    query: str
    retrieved_ids: list[str]


class GradedDocument(TypedDict):
    card_id: str
    card_data: dict
    grade: Literal["relevant", "not_relevant"]
    reasoning: str


class TrajectoryStep(TypedDict):
    node: str
    timestamp: str
    detail: str
    metadata: dict


class AgentState(TypedDict):
    # Input
    original_query: str

    # Query decomposition
    is_complex: bool
    sub_queries: list[SubQuery]
    active_query: str

    # Retrieval
    retrieved_docs: list[dict]
    iteration_count: int

    # Grading
    graded_docs: list[GradedDocument]
    relevant_docs: list[dict]

    # Generation
    final_answer: str
    final_cards: list[dict]

    # Append-only trajectory — drives SSE stream
    trajectory: Annotated[list[TrajectoryStep], operator.add]
