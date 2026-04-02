import os
from typing import Literal

from app.agent.state import AgentState


def route_after_analyze(state: AgentState) -> Literal["retrieve"]:
    # Both simple and complex queries go to retrieve.
    # The retrieve node reads is_complex to decide fan-out vs single search.
    return "retrieve"


def route_after_grade(state: AgentState) -> Literal["generate_answer", "rewrite_query"]:
    relevant_count = len(state.get("relevant_docs", []))
    iteration = state.get("iteration_count", 1)
    min_relevant = int(os.getenv("MIN_RELEVANT_DOCS", "3"))
    max_iterations = int(os.getenv("MAX_RETRIEVE_ITERATIONS", "3"))

    if relevant_count >= min_relevant:
        return "generate_answer"
    if iteration >= max_iterations:
        # Exhausted retries — generate with whatever is available
        return "generate_answer"
    return "rewrite_query"
