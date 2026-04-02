"""Agent orchestrator — plain Python, no framework.

The loop is explicit:
  analyze_query → retrieve → grade_documents → [rewrite_query → retrieve → ...]* → generate_answer

Each node is an async function that takes the current state dict and returns a
dict of updates. _apply() merges updates back into state, appending to the
trajectory list rather than replacing it.
"""
from typing import AsyncGenerator

from app.agent.state import AgentState
from app.agent.nodes.analyze import analyze_query
from app.agent.nodes.retrieve import retrieve
from app.agent.nodes.grade import grade_documents
from app.agent.nodes.rewrite import rewrite_query
from app.agent.nodes.generate import generate_answer
from app.agent.conditionals import route_after_grade


def _apply(state: dict, updates: dict) -> None:
    """Merge node updates into state. Trajectory is append-only."""
    for k, v in updates.items():
        if k == "trajectory":
            state[k].extend(v)
        else:
            state[k] = v


async def run_agent(state: AgentState) -> AsyncGenerator[dict, None]:
    """Execute the agentic search loop, yielding full state after each step."""

    # Step 1: Classify query complexity, optionally decompose into sub-queries
    _apply(state, await analyze_query(state))
    yield state

    # Step 2: Retrieve → Grade → rewrite loop (or proceed to generate)
    while True:
        _apply(state, await retrieve(state))
        yield state

        _apply(state, await grade_documents(state))
        yield state

        if route_after_grade(state) == "rewrite_query":
            # Not enough relevant docs — rewrite and search again
            _apply(state, await rewrite_query(state))
            yield state
        else:
            break

    # Step 3: Synthesize final answer from relevant cards
    _apply(state, await generate_answer(state))
    yield state
