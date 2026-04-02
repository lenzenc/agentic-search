"""
rewrite_query node — Claude rewrites the search query based on grading feedback.
Only fires when the self-correcting loop is triggered.
"""
import os
from datetime import datetime, timezone

import anthropic

from app.agent.state import AgentState, TrajectoryStep

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


REWRITE_SYSTEM = """You are a search query optimizer for a Pokemon card database.

The previous query retrieved cards but most were not relevant to the user's intent.
Your job: rewrite the query to better target what the user is looking for.

Focus on Pokemon card-specific terminology: attack names, ability types, damage amounts,
energy types, status conditions (paralysis, confusion, poison, burn, sleep), HP ranges,
card stages (Basic, Stage 1, Stage 2, VMAX, VSTAR, ex, EX, GX, V), set names.

Return ONLY the new query string — no explanation, no quotes."""


async def rewrite_query(state: AgentState) -> dict:
    original = state["original_query"]
    current_query = state.get("active_query", original)
    iteration = state.get("iteration_count", 1)
    max_iter = int(os.getenv("MAX_RETRIEVE_ITERATIONS", "3"))
    graded_docs = state.get("graded_docs", [])

    not_relevant = [g for g in graded_docs if g["grade"] == "not_relevant"]
    relevant = [g for g in graded_docs if g["grade"] == "relevant"]

    not_relevant_summary = "\n".join(
        f"- {g['card_data'].get('name', '')}: {g['reasoning']}"
        for g in not_relevant[:5]
    )
    relevant_summary = "\n".join(
        f"- {g['card_data'].get('name', '')}: {g['reasoning']}"
        for g in relevant[:3]
    ) if relevant else "None found yet."

    user_msg = f"""Original user question: "{original}"
Current search query: "{current_query}"
This is loop iteration {iteration} of {max_iter}.

Cards that were NOT relevant:
{not_relevant_summary}

Cards that WERE relevant (if any):
{relevant_summary}

Rewrite the search query to find more cards like the relevant ones and avoid cards like the irrelevant ones."""

    client = get_client()
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=128,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    new_query = message.content[0].text.strip().strip('"').strip("'")

    step: TrajectoryStep = {
        "node": "rewrite_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": (
            f"Self-correction — rewrote query (loop {iteration} of {max_iter}): "
            f'"{current_query}" → "{new_query}"'
        ),
        "metadata": {
            "old_query": current_query,
            "new_query": new_query,
            "iteration": iteration,
            "max_iterations": max_iter,
            "not_relevant_count": len(not_relevant),
            "relevant_count": len(relevant),
        },
    }

    return {
        "active_query": new_query,
        "trajectory": [step],
        # Reset retrieved/graded for the next retrieval pass
        "retrieved_docs": [],
        "graded_docs": [],
        "relevant_docs": [],
    }
