"""
generate_answer node — Claude synthesizes a final answer from relevant cards.
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


GENERATE_SYSTEM = """You are a helpful Pokemon card expert assistant.

The user asked a question and relevant Pokemon cards have been found for them.
Write a clear, helpful answer that:
- Directly addresses the user's question
- Highlights the specific attacks, abilities, or attributes that make each card relevant
- Is concise (3-5 sentences or a short bulleted list)
- Uses Pokemon card terminology naturally

Do not mention the search process or that you retrieved documents."""


def format_card_for_prompt(card: dict) -> str:
    name = card.get("name", "Unknown")
    types = ", ".join(card.get("types", [])) or card.get("supertype", "")
    hp = card.get("hp", "?")
    stage = card.get("stage", "")
    attacks = card.get("attacks_text", "")
    abilities = card.get("abilities_text", "")
    set_name = card.get("set_name", "")

    parts = [f"**{name}**"]
    if stage:
        parts.append(f"({stage})")
    parts.append(f"— {types}, HP {hp}")
    if set_name:
        parts.append(f"[{set_name}]")
    if attacks:
        parts.append(f"\n  Attacks: {attacks}")
    if abilities:
        parts.append(f"\n  Abilities: {abilities}")
    return " ".join(parts)


async def generate_answer(state: AgentState) -> dict:
    query = state["original_query"]
    relevant = state.get("relevant_docs", [])
    iteration_count = state.get("iteration_count", 1)

    # Fall back to all retrieved docs if grading found nothing
    cards_to_use = relevant if relevant else state.get("retrieved_docs", [])
    cards_for_prompt = cards_to_use[:8]  # cap for prompt length; final_cards returns all

    if not cards_to_use:
        answer = (
            "I couldn't find Pokemon cards that closely match your query. "
            "Try rephrasing with different terms — for example, mention specific "
            "energy types, attack effects, or card stages."
        )
        step: TrajectoryStep = {
            "node": "generate_answer",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detail": "No relevant cards found — generated fallback response",
            "metadata": {"relevant_count": 0, "total_iterations": iteration_count},
        }
        return {
            "final_answer": answer,
            "final_cards": [],
            "trajectory": [step],
        }

    cards_text = "\n\n".join(format_card_for_prompt(c) for c in cards_for_prompt)
    user_msg = f'User question: "{query}"\n\nRelevant Pokemon cards found:\n\n{cards_text}'

    client = get_client()
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=GENERATE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    answer = message.content[0].text.strip()

    step = {
        "node": "generate_answer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": f"Generated answer from {len(cards_for_prompt)} cards (returning {len(cards_to_use)} total)",
        "metadata": {
            "relevant_count": len(cards_to_use),
            "total_iterations": iteration_count,
            "card_names": [c.get("name", "") for c in cards_for_prompt],
        },
    }

    return {
        "final_answer": answer,
        "final_cards": cards_to_use,
        "trajectory": [step],
    }
