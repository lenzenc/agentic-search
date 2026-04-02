"""
grade_documents node — Claude grades each retrieved card for relevance to the query.
Grades run concurrently (asyncio.gather).
"""
import asyncio
import json
import os
from datetime import datetime, timezone

import anthropic

from app.agent.state import AgentState, GradedDocument, TrajectoryStep

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


GRADE_SYSTEM = """You are grading whether a Pokemon card is relevant to a user's search query.

Be strict but fair. Grade "relevant" only if the card meaningfully matches what the user is looking for.

Respond ONLY with valid JSON:
{"grade": "relevant" | "not_relevant", "reasoning": "one concise sentence explaining your decision"}"""


async def grade_one(client: anthropic.AsyncAnthropic, query: str, card: dict) -> GradedDocument:
    name = card.get("name", "Unknown")
    types = ", ".join(card.get("types", [])) or card.get("supertype", "")
    hp = card.get("hp", "?")
    stage = card.get("stage", "")
    attacks = card.get("attacks_text", "")
    abilities = card.get("abilities_text", "")

    card_desc = f"Card: {name}"
    if stage:
        card_desc += f" ({stage})"
    card_desc += f"\nType: {types} | HP: {hp}"
    if attacks:
        card_desc += f"\nAttacks: {attacks}"
    if abilities:
        card_desc += f"\nAbilities: {abilities}"

    message = await client.messages.create(
        model="claude-haiku-4-5-20251001",  # haiku is fast + cheap for grading
        max_tokens=128,
        system=GRADE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f'Query: "{query}"\n\n{card_desc}',
            }
        ],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    grade = parsed.get("grade", "not_relevant")
    reasoning = parsed.get("reasoning", "")

    return GradedDocument(
        card_id=card.get("card_id", ""),
        card_data=card,
        grade=grade,
        reasoning=reasoning,
    )


async def grade_documents(state: AgentState) -> dict:
    query = state.get("active_query") or state["original_query"]
    docs = state.get("retrieved_docs", [])
    client = get_client()

    # Concurrently grade all docs
    tasks = [grade_one(client, query, doc) for doc in docs]
    graded: list[GradedDocument] = await asyncio.gather(*tasks)

    relevant = [g["card_data"] for g in graded if g["grade"] == "relevant"]
    not_relevant_count = len(graded) - len(relevant)

    grade_details = [
        {
            "name": g["card_data"].get("name", ""),
            "grade": g["grade"],
            "reasoning": g["reasoning"],
        }
        for g in graded
    ]

    step: TrajectoryStep = {
        "node": "grade_documents",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": (
            f"Graded {len(graded)} cards against query — "
            f"{len(relevant)} relevant, {not_relevant_count} not relevant"
        ),
        "metadata": {
            "query": query,
            "total_graded": len(graded),
            "relevant_count": len(relevant),
            "not_relevant_count": not_relevant_count,
            "grades": grade_details,
        },
    }

    return {
        "graded_docs": list(graded),
        "relevant_docs": relevant,
        "trajectory": [step],
    }
