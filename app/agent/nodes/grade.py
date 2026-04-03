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

Pay close attention to these specific criteria:

- **Set matching**: If the query specifies a set (e.g., "Brilliant Stars"), the card MUST be from that exact set to be relevant. Cards from other sets are not relevant even if the Pokemon name matches.
- **Card number matching**: If the query includes a card number (e.g., "154/172"), the card MUST have that exact collector number. A different card with a similar number or from the same set is not relevant.
- **Artist matching**: If the query specifies an artist (e.g., "Mitsuhiro Arita"), the card MUST be illustrated by that artist. Prioritize the most iconic/well-known card matching the artist+Pokemon combination (e.g., Mitsuhiro Arita + Charizard = Base Set Charizard 4/102).
- **Nickname/slang matching**: If the query uses a popular nickname (e.g., "Moonbreon" = Umbreon VMAX from Evolving Skies), the card MUST be the specific card that nickname refers to, not just any card featuring that Pokemon.
- **Name matching**: The card's Pokemon or card name must match what the user is searching for. Do not accept cards of different Pokemon or trainers as substitutes.

If any required attribute (set, number, artist, nickname target) does not match, grade as "not_relevant".

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

    set_name = card.get("set_name", "")
    rarity = card.get("rarity", "")
    collector_number = card.get("collector_number", "")

    if set_name:
        card_desc += f"\nSet: {set_name}"
    if rarity:
        card_desc += f" | Rarity: {rarity}"
    if collector_number:
        card_desc += f" | Number: {collector_number}"

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
