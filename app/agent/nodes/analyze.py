"""
analyze_query node — classifies query complexity and decomposes into sub-queries if needed.
"""
import json
import os
from datetime import datetime, timezone

import anthropic

from app.agent.state import AgentState, SubQuery, TrajectoryStep

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


SYSTEM_PROMPT = """You are a search query analyst for a Pokemon card database.

Your job: decide if a user query is "simple" (single intent, one search covers it) or "complex" (multiple independent facets that benefit from separate searches).

Examples of SIMPLE queries:
- "fire type pokemon" — single type filter
- "Charizard" — specific card name
- "psychic pokemon with high HP" — one type + one attribute
- "pokemon with more than 200 HP" — single attribute

Examples of COMPLEX queries:
- "electric pokemon that can paralyze or confuse" — type + specific status effects (need separate searches)
- "water type pokemon with healing abilities and strong attacks" — healing + attack strength are independent facets
- "grass pokemon that poison opponents or have abilities that restore energy" — two distinct ability types

If simple: return is_complex=false and sub_queries=[the original query].
If complex: return is_complex=true and 2-4 targeted sub_queries.

Sub-queries should be specific search strings optimized for a Pokemon card search engine.

Respond ONLY with valid JSON matching this schema:
{"is_complex": boolean, "sub_queries": ["query1", "query2", ...]}"""


async def analyze_query(state: AgentState) -> dict:
    query = state["original_query"]
    client = get_client()

    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Query: {query}"}],
    )

    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    is_complex: bool = parsed.get("is_complex", False)
    sub_query_strings: list[str] = parsed.get("sub_queries", [query])

    # Always have at least the original query
    if not sub_query_strings:
        sub_query_strings = [query]

    sub_queries: list[SubQuery] = [
        {"query": q, "retrieved_ids": []} for q in sub_query_strings
    ]

    if is_complex:
        detail = f"Complex query — expanding into {len(sub_queries)} sub-queries: {sub_query_strings}"
    else:
        detail = f"Simple query — proceeding directly to retrieval"

    step: TrajectoryStep = {
        "node": "analyze_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
        "metadata": {
            "original_query": query,
            "is_complex": is_complex,
            "sub_queries": sub_query_strings,
        },
    }

    return {
        "is_complex": is_complex,
        "sub_queries": sub_queries,
        "active_query": sub_query_strings[0],
        "trajectory": [step],
    }
