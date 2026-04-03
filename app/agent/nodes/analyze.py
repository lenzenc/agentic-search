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

IMPORTANT — recognize these specific query patterns and treat them as SIMPLE:

1. **Name + Set queries**: e.g., "Charizard Brilliant Stars" → single search for that card in that set
2. **Card number queries**: e.g., "154/172" or "Charizard 154/172" → single search by number/name+number
3. **Artist + Pokemon queries**: e.g., "Mitsuhiro Arita Charizard" → single search combining artist and card name
4. **Slang/nickname queries**: e.g., "Moonbreon" (= Umbreon VMAX Evolving Skies), "Shadowless Charizard", "Gold Star" cards → translate the nickname into the real card name/set and use a single targeted search
5. **Single Pokémon name queries** (just a Pokémon species name, no set/mechanic/number/type qualifier):
   e.g., "Charizard", "Pikachu", "Mewtwo" → treat as SIMPLE with sub_queries=[the bare name].
   A direct name search retrieves all cards for that Pokémon across all eras — do NOT decompose into era sub-queries.

For slang/nicknames, expand the query to its known meaning:
- "Moonbreon" → "Umbreon VMAX Evolving Skies"
- "Skyridge" → search for cards from the Skyridge set
- Use your knowledge of Pokemon TCG community nicknames to resolve these before searching

Examples of SIMPLE queries:
- "fire type pokemon" — single type filter
- "Charizard Brilliant Stars" — name + set
- "154/172" — card number lookup
- "Charizard 154/172" — name + number lookup
- "Mitsuhiro Arita Charizard" — artist + name
- "Moonbreon" — slang for Umbreon VMAX Evolving Skies
- "psychic pokemon with high HP" — one type + one attribute
- "Charizard" — single Pokémon name, direct name lookup
- "Pikachu" — single Pokémon name, direct name lookup

Examples of COMPLEX queries:
- "electric pokemon that can paralyze or confuse" — type + multiple distinct status effects
- "water type pokemon with healing abilities and strong attacks" — healing + attack strength are independent facets
- "grass pokemon that poison opponents or have abilities that restore energy" — two distinct ability types

If simple: return is_complex=false and sub_queries=[the best search string for this query — expand slang/nicknames to their real card names].
If complex: return is_complex=true and 2-4 targeted sub_queries.

Sub-queries should be specific search strings optimized for a Pokemon card search engine. For nickname/slang queries, always expand to the real card name and set in the sub_query.

Also extract detected_set: if the query specifies a set name (e.g. "Brilliant Stars", "Evolving Skies", "Base Set"), output the exact set name as it appears in the Pokemon TCG. If no set is specified, output "".

Respond ONLY with valid JSON matching this schema:
{"is_complex": boolean, "sub_queries": ["query1", "query2", ...], "detected_set": "set name or empty string"}"""


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

    # Robustly extract JSON object — ignore any text before/after
    start = raw.find('{')
    end = raw.rfind('}') + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        import re
        is_complex_m = re.search(r'"is_complex"\s*:\s*(true|false)', raw)
        is_complex_val = is_complex_m.group(1) == "true" if is_complex_m else False
        parsed = {"is_complex": is_complex_val, "sub_queries": [query], "detected_set": ""}
    is_complex: bool = parsed.get("is_complex", False)
    sub_query_strings: list[str] = parsed.get("sub_queries", [query])
    detected_set: str = parsed.get("detected_set", "")

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
    if detected_set:
        detail += f" | Set filter: {detected_set}"

    step: TrajectoryStep = {
        "node": "analyze_query",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
        "metadata": {
            "original_query": query,
            "is_complex": is_complex,
            "sub_queries": sub_query_strings,
            "detected_set": detected_set,
        },
    }

    return {
        "is_complex": is_complex,
        "sub_queries": sub_queries,
        "active_query": sub_query_strings[0],
        "detected_set": detected_set,
        "trajectory": [step],
    }
