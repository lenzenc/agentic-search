"""
LLM judge for evaluating search results against the golden dataset.
Uses claude-sonnet-4-6 to score each result 0.0–1.0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

PASS_THRESHOLD = 0.7

# Categories where the current system likely lacks data — judge scores leniently
OUT_OF_SCOPE_CATEGORIES = {"graded_card", "foreign_language", "sealed_product", "vintage_edition"}

_CAPABILITY_NOTES: dict[str, str] = {
    "graded_card": (
        "SYSTEM CAPABILITY NOTE: This system does not track graded card condition or scores (PSA, BGS, etc.). "
        "If the correct base card is returned, that should be considered a directional pass."
    ),
    "foreign_language": (
        "SYSTEM CAPABILITY NOTE: This system may not index Japanese or non-English cards. "
        "If the correct English-language equivalent is returned, that should score well."
    ),
    "sealed_product": (
        "SYSTEM CAPABILITY NOTE: This system indexes individual card singles, not sealed products "
        "(booster boxes, ETBs, etc.). A graceful 'no matching results' response should score well."
    ),
    "vintage_edition": (
        "SYSTEM CAPABILITY NOTE: Edition distinctions (1st Edition stamps, Shadowless) may not be "
        "a searchable field in this system. Returning the correct base card should be accepted as a pass."
    ),
}

_JUDGE_SYSTEM = """You are evaluating the results of a Pokemon card search engine.

Your task: score a search result 0.0–1.0 based on how well it matches what the user was looking for.

Scoring guidelines:
- 1.0: Perfect — returned exactly the right cards with a relevant, accurate answer
- 0.8–0.9: Good — mostly correct, minor gaps (missing a variant, slight imprecision)
- 0.7: Acceptable pass — directionally correct, found the right card family/type
- 0.4–0.6: Partial — some relevant cards but mixed with wrong ones, or missing key results
- 0.1–0.3: Poor — mostly wrong cards, or major missed intent
- 0.0: Failed — wrong Pokemon entirely, returned negative results that should be excluded, or empty when cards exist

Heavy penalties for:
- Returning a card explicitly listed in MUST NOT RETURN
- Returning completely wrong Pokemon (e.g., Charmander when Charizard was requested)
- Confidently wrong answer text (hallucinated card names or abilities not in the result set)

Partial credit for:
- Finding the right Pokemon family when an exact variant isn't available
- Returning relevant cards even if not all expected ones are present
- Honest answer acknowledging limited results

Respond ONLY with valid JSON:
{"score": <float 0.0-1.0>, "reasoning": "<one to three sentences>", "failure_category": "<category or null>"}

Valid failure_category values: "wrong_cards", "negative_result_returned", "empty_result", "missing_key_variant", "hallucinated_answer", null"""


@dataclass
class JudgeResult:
    case_id: str
    score: float
    passed: bool
    reasoning: str
    failure_category: str | None


def _format_expected(expected_results: list[dict], max_items: int = 10) -> str:
    if not expected_results:
        return "None specified"
    items = expected_results[:max_items]
    lines = []
    for item in items:
        # Pull the most informative fields
        parts = []
        for key in ("name", "pokemon", "set", "number", "rarity", "type", "mechanic"):
            if key in item and item[key]:
                parts.append(f"{key}={item[key]}")
        if "note" in item:
            parts.append(f"note={item['note']}")
        lines.append(", ".join(parts) if parts else str(item))
    suffix = f"\n  ... and {len(expected_results) - max_items} more" if len(expected_results) > max_items else ""
    return "\n  - " + "\n  - ".join(lines) + suffix


def _format_negative(negative_results: list) -> str:
    if not negative_results:
        return "None"
    items = []
    for item in negative_results[:10]:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, dict):
            name = item.get("name") or item.get("pokemon") or str(item)
            items.append(name)
    return ", ".join(items)


def _format_actual_cards(cards: list[dict], max_cards: int = 20) -> str:
    if not cards:
        return "No cards returned"
    lines = []
    for card in cards[:max_cards]:
        name = card.get("name", "?")
        set_name = card.get("set_name", "?")
        types = ", ".join(card.get("types", [])) or "?"
        rarity = card.get("rarity", "?")
        lines.append(f"  - {name} | {set_name} | {types} | {rarity}")
    if len(cards) > max_cards:
        lines.append(f"  ... and {len(cards) - max_cards} more")
    return "\n".join(lines)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_judge(
    user_message: str,
    client: anthropic.AsyncAnthropic,
) -> dict:
    message = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


async def judge_case(
    case_id: str,
    query: str,
    intent: str,
    category: str,
    difficulty: str,
    expected_results: list[dict],
    negative_results: list,
    actual_cards: list[dict],
    actual_answer: str,
    client: anthropic.AsyncAnthropic,
) -> JudgeResult:
    capability_note = _CAPABILITY_NOTES.get(category, "")

    user_message_parts = [
        f'QUERY: "{query}"',
        f"INTENT: {intent}",
        f"CATEGORY: {category} (DIFFICULTY: {difficulty})",
    ]
    if capability_note:
        user_message_parts.append(capability_note)

    user_message_parts += [
        f"\nEXPECTED RESULTS:\n{_format_expected(expected_results)}",
        f"\nMUST NOT RETURN: {_format_negative(negative_results)}",
        f"\nACTUAL ANSWER (first 500 chars):\n{actual_answer[:500]}",
        f"\nACTUAL CARDS RETURNED ({len(actual_cards)} cards):\n{_format_actual_cards(actual_cards)}",
    ]

    user_message = "\n".join(user_message_parts)

    try:
        parsed = await _call_judge(user_message, client)
        score = float(parsed.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = parsed.get("reasoning", "")
        failure_category = parsed.get("failure_category") or None
    except Exception as exc:
        score = 0.0
        reasoning = f"Judge call failed: {exc}"
        failure_category = "judge_error"

    return JudgeResult(
        case_id=case_id,
        score=score,
        passed=score >= PASS_THRESHOLD,
        reasoning=reasoning,
        failure_category=failure_category if not (score >= PASS_THRESHOLD) else None,
    )
