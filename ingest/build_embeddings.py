"""
Build text blobs for each card and generate OpenAI embeddings.
Reads data/raw_cards.ndjson, writes data/embedded_cards.ndjson.

Usage: uv run python -m ingest.build_embeddings
"""
import asyncio
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

console = Console()

INPUT_FILE = Path("data/raw_cards.ndjson")
OUTPUT_FILE = Path("data/embedded_cards.ndjson")
BATCH_SIZE = 100
EMBEDDING_MODEL = "text-embedding-3-small"


def build_text_blob(card: dict) -> str:
    """Build a rich text representation of a card for embedding."""
    parts = []

    name = card.get("name", "Unknown")
    supertype = card.get("supertype", "")
    subtypes = card.get("subtypes", [])
    types = card.get("types", [])
    hp = card.get("hp")
    rarity = card.get("rarity", "")
    set_info = card.get("set", {})
    set_name = set_info.get("name", "")

    # Stage detection from subtypes
    stage = None
    for sub in subtypes:
        if sub in ("Basic", "Stage 1", "Stage 2", "VMAX", "VSTAR", "ex", "EX", "GX", "V"):
            stage = sub
            break

    number = card.get("number", "")
    printed_total = set_info.get("printedTotal", "")
    number_str = f" Card number: {number}/{printed_total}." if number and printed_total else (f" Card number: {number}." if number else "")

    type_str = "/".join(types) if types else supertype
    stage_str = f"{stage} " if stage else ""
    hp_str = f" HP: {hp}." if hp else ""
    rarity_str = f" Rarity: {rarity}." if rarity else ""
    set_str = f" Set: {set_name}." if set_name else ""

    parts.append(f"{name} is a {stage_str}{type_str} Pokemon card.{hp_str}{rarity_str}{set_str}{number_str}")

    # Attacks
    attacks = card.get("attacks", [])
    attack_texts = []
    for atk in attacks:
        atk_name = atk.get("name", "")
        damage = atk.get("damage", "")
        text = atk.get("text", "")
        cost = "/".join(atk.get("cost", []))
        atk_str = f"Attack '{atk_name}'"
        if cost:
            atk_str += f" (cost: {cost})"
        if damage:
            atk_str += f" deals {damage} damage"
        if text:
            atk_str += f": {text}"
        atk_str += "."
        attack_texts.append(atk_str)
    if attack_texts:
        parts.append(" ".join(attack_texts))

    # Abilities
    abilities = card.get("abilities", [])
    ability_texts = []
    for ab in abilities:
        ab_name = ab.get("name", "")
        ab_type = ab.get("type", "Ability")
        ab_text = ab.get("text", "")
        if ab_text:
            ability_texts.append(f"{ab_type} '{ab_name}': {ab_text}.")
    if ability_texts:
        parts.append(" ".join(ability_texts))

    # Flavor text
    flavor = card.get("flavorText", "")
    if flavor:
        parts.append(f"Flavor text: {flavor}")

    # Weaknesses / resistances
    weaknesses = [w.get("type", "") for w in card.get("weaknesses", []) if w.get("type")]
    resistances = [r.get("type", "") for r in card.get("resistances", []) if r.get("type")]
    if weaknesses:
        parts.append(f"Weaknesses: {', '.join(weaknesses)}.")
    if resistances:
        parts.append(f"Resistances: {', '.join(resistances)}.")

    return " ".join(parts)


def card_to_document(card: dict) -> dict:
    """Convert raw API card dict to our ES document shape (without embedding)."""
    set_info = card.get("set", {})
    subtypes = card.get("subtypes", [])

    stage = None
    for sub in subtypes:
        if sub in ("Basic", "Stage 1", "Stage 2", "VMAX", "VSTAR", "ex", "EX", "GX", "V", "BREAK"):
            stage = sub
            break

    attacks = card.get("attacks", [])
    attacks_parts = []
    for atk in attacks:
        name = atk.get("name", "")
        damage = atk.get("damage", "")
        text = atk.get("text", "")
        piece = name
        if damage:
            piece += f" {damage}"
        if text:
            piece += f": {text}"
        attacks_parts.append(piece)
    attacks_text = " | ".join(attacks_parts)

    abilities = card.get("abilities", [])
    abilities_parts = []
    for ab in abilities:
        name = ab.get("name", "")
        text = ab.get("text", "")
        ab_type = ab.get("type", "Ability")
        if text:
            abilities_parts.append(f"{ab_type} {name}: {text}")
        else:
            abilities_parts.append(f"{ab_type} {name}")
    abilities_text = " | ".join(abilities_parts)

    images = card.get("images", {})

    hp_raw = card.get("hp")
    try:
        hp = int(hp_raw) if hp_raw else None
    except (ValueError, TypeError):
        hp = None

    weaknesses = [w.get("type", "") for w in card.get("weaknesses", []) if w.get("type")]
    resistances = [r.get("type", "") for r in card.get("resistances", []) if r.get("type")]
    pokedex = card.get("nationalPokedexNumbers", [])

    full_text = build_text_blob(card)

    return {
        "card_id": card.get("id", ""),
        "collector_number": card.get("number", ""),
        "name": card.get("name", ""),
        "supertype": card.get("supertype", ""),
        "subtypes": subtypes,
        "types": card.get("types", []),
        "hp": hp,
        "stage": stage,
        "rarity": card.get("rarity"),
        "set_name": set_info.get("name", ""),
        "set_id": set_info.get("id", ""),
        "artist": card.get("artist"),
        "flavor_text": card.get("flavorText"),
        "attacks_text": attacks_text,
        "abilities_text": abilities_text,
        "full_text": full_text,
        "image_small": images.get("small"),
        "image_large": images.get("large"),
        "weaknesses": weaknesses,
        "resistances": resistances,
        "national_pokedex_numbers": pokedex,
    }


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
async def embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


async def main() -> None:
    if not INPUT_FILE.exists():
        console.print(f"[red]Input file {INPUT_FILE} not found. Run `make ingest-fetch` first.[/red]")
        return

    console.print(f"[bold cyan]Building embeddings from {INPUT_FILE}...[/bold cyan]")

    # Load all cards
    cards_raw = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards_raw.append(json.loads(line))

    console.print(f"Loaded {len(cards_raw)} cards")

    # Load already-embedded card IDs to support resume
    embedded_ids: set[str] = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    embedded_ids.add(doc.get("card_id", ""))
        console.print(f"Resuming — {len(embedded_ids)} cards already embedded")

    # Filter to only cards needing embedding
    pending = [c for c in cards_raw if c.get("id", "") not in embedded_ids]
    console.print(f"Cards to embed: {len(pending)}")

    if not pending:
        console.print("[green]All cards already embedded.[/green]")
        return

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with (
        OUTPUT_FILE.open("a", encoding="utf-8") as out_f,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("Embedding batches", total=len(pending))

        for i in range(0, len(pending), BATCH_SIZE):
            batch = pending[i : i + BATCH_SIZE]
            docs = [card_to_document(card) for card in batch]
            texts = [doc["full_text"] for doc in docs]

            embeddings = await embed_batch(client, texts)

            for doc, embedding in zip(docs, embeddings):
                doc["embedding"] = embedding
                out_f.write(json.dumps(doc) + "\n")

            progress.advance(task, len(batch))

    console.print(f"[bold green]Done. Embeddings written to {OUTPUT_FILE}[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
