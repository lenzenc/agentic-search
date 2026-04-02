"""
Fetch all Pokemon cards from the pokemontcg.io API and save to data/raw_cards.ndjson.

Usage: uv run python -m ingest.fetch_cards
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

console = Console()

API_BASE = "https://api.pokemontcg.io/v2"
PAGE_SIZE = 250
OUTPUT_FILE = Path("data/raw_cards.ndjson")


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
)
async def fetch_page(client: httpx.AsyncClient, page: int) -> dict:
    params = {"pageSize": PAGE_SIZE, "page": page}
    headers = {}
    api_key = os.getenv("POKEMON_TCG_API_KEY", "")
    if api_key:
        headers["X-Api-Key"] = api_key

    resp = await client.get(f"{API_BASE}/cards", params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def main() -> None:
    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    console.print("[bold cyan]Fetching Pokemon cards from pokemontcg.io...[/bold cyan]")

    # Get first page to determine total count
    async with httpx.AsyncClient() as client:
        first = await fetch_page(client, 1)

    total_count = first.get("totalCount", 0)
    total_pages = (total_count + PAGE_SIZE - 1) // PAGE_SIZE
    console.print(f"Total cards: {total_count} across {total_pages} pages")

    all_cards: list[dict] = []
    all_cards.extend(first.get("data", []))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching pages", total=total_pages)
        progress.advance(task)  # page 1 already fetched

        async with httpx.AsyncClient() as client:
            for page in range(2, total_pages + 1):
                data = await fetch_page(client, page)
                all_cards.extend(data.get("data", []))
                progress.advance(task)

    console.print(f"[green]Fetched {len(all_cards)} cards. Writing to {OUTPUT_FILE}...[/green]")

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for card in all_cards:
            f.write(json.dumps(card) + "\n")

    console.print(f"[bold green]Done. {OUTPUT_FILE} written ({OUTPUT_FILE.stat().st_size // 1024} KB)[/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
