"""
Create the Elasticsearch index and bulk-upsert all embedded card documents.
Reads data/embedded_cards.ndjson.

Usage: uv run python -m ingest.index_cards
"""
import asyncio
import json
import os
from pathlib import Path

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

console = Console()

INPUT_FILE = Path("data/embedded_cards.ndjson")
INDEX_NAME = os.getenv("ES_INDEX_NAME", "pokemon_cards")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
BULK_BATCH = 50

INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "pokemon_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "card_id":        {"type": "keyword"},
            "name":           {"type": "text", "analyzer": "pokemon_text", "fields": {"keyword": {"type": "keyword"}}},
            "supertype":      {"type": "keyword"},
            "subtypes":       {"type": "keyword"},
            "types":          {"type": "keyword"},
            "hp":             {"type": "integer"},
            "stage":          {"type": "keyword"},
            "rarity":         {"type": "keyword"},
            "set_name":       {"type": "keyword"},
            "set_id":         {"type": "keyword"},
            "artist":         {"type": "keyword"},
            "flavor_text":    {"type": "text", "analyzer": "pokemon_text"},
            "attacks_text":   {"type": "text", "analyzer": "pokemon_text"},
            "abilities_text": {"type": "text", "analyzer": "pokemon_text"},
            "full_text":      {"type": "text", "analyzer": "pokemon_text"},
            "image_small":    {"type": "keyword", "index": False},
            "image_large":    {"type": "keyword", "index": False},
            "weaknesses":     {"type": "keyword"},
            "resistances":    {"type": "keyword"},
            "collector_number": {"type": "keyword"},
            "national_pokedex_numbers": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
        }
    },
}


def generate_actions(docs: list[dict]):
    for doc in docs:
        yield {
            "_index": INDEX_NAME,
            "_id": doc["card_id"],
            "_source": doc,
        }


async def main() -> None:
    if not INPUT_FILE.exists():
        console.print(f"[red]{INPUT_FILE} not found. Run `make ingest-embed` first.[/red]")
        return

    # Load documents
    docs = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    console.print(f"[bold cyan]Indexing {len(docs)} cards into Elasticsearch ({ES_URL})...[/bold cyan]")

    es = AsyncElasticsearch(ES_URL)

    try:
        # Verify ES is reachable
        info = await es.info()
        console.print(f"Connected to Elasticsearch {info['version']['number']}")

        # Recreate index
        if await es.indices.exists(index=INDEX_NAME):
            console.print(f"Deleting existing index '{INDEX_NAME}'...")
            await es.indices.delete(index=INDEX_NAME)

        console.print(f"Creating index '{INDEX_NAME}'...")
        await es.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)

        # Bulk index
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing documents", total=len(docs))

            success = 0
            errors = 0
            for i in range(0, len(docs), BULK_BATCH):
                batch = docs[i : i + BULK_BATCH]
                ok, errs = await async_bulk(es, generate_actions(batch), raise_on_error=False)
                success += ok
                if isinstance(errs, list):
                    errors += len(errs)
                progress.advance(task, len(batch))

        await es.indices.refresh(index=INDEX_NAME)
        count = await es.count(index=INDEX_NAME)
        console.print(f"[bold green]Done. {count['count']} documents in index '{INDEX_NAME}'. Errors: {errors}[/bold green]")

    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(main())
