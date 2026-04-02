"""
Integration tests for the Elasticsearch client.
Requires a running Elasticsearch instance.

Run with: uv run pytest tests/test_elasticsearch_client.py -v -m integration
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.integration

SYNTHETIC_CARDS = [
    {
        "card_id": "test-fire-1",
        "name": "Inferno Dragon",
        "supertype": "Pokémon",
        "subtypes": ["Stage 2"],
        "types": ["Fire"],
        "hp": 300,
        "stage": "Stage 2",
        "rarity": "Rare",
        "set_name": "Test Set",
        "set_id": "test",
        "artist": None,
        "flavor_text": "Burns everything in its path.",
        "attacks_text": "Flame Thrower 200: Discard 2 fire energy",
        "abilities_text": "Blaze: Boost fire attack damage",
        "full_text": "Inferno Dragon is a Stage 2 Fire Pokemon card. HP: 300. Attack 'Flame Thrower' deals 200 damage: Discard 2 fire energy. Ability 'Blaze': Boost fire attack damage.",
        "image_small": None,
        "image_large": None,
        "weaknesses": ["Water"],
        "resistances": [],
        "national_pokedex_numbers": [],
        "embedding": [0.0] * 1536,
    },
    {
        "card_id": "test-water-1",
        "name": "Aqua Serpent",
        "supertype": "Pokémon",
        "subtypes": ["Basic"],
        "types": ["Water"],
        "hp": 120,
        "stage": "Basic",
        "rarity": "Common",
        "set_name": "Test Set",
        "set_id": "test",
        "artist": None,
        "flavor_text": "Heals itself using the power of water.",
        "attacks_text": "Bubble 30: Flip a coin, if heads opponent is paralyzed",
        "abilities_text": "Water Veil: Once per turn, heal 30 HP from this Pokemon",
        "full_text": "Aqua Serpent is a Basic Water Pokemon card. HP: 120. Attack 'Bubble' deals 30 damage: Flip a coin, if heads opponent is paralyzed. Ability 'Water Veil': Once per turn, heal 30 HP from this Pokemon.",
        "image_small": None,
        "image_large": None,
        "weaknesses": ["Electric"],
        "resistances": [],
        "national_pokedex_numbers": [],
        "embedding": [0.0] * 1536,
    },
]


@pytest.fixture(scope="module")
async def es_test_index():
    """Create a test index, yield, then clean up."""
    from elasticsearch import AsyncElasticsearch
    from ingest.index_cards import INDEX_MAPPING

    ES_URL = "http://localhost:9200"
    TEST_INDEX = "test_pokemon_cards"

    es = AsyncElasticsearch(ES_URL)
    try:
        # Ensure ES is up
        await es.ping()
    except Exception:
        pytest.skip("Elasticsearch not available at localhost:9200")

    # Create index
    if await es.indices.exists(index=TEST_INDEX):
        await es.indices.delete(index=TEST_INDEX)

    mapping = {**INDEX_MAPPING}
    await es.indices.create(index=TEST_INDEX, body=mapping)

    # Index synthetic cards
    for card in SYNTHETIC_CARDS:
        await es.index(index=TEST_INDEX, id=card["card_id"], document=card)
    await es.indices.refresh(index=TEST_INDEX)

    yield TEST_INDEX

    await es.indices.delete(index=TEST_INDEX)
    await es.close()


@pytest.mark.asyncio
async def test_hybrid_search_returns_results(es_test_index):
    from app.search.elasticsearch_client import hybrid_search

    # Use zero vector — tests that the query runs without errors
    zero_vec = [0.0] * 1536
    results = await hybrid_search("fire pokemon", zero_vec, es_test_index, top_k=5)
    assert isinstance(results, list)
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_hybrid_search_excludes_embedding_field(es_test_index):
    from app.search.elasticsearch_client import hybrid_search

    zero_vec = [0.0] * 1536
    results = await hybrid_search("dragon", zero_vec, es_test_index, top_k=5)
    for doc in results:
        assert "embedding" not in doc, "embedding field should be excluded from results"


@pytest.mark.asyncio
async def test_hybrid_search_returns_card_id(es_test_index):
    from app.search.elasticsearch_client import hybrid_search

    zero_vec = [0.0] * 1536
    results = await hybrid_search("fire", zero_vec, es_test_index, top_k=5)
    for doc in results:
        assert "card_id" in doc
        assert "name" in doc
