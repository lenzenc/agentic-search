import os
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

_es_client: AsyncElasticsearch | None = None
_openai_client: AsyncOpenAI | None = None


def get_es_client() -> AsyncElasticsearch:
    global _es_client
    if _es_client is None:
        url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        _es_client = AsyncElasticsearch(url)
    return _es_client


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def close_clients() -> None:
    global _es_client
    if _es_client:
        await _es_client.close()
        _es_client = None


async def embed_query(query: str) -> list[float]:
    client = get_openai_client()
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    return response.data[0].embedding


async def hybrid_search(
    query_text: str,
    query_vector: list[float],
    index_name: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Hybrid kNN + BM25 search using ES 8.x combined knn+query request.
    ES merges the two result sets by summing normalised scores.
    Returns a list of _source dicts (embedding excluded).
    """
    es = get_es_client()

    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "num_candidates": 50,
            "k": top_k,
            "boost": 0.7,
        },
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "name^3",
                    "attacks_text^2",
                    "abilities_text^2",
                    "collector_number^3",
                    "full_text",
                ],
                "type": "best_fields",
                "boost": 0.3,
            }
        },
        "size": top_k,
        "_source": {"excludes": ["embedding"]},
    }

    response = await es.search(index=index_name, body=body)
    hits = response["hits"]["hits"]
    return [hit["_source"] for hit in hits]


async def search_for_query(
    query: str,
    index_name: str,
    top_k: int = 10,
) -> list[dict]:
    """Convenience wrapper: embed query then run hybrid search."""
    vector = await embed_query(query)
    return await hybrid_search(query, vector, index_name, top_k)
