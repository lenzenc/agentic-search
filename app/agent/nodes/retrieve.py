"""
retrieve node — runs hybrid ES search for active_query (or all sub_queries if complex).
"""
import os
from datetime import datetime, timezone

from app.agent.state import AgentState, SubQuery, TrajectoryStep
from app.search.elasticsearch_client import search_for_query

INDEX_NAME = os.getenv("ES_INDEX_NAME", "pokemon_cards")
TOP_K = int(os.getenv("TOP_K_RETRIEVE", "100"))
SCORE_THRESHOLD_RATIO = float(os.getenv("SCORE_THRESHOLD_RATIO", "0.4"))


async def retrieve(state: AgentState) -> dict:
    is_complex = state.get("is_complex", False)
    iteration = state.get("iteration_count", 0) + 1
    sub_queries: list[SubQuery] = state.get("sub_queries", [])

    seen_ids: set[str] = set()
    merged_docs: list[dict] = []

    if is_complex and sub_queries and iteration == 1:
        # Fan-out: retrieve for each sub-query, merge unique results
        updated_sub_queries: list[SubQuery] = []
        for sq in sub_queries:
            results = await search_for_query(sq["query"], INDEX_NAME, TOP_K)
            retrieved_ids = []
            for doc in results:
                card_id = doc.get("card_id", "")
                if card_id not in seen_ids:
                    seen_ids.add(card_id)
                    merged_docs.append(doc)
                    retrieved_ids.append(card_id)
            updated_sub_queries.append({"query": sq["query"], "retrieved_ids": retrieved_ids})

        query_desc = f"{len(sub_queries)} sub-queries"
        meta_queries = [sq["query"] for sq in sub_queries]
    else:
        # Simple or rewrite loop: use active_query
        active_query = state.get("active_query", state.get("original_query", ""))
        results = await search_for_query(active_query, INDEX_NAME, TOP_K)
        for doc in results:
            card_id = doc.get("card_id", "")
            if card_id not in seen_ids:
                seen_ids.add(card_id)
                merged_docs.append(doc)

        updated_sub_queries = sub_queries  # unchanged
        meta_queries = [active_query]

    # Apply relative score threshold: keep docs scoring >= max_score * ratio
    if merged_docs:
        max_score = max(d.get("_score", 0) for d in merged_docs)
        cutoff = max_score * SCORE_THRESHOLD_RATIO
        merged_docs = [d for d in merged_docs if d.get("_score", 0) >= cutoff]

    if is_complex and sub_queries and iteration == 1:
        detail = (
            f"Query expansion retrieval — {query_desc} → {len(merged_docs)} unique cards "
            f"via hybrid kNN+BM25 (score threshold {SCORE_THRESHOLD_RATIO})"
        )
    else:
        detail = (
            f"Retrieved {len(merged_docs)} cards via hybrid kNN+BM25 "
            f"(iteration {iteration}/{os.getenv('MAX_RETRIEVE_ITERATIONS', '3')}, "
            f"score threshold {SCORE_THRESHOLD_RATIO})"
        )

    top_names = [d.get("name", "") for d in merged_docs[:5]]

    step: TrajectoryStep = {
        "node": "retrieve",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
        "metadata": {
            "queries": meta_queries,
            "result_count": len(merged_docs),
            "top_cards": top_names,
            "iteration": iteration,
            "method": "hybrid kNN + BM25",
        },
    }

    return {
        "retrieved_docs": merged_docs,
        "iteration_count": iteration,
        "sub_queries": updated_sub_queries,
        "trajectory": [step],
    }
