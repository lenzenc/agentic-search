# Load .env before any other imports so os.getenv() calls in nodes get correct values
from dotenv import load_dotenv
load_dotenv(override=True)

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent.graph import run_agent
from app.models.api import SearchRequest, SearchResponse, TrajectoryStep
from app.models.card import CardResult
from app.search.elasticsearch_client import get_es_client, close_clients


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify ES connection on startup
    es = get_es_client()
    try:
        await es.ping()
    except Exception as e:
        print(f"WARNING: Could not connect to Elasticsearch: {e}")
    yield
    await close_clients()


app = FastAPI(title="Agentic RAG — Pokemon Card Search", lifespan=lifespan)


def detect_pattern(trajectory: list[dict]) -> str:
    """Detect which agentic pattern fired based on trajectory nodes."""
    nodes = [step["node"] for step in trajectory]
    if "rewrite_query" in nodes:
        return "self_correcting_loop"
    analyze_steps = [s for s in trajectory if s["node"] == "analyze_query"]
    if analyze_steps and analyze_steps[0]["metadata"].get("is_complex"):
        return "query_expansion"
    return "direct"


async def search_stream(query: str):
    """Generator that streams SSE events from agent execution.

    run_agent() yields the full accumulated state after each node completes.
    We track the trajectory length to emit only new steps as SSE events.
    """
    state: dict = {
        "original_query": query,
        "is_complex": False,
        "sub_queries": [],
        "active_query": query,
        "retrieved_docs": [],
        "iteration_count": 0,
        "graded_docs": [],
        "relevant_docs": [],
        "final_answer": "",
        "final_cards": [],
        "trajectory": [],
    }

    prev_trajectory_len = 0
    final_state: dict | None = None

    try:
        async for state in run_agent(state):
            final_state = state
            new_steps = state["trajectory"][prev_trajectory_len:]
            prev_trajectory_len = len(state["trajectory"])
            for step in new_steps:
                yield f"event: trajectory_step\ndata: {json.dumps(step)}\n\n"

        if final_state is None:
            raise RuntimeError("Graph produced no output")

        cards = [CardResult.from_es_hit({"_source": c}).model_dump() for c in (final_state.get("final_cards") or [])]
        trajectory = final_state.get("trajectory", [])
        pattern = detect_pattern(trajectory)

        result_payload = {
            "answer": final_state.get("final_answer", ""),
            "cards": cards,
            "trajectory": trajectory,
            "total_iterations": final_state.get("iteration_count", 1),
            "pattern": pattern,
        }
        yield f"event: result\ndata: {json.dumps(result_payload)}\n\n"

    except Exception as e:
        error_payload = {"error": str(e)}
        yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"

    yield "event: done\ndata: {}\n\n"


@app.post("/api/search")
async def search(request: SearchRequest):
    es = get_es_client()
    try:
        index_name = os.getenv("ES_INDEX_NAME", "pokemon_cards")
        exists = await es.indices.exists(index=index_name)
        if not exists:
            raise HTTPException(
                status_code=503,
                detail="Search index not ready. Run `make ingest` first.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Elasticsearch unavailable: {e}")

    return StreamingResponse(
        search_stream(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/health")
async def health():
    es = get_es_client()
    try:
        await es.ping()
        index_name = os.getenv("ES_INDEX_NAME", "pokemon_cards")
        count = await es.count(index=index_name)
        return {"status": "ok", "card_count": count["count"]}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# Serve frontend — must be last to not shadow API routes
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
