# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Environment

This project uses `uv` for all Python tooling.

- Use `uv run` instead of `python` or `python3`
- Use `uv add` instead of `pip install`
- Use `uv remove` instead of `pip uninstall`

## Common Commands

```bash
# Start Elasticsearch (required before running app or ingest)
make docker-up

# Run the ingest pipeline (one-time, ~10-15 min — requires ES running and .env set)
make ingest          # runs all three steps in sequence
make ingest-fetch    # step 1: fetch cards from pokemontcg.io → data/raw_cards.ndjson
make ingest-embed    # step 2: generate embeddings → data/embedded_cards.ndjson
make ingest-index    # step 3: create ES index + bulk upsert

# Start the web app
make dev             # uvicorn with --reload on port 8000

# Run tests
make test            # unit tests only (no ES required)
make test-integration  # includes ES integration tests (requires running ES)

# Run a single test file
uv run pytest tests/test_agent_graph.py -v
```

## Architecture Overview

This is an **Agentic RAG demo** using Pokemon cards as the domain. It demonstrates two advanced RAG patterns — self-correcting loops and iterative query expansion — by making the agent's internal reasoning visible in real-time via a browser UI.

### Key architectural decisions

**LangGraph cyclic graph** (`app/agent/graph.py`) — the core of the demo. The graph has a loop: `retrieve → grade_documents → rewrite_query → retrieve` that repeats up to `MAX_RETRIEVE_ITERATIONS` times if Claude's grader finds too few relevant cards. This is the self-correcting loop pattern.

**Query expansion** (`app/agent/nodes/analyze.py`) — Claude classifies whether a query is "complex" (multi-facet). If so, it decomposes it into N sub-queries. The `retrieve` node fans out across all sub-queries and merges unique results.

**Hybrid search** (`app/search/elasticsearch_client.py`) — uses Elasticsearch 8.9+ native RRF (Reciprocal Rank Fusion) retriever to combine kNN vector search (OpenAI embeddings) with BM25 keyword search. This is better than manual score blending because RRF is scale-invariant.

**SSE streaming** (`app/main.py`) — `POST /api/search` returns `text/event-stream`. LangGraph's `astream_events()` drives the stream. Each node completion emits a `trajectory_step` SSE event; after graph completion a `result` event is sent. The frontend uses `fetch()` with `ReadableStream` (not `EventSource`) because the endpoint is POST.

**Trajectory as append-only state** (`app/agent/state.py`) — `AgentState.trajectory` uses LangGraph's `Annotated[list, operator.add]` reducer. Each node appends its own `TrajectoryStep` dict. This accumulates the full reasoning chain, which the SSE endpoint streams incrementally and the frontend renders as an interactive execution log.

### Data flow

1. `ingest/` scripts fetch cards from pokemontcg.io, build text blobs (card name + attacks + abilities text), embed with OpenAI, and bulk-index into Elasticsearch.
2. User query → FastAPI → LangGraph graph → SSE stream to browser.
3. Frontend shows: trace panel (left, agent internals) + answer + card grid (right).

### Config

All config is in `app/config.py` via `pydantic-settings`. Values come from `.env` (copy `.env.example`). Key thresholds: `MIN_RELEVANT_DOCS=3` (triggers rewrite if fewer relevant cards found), `MAX_RETRIEVE_ITERATIONS=3` (max loop count).
