# Agentic RAG — Pokemon Card Search Demo

A demo web application that makes the shift from **Naive RAG** to **Agentic RAG** visible in real time, using Pokemon cards as the search domain.

---

## The Shift: Naive RAG → Agentic RAG

**Naive RAG** is a stateless, one-shot pipeline. A query goes in, a retrieval happens, and an answer comes out — with no feedback loop and no awareness of whether the retrieved documents were actually relevant.

```
Naive RAG:    Query → [Retrieve] → [Generate] → Answer
```

**Agentic RAG** is a stateful, iterative runtime. An LLM orchestrates a Reason → Act → Observe cycle. Retrieval is one tool among many. The system detects failures and self-corrects.

```
Agentic RAG:  Query → [Analyze] → [Retrieve] → [Grade]
                                      ↑              |
                                      └── [Rewrite] ←┘  (if not relevant)
                                                    ↓    (if relevant enough)
                                              [Generate] → Answer
```

This demo implements two core Agentic RAG patterns using [LangGraph](https://github.com/langchain-ai/langgraph) to build the cyclic state graph:

### Pattern 1 — Self-Correcting Loop

> "Find a water-type Pokemon that can heal itself"

The agent retrieves cards, then grades each one for relevance using Claude. If too few relevant cards are found, it rewrites the search query based on the grading feedback and tries again — up to 3 times. This surfaces attack and ability text that simple keyword or vector search would miss on the first pass.

### Pattern 2 — Iterative Query Expansion

> "Electric Pokemon that can paralyze or confuse the opponent"

Claude detects that this query has two independent facets — "electric types" and "status effect attacks" — and decomposes it into targeted sub-searches. Each sub-search is run independently, results are merged and deduplicated, then graded together.

---

## Demo Scenarios

Five pre-built scenarios are shown on the home screen. Click any to run it immediately.

### Baseline — Direct Search
**Query:** `Charizard`

The simplest case. One intent, one retrieval pass, one answer. Use this to contrast against the two agentic patterns — the trace will show a clean linear path with no loop or expansion badges.

**Watch for:** Analyze → Retrieve → Grade → Generate, no rewrite step.

---

### Pattern 1 — Self-Correcting Loop
**Query:** `Find a water-type Pokemon that can heal itself`

The first retrieval returns mostly generic water types with no healing mechanics. Claude grades them, finds too few relevant cards, rewrites the query toward ability text like "restore HP" or "recover damage," and retries.

**Watch for:** A red **↺ loop N of 3** badge on the Rewrite Query step. The Grade step on iteration 1 will show mostly ✗ not-relevant verdicts with reasonings like "no healing ability found."

---

### Pattern 1 Variant — Loop with Energy Abilities
**Query:** `Grass type Pokemon with an ability that powers up or restores energy`

"Powers up attacks" is a semantic concept that simple keyword or vector search misses on the first pass. The agent iterates until it finds ability text describing energy acceleration.

**Watch for:** The rewrite step shifting vocabulary toward "attach energy," "energy acceleration," or "search deck for energy."

---

### Pattern 2 — Iterative Query Expansion
**Query:** `Electric Pokemon that can paralyze or confuse the opponent`

Two independent facets: "electric type" and "status effect attacks." Claude decomposes this into separate sub-searches, retrieves independently, and merges results before grading.

**Watch for:** A blue **⇥ expanded to N sub-queries** badge on the Analyze Query step, with the sub-queries listed below it (e.g., "electric type pokemon cards" and "pokemon attack causes paralysis confusion").

---

### Pattern 2 Variant — Expansion with Disruption Deck
**Query:** `Dark or Psychic Pokemon that can discard the opponent's cards or energy`

A wide multi-facet query spanning two types and two disruption mechanics. The agent generates 3–4 sub-searches and fans out across all of them.

**Watch for:** A higher card count in the Retrieve step (fan-out across sub-queries), and 3–4 sub-queries listed in the Analyze step.

---

## What to Watch in the UI

The **Agent Execution Trace** panel (left side) is the main feature of this demo. It streams the agent's reasoning in real time as each node fires:

| Node | What it shows |
|---|---|
| 🔍 **Analyze Query** | Whether the query was classified as simple or complex; sub-queries if expanded |
| 📡 **Retrieve** | Which query was searched, how many cards were returned, the top card names |
| ⚖️ **Grade Documents** | Each card's relevance verdict with Claude's one-sentence reasoning |
| ↺ **Rewrite Query** | The old query, why it failed, and the new query (loop badge shows "loop N of 3") |
| ✨ **Generate Answer** | How many relevant cards were used to write the final answer |

Click any step to expand the full detail. Toggle **Show raw log** for a terminal-style execution log with timestamps.

The **pattern badge** in the header updates after the search to show which pattern fired: `↺ Self-Correcting Loop`, `⇥ Query Expansion`, or `Direct`.

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Elasticsearch)
- [uv](https://docs.astral.sh/uv/) — Python package manager
- API keys: Anthropic (Claude) and OpenAI (embeddings)
- Optional: [Pokemon TCG API key](https://dev.pokemontcg.io/) — raises the rate limit from 1,000 requests/day to unlimited

---

## Setup

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start Elasticsearch

```bash
make docker-up
```

Elasticsearch starts on `http://localhost:9200`. Data is persisted in a Docker volume across restarts.

### 4. Run the ingest pipeline (one-time, ~10–15 min)

This fetches ~15,000 Pokemon cards from the API, generates OpenAI embeddings, and indexes everything into Elasticsearch.

```bash
make ingest
```

Each step writes intermediate output to `data/` — if the pipeline is interrupted, re-running resumes from where it stopped.

You can also run steps individually:
```bash
make ingest-fetch    # fetch cards from pokemontcg.io → data/raw_cards.ndjson
make ingest-embed    # generate embeddings → data/embedded_cards.ndjson
make ingest-index    # create ES index + bulk upsert
```

### 5. Start the app

```bash
make dev
```

Open [http://localhost:8000](http://localhost:8000).

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key (claude.ai) |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for `text-embedding-3-small` |
| `POKEMON_TCG_API_KEY` | No | — | Raises rate limit on pokemontcg.io |
| `ELASTICSEARCH_URL` | No | `http://localhost:9200` | ES connection URL |
| `ES_INDEX_NAME` | No | `pokemon_cards` | Index name |
| `MAX_RETRIEVE_ITERATIONS` | No | `3` | Max self-correction loops |
| `MIN_RELEVANT_DOCS` | No | `3` | Relevant docs needed to stop looping |
| `TOP_K_RETRIEVE` | No | `10` | Cards retrieved per search |

---

## Running Tests

```bash
make test              # unit tests (no ES or API keys required)
make test-integration  # includes ES integration tests (requires running ES)
```

---

## Architecture

| Layer | Technology |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) — cyclic state graph |
| LLM (reasoning, grading, generation) | Claude `claude-sonnet-4-6` via Anthropic SDK |
| LLM (grading, cost-optimized) | Claude `claude-haiku-4-5` |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector + keyword search | Elasticsearch 8.x — hybrid kNN + BM25 via native RRF retriever |
| Streaming | FastAPI + Server-Sent Events |
| Frontend | Vanilla HTML/CSS/JS — no build step |

Search uses Elasticsearch's native [Reciprocal Rank Fusion (RRF)](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) to combine vector similarity and BM25 keyword scores — avoiding the need for manually tuned score weights.
