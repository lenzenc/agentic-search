.PHONY: dev ingest ingest-fetch ingest-embed ingest-index test test-integration docker-up docker-down

dev:
	uv run uvicorn app.main:app --reload --port 8000

docker-up:
	docker compose up -d
	@echo "Waiting for Elasticsearch to be ready..."
	@until curl -sf http://localhost:9200/_cluster/health > /dev/null 2>&1; do sleep 2; done
	@echo "Elasticsearch is ready."

docker-down:
	docker compose down

ingest-fetch:
	uv run python -m ingest.fetch_cards

ingest-embed:
	uv run python -m ingest.build_embeddings

ingest-index:
	uv run python -m ingest.index_cards

ingest: ingest-fetch ingest-embed ingest-index

test:
	uv run pytest tests/ -v -m "not integration"

test-integration:
	uv run pytest tests/ -v -m integration
