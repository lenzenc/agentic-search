.PHONY: dev ingest ingest-fetch ingest-embed ingest-index test test-integration docker-up docker-down eval improve

CASE_FILTER := $(filter q%,$(MAKECMDGOALS))

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

eval:
	@echo "Running evaluation against localhost:8000 (server must be running)..."
	uv run python -m eval.run_eval $(if $(CASE_FILTER),--case $(CASE_FILTER),)

q%: ;


improve:
	@echo "Running prompt improvement loop (server must be running with 'make dev')..."
	uv run python -m eval.improve
