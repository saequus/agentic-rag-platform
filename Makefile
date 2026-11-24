.PHONY: dev lint test smoke compose-up compose-down install hooks db-init download ingest bench otel-up otel-down qdrant-up qdrant-down qdrant-sync eval-labels eval

install:
	uv sync --dev
	uv run pre-commit install

hooks:
	uv run pre-commit install

dev:
	uv run fastapi dev src/agentic_rag_platform/main.py --port 8000

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

test:
	uv run pytest

smoke:
	uv run python scripts/smoke.py

compose-up:
	docker compose up --build

compose-down:
	docker compose down

db-init:
	uv run python -m agentic_rag_platform.ingest.migrate

download:
	uv run python scripts/download_corpus.py

ingest: download
	uv run python -m agentic_rag_platform.ingest.cli

bench:
	uv run python scripts/bench.py

otel-up:
	docker compose --profile otel up -d otel-collector jaeger

otel-down:
	docker compose --profile otel stop otel-collector jaeger

# Day 3: hybrid search.
qdrant-up:
	docker compose up -d qdrant

qdrant-down:
	docker compose stop qdrant

# Copy chunks + stored embeddings from Postgres into Qdrant (dense + BM25 sparse).
# Uses QDRANT_URL if set, otherwise embedded mode at QDRANT_PATH (data/qdrant).
qdrant-sync:
	uv run python -m agentic_rag_platform.retrieval.qdrant_sync --recreate

# Re-derive relevance labels after re-chunking (data/eval/seeds.jsonl -> questions.jsonl).
eval-labels:
	uv run python scripts/build_eval_set.py

# recall@5 / MRR@10 for dense, sparse, hybrid, hybrid_rerank on pgvector and Qdrant.
# Skip Qdrant with: make eval EVAL_BACKENDS=pgvector
EVAL_BACKENDS ?= pgvector,qdrant
eval:
	uv run python scripts/eval_retrieval.py --backends $(EVAL_BACKENDS)
