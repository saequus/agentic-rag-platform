.PHONY: dev lint test smoke compose-up compose-down install hooks

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
