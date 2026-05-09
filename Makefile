install:
	uv sync

test:
	uv run pytest tests/

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
