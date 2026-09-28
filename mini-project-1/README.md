# Mini MLOps Project 1: NYC Taxi Ride Duration

## Workflow

```bash
# install
uv sync --extra dev

# lint
uv run ruff check src tests && uv run black --check src tests

# test
uv run pytest -v --cov=src/prodml --cov-report=term-missing

# train
uv run python -m prodml.train

# serve (added in a later step)
uv run uvicorn prodml.api.main:app --reload --port 8000
```

## Configuration

Settings are read from environment variables or a `.env` file:
`DATA_PATH`, `MODEL_PATH`, `PORT`.
