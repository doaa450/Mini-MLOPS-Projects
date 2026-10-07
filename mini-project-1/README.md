# prodml — NYC Green Taxi Duration Prediction API

A machine learning service that predicts NYC Green Taxi trip duration (in minutes)
from pickup/dropoff zones and trip distance. Built as a packaged Python service with
structured logging, ONNX-based inference, and a FastAPI REST API, fully containerized
for one-command deployment.

## Quickstart (3 commands)

```bash
docker pull doaamohammed/prodml-api:0.1.0
docker run --rm -p 8000:8000 doaamohammed/prodml-api:0.1.0
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"PULocationID": 82, "DOLocationID": 129, "trip_distance": 2.5}'
```

Expected response:
```json
{
  "prediction": 14.95,
  "model_version": "0.1.0",
  "correlation_id": "...",
  "latency_ms": 160.98
}
```

Full interactive docs: http://localhost:8000/docs

## Development workflow

```bash
uv sync --extra dev                                              # install
uv run ruff check src tests && uv run black --check src tests    # lint
uv run pytest -v --cov=src/prodml --cov-report=term-missing      # test
uv run python -m prodml.train                                    # train
uv run uvicorn prodml.api.main:app --reload --port 8000           # serve
```

## Configuration

Settings are read from environment variables or a `.env` file:
`DATA_PATH`, `MODEL_PATH`, `PORT`, `LOG_LEVEL`.

## Repository structure

```
mini-project-1/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── lin_reg.pkl
│   ├── model.onnx
│   └── metadata.json
├── notebooks/
│   └── 00_baseline.ipynb
├── reports/
│   └── module-1.md
├── src/prodml/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── config.py
│   ├── data.py
│   ├── export.py
│   ├── features.py
│   ├── logging_conf.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
├── pyproject.toml
└── README.md
```

## ☑️ Definition of Done

1. ✅ GitHub repo public with more than one commit
2. ✅ Docker Hub token saved securely — logged in via `docker login`, image pushed successfully
3. ✅ `docker run hello-world` works; WSL2 confirmed (Docker Desktop running on WSL2 integration)
4. ⚠️ Package installs via `uv sync --extra dev` in a clean environment (project uses `uv`, not `pip install -e .`, as its package manager — same guarantee, different tool)
5. ✅ Lint passes; pre-commit hooks installed
6. ✅ Tests pass with coverage ≥ 70% (76.57%)
7. ✅ Zero `print()` statements in `src/`
8. ✅ `/health`, `/metadata`, `/predict`, `/predict/batch` all respond correctly
9. ✅ ONNX parity test passes
10. ⏳ Image pushed to Docker Hub and pullable by someone else; container confirmed to run as non-root (`appuser`) — push in progress, interrupted by network timeouts, retrying
11. ✅ `README.md` gets a stranger to a prediction in 3 commands
12. ✅ `reports/module-1.md` has: MAE, latency comparison, image-size comparison, serialization table, maturity self-assessment
13. ⏳ Pull request opened, merged; `v0.1.0` tagged — no peer reviewers available for this solo submission; PR opened and self-reviewed before merge instead of the two-peer review described in the handbook
