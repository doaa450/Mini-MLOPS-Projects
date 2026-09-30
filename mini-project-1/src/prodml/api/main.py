import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.config import settings
from prodml.logging_conf import correlation_id_var, setup_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger(__name__)

model_state: dict = {}


def compute_hash(path: str) -> str:
    """Compute the SHA-256 hash of a file, used to identify the exact model artifact."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts, release it when it stops."""
    setup_logging()
    logger.info("Loading model", extra={"model_path": settings.onnx_path})

    predictor = DurationPredictor(model_path=settings.model_path).load()
    model_state["predictor"] = predictor
    model_state["artifact_hash"] = compute_hash(settings.model_path)
    model_state["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(settings.metadata_path) as f_in:  # noqa: ASYNC230
            model_state["training_metadata"] = json.load(f_in)
    except FileNotFoundError:
        model_state["training_metadata"] = {}

    yield

    model_state.clear()


app = FastAPI(
    title="NYC Green Taxi Duration Prediction API",
    version=settings.model_version,
    description="API for predicting NYC Green Taxi trip duration.",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Generate a correlation ID for every request, attach it, return it as a header."""
    correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)

    logger.info(
        "request received",
        extra={"method": request.method, "path": request.url.path},
    )

    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response


@app.get("/health")
async def health() -> dict:
    """Return 200 only if the model is actually loaded in memory."""
    if "predictor" not in model_state:
        return {"status": "unhealthy", "model_loaded": False}
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata")
async def metadata() -> dict:
    """Return model version, training date, feature names, framework, artifact hash."""
    training_meta = model_state.get("training_metadata", {})
    return {
        "model_version": settings.model_version,
        "trained_at": training_meta.get("trained_at"),
        "training_mae": training_meta.get("mae"),
        "training_rmse": training_meta.get("rmse"),
        "feature_names": ["PU_DO", "trip_distance"],
        "framework": "onnx",
        "artifact_hash": model_state.get("artifact_hash"),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict trip duration for a single ride."""
    predictor: DurationPredictor = model_state["predictor"]

    start = time.perf_counter()
    duration = predictor.predict_one(payload.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        prediction=duration,
        model_version=settings.model_version,
        correlation_id=correlation_id_var.get(),
        latency_ms=round(latency_ms, 2),
    )
