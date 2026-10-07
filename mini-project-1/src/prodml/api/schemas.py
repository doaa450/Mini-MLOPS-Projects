from pydantic import BaseModel, ConfigDict, Field

from prodml.config import settings


class PredictionRequest(BaseModel):
    """A single ride's features for duration prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PULocationID": 82,
                "DOLocationID": 129,
                "trip_distance": 2.5,
            }
        }
    )

    PULocationID: int = Field(gt=0, description="Pickup zone ID")
    DOLocationID: int = Field(gt=0, description="Dropoff zone ID")
    trip_distance: float = Field(
        gt=0,
        lt=settings.distance_warning_threshold * 2,
        description="Trip distance in miles",
    )


class PredictionResponse(BaseModel):
    """Prediction result with tracing metadata."""

    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    """A batch of rides for duration prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rides": [
                    {"PULocationID": 82, "DOLocationID": 129, "trip_distance": 2.5},
                    {"PULocationID": 226, "DOLocationID": 143, "trip_distance": 5.2},
                ]
            }
        }
    )

    rides: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Batch prediction results with tracing metadata."""

    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float
