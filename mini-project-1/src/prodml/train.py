import json
import logging
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import TARGET, fit_vectorizer, to_dicts
from prodml.logging_conf import setup_logging

logger = logging.getLogger(__name__)


def save_model(dv: DictVectorizer, model: LinearRegression, path: str) -> None:
    """Persist the vectorizer and the model together in one pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f_out:
        pickle.dump((dv, model), f_out)


def save_metadata(metrics: dict[str, float], path: str) -> None:
    """Persist training metadata (timestamp and metrics) alongside the model."""
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f_out:
        json.dump(metadata, f_out, indent=2)


def train() -> dict[str, float]:
    """Train the model, evaluate on the validation set, save it."""
    train_df, val_df = split_data(load_data())

    dv, X_train = fit_vectorizer(to_dicts(train_df))
    X_val = dv.transform(to_dicts(val_df))

    model = LinearRegression()
    model.fit(X_train, train_df[TARGET])

    y_pred = model.predict(X_val)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(val_df[TARGET], y_pred))),
        "mae": float(mean_absolute_error(val_df[TARGET], y_pred)),
    }

    save_model(dv, model, settings.model_path)
    save_metadata(metrics, settings.metadata_path)
    return metrics


if __name__ == "__main__":
    setup_logging()
    metrics = train()
    logger.info("training finished", extra=metrics)
