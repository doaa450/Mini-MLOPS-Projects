import pickle
from typing import Any

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import settings
from prodml.features import to_dicts
from prodml.utils import timed


class DurationPredictor:
    """Load a trained model and predict trip duration in minutes."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or settings.model_path
        self._dv: DictVectorizer | None = None
        self._model: LinearRegression | None = None

    def load(self) -> DurationPredictor:
        with open(self.model_path, "rb") as f_in:
            self._dv, self._model = pickle.load(f_in)
        return self

    def predict_batch(self, records: list[dict[str, Any]]) -> list[float]:
        if self._dv is None or self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        features = self._dv.transform(to_dicts(pd.DataFrame(records)))
        return [float(p) for p in self._model.predict(features)]

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        return self.predict_batch([features])[0]
