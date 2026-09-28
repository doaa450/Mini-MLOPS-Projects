from typing import Any

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction import DictVectorizer

FEATURE_COLUMNS = ["PU_DO", "trip_distance"]
TARGET = "duration"


def add_pu_do(df: pd.DataFrame) -> pd.DataFrame:
    """Create the pickup-dropoff pair feature."""
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


def to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn a DataFrame into the list of dicts DictVectorizer expects."""
    df = add_pu_do(df)
    return df[FEATURE_COLUMNS].to_dict(orient="records")


def fit_vectorizer(
    train_dicts: list[dict[str, Any]],
) -> tuple[DictVectorizer, spmatrix]:
    """Fit a DictVectorizer on the training data and return it with the matrix."""
    dv = DictVectorizer()
    matrix = dv.fit_transform(train_dicts)
    return dv, matrix
