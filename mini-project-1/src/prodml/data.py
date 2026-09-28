import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the parquet file, compute duration (minutes) and drop outliers."""
    df = pd.read_parquet(path or settings.data_path)

    df["duration"] = df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    df["duration"] = df["duration"].dt.total_seconds() / 60

    df = df[
        (df["duration"] >= settings.min_duration)
        & (df["duration"] <= settings.max_duration)
    ]
    df = df[
        (df["trip_distance"] > settings.min_distance)
        & (df["trip_distance"] <= settings.max_distance)
    ]
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and validation sets."""
    train_df, val_df = train_test_split(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    return train_df, val_df
