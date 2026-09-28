from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # Paths
    data_path: str = "data/green_tripdata_2026-02.parquet"
    model_path: str = "models/model.pkl"

    # Serving
    port: int = 8000

    # Outlier filtering (from the notebook cells 9 and 12)
    min_duration: float = 1.0
    max_duration: float = 60.0
    min_distance: float = 0.0
    max_distance: float = 20.0

    # Train/validation split (from cell 17)
    test_size: float = 0.20
    random_state: int = 42

    # Logging
    log_level: str = "INFO"

    distance_warning_threshold: float = 30.0


settings = Settings()
