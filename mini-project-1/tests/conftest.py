import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from prodml.api.main import app
from prodml.data import load_data, split_data
from prodml.features import TARGET, fit_vectorizer, to_dicts


@pytest.fixture
def sample_features() -> dict:
    """A single valid ride's raw features, before PU_DO is computed."""
    return {"PULocationID": 82, "DOLocationID": 129, "trip_distance": 2.5}


@pytest.fixture(scope="session")
def trained_model():
    """Train once per test session, reuse across every test that needs a model."""
    train_df, _ = split_data(load_data())
    dv, X_train = fit_vectorizer(to_dicts(train_df))

    model = LinearRegression()
    model.fit(X_train, train_df[TARGET])

    return dv, model


@pytest.fixture
def client():
    """A FastAPI TestClient for hitting the API endpoints in tests."""
    with TestClient(app) as test_client:
        yield test_client
