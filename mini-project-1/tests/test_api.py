from unittest.mock import patch


def test_health_returns_200(client):
    """GET /health should return 200 with the model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_happy_path(client, sample_features):
    """POST /predict with valid data should return a successful prediction."""
    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["prediction"], float)
    assert "correlation_id" in body
    assert "latency_ms" in body


def test_predict_invalid_payload_returns_422(client):
    """POST /predict with an invalid trip_distance should return 422."""
    response = client.post(
        "/predict",
        json={"PULocationID": 82, "DOLocationID": 129, "trip_distance": -5},
    )
    assert response.status_code == 422
    assert "errors" in response.json()


def test_predict_response_schema(client, sample_features):
    """The /predict response must match PredictionResponse exactly."""
    response = client.post("/predict", json=sample_features)
    body = response.json()

    expected_keys = {"prediction", "model_version", "correlation_id", "latency_ms"}
    assert set(body.keys()) == expected_keys


def test_predict_with_mocked_model(client, sample_features):
    """The /predict endpoint should work even with a mocked predictor (no real training needed)."""
    with patch(
        "prodml.api.main.model_state",
        {
            "predictor": __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(
                predict_one=lambda features: 10.0
            )
        },
    ):
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200
        assert response.json()["prediction"] == 10.0
