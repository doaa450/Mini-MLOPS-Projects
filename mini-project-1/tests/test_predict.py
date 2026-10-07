import pandas as pd

from prodml.features import to_dicts


def test_predict_one_returns_float(sample_features, trained_model):
    """A prediction from the trained model should be a plain float."""
    dv, model = trained_model
    features = dv.transform(to_dicts(pd.DataFrame([sample_features])))
    result = float(model.predict(features)[0])
    assert isinstance(result, float)


def test_predict_one_in_sane_range(sample_features, trained_model):
    """A normal short ride should predict a duration between 0 and 120 minutes."""
    dv, model = trained_model
    features = dv.transform(to_dicts(pd.DataFrame([sample_features])))
    result = float(model.predict(features)[0])
    assert 0 < result < 120


def test_predict_one_is_deterministic(sample_features, trained_model):
    """The same input should always produce the same prediction."""
    dv, model = trained_model
    features = dv.transform(to_dicts(pd.DataFrame([sample_features])))
    result1 = float(model.predict(features)[0])
    result2 = float(model.predict(features)[0])
    assert result1 == result2
