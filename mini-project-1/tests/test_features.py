from prodml.features import add_pu_do, to_dicts


def test_add_pu_do_combines_zones():
    """PU_DO should combine PULocationID and DOLocationID with an underscore."""
    import pandas as pd

    df = pd.DataFrame({"PULocationID": [82], "DOLocationID": [129]})
    result = add_pu_do(df)

    assert result["PU_DO"].iloc[0] == "82_129"


import pandas as pd
import pytest


@pytest.mark.parametrize(
    "row, expected_pu_do",
    [
        ({"PULocationID": 82, "DOLocationID": 129}, "82_129"),
        ({"PULocationID": 0, "DOLocationID": 0}, "0_0"),
        ({"PULocationID": 999, "DOLocationID": 999}, "999_999"),
    ],
)
def test_add_pu_do_parametrized(row, expected_pu_do):
    """PU_DO combination works across normal, zero, and unseen zone IDs."""
    df = pd.DataFrame([row])
    result = add_pu_do(df)
    assert result["PU_DO"].iloc[0] == expected_pu_do


def test_to_dicts_handles_zero_distance():
    """trip_distance of zero should pass through without error."""
    df = pd.DataFrame(
        {"PULocationID": [82], "DOLocationID": [129], "trip_distance": [0.0]}
    )
    result = to_dicts(df)
    assert result[0]["trip_distance"] == 0.0


def test_to_dicts_handles_missing_category():
    """A missing PULocationID should not crash feature extraction."""
    df = pd.DataFrame(
        {"PULocationID": [None], "DOLocationID": [129], "trip_distance": [2.5]}
    )
    result = to_dicts(df)
    assert "PU_DO" in result[0]
