import pandas as pd

from src.preparation.clean import clean_interactions, clean_users
from src.preparation.split import time_split


def test_time_split_has_no_overlap_in_time():
    ts = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    x = pd.DataFrame({"user_id": "U", "item_id": [f"I{i}" for i in range(100)], "rating": 4.0, "timestamp": ts})
    train, test, cutoff = time_split(x, 0.2)
    assert train.timestamp.max() < test.timestamp.min()
    assert abs(len(test) - 20) <= 1


def test_clean_interactions_keeps_latest_rating():
    ts = pd.to_datetime(["2023-01-01", "2023-02-01"], utc=True)
    x = pd.DataFrame({"user_id": ["U1", "U1"], "item_id": ["I1", "I1"], "rating": [2.0, 5.0], "timestamp": ts})
    out, dropped = clean_interactions(x)
    assert dropped == 1 and out.rating.iloc[0] == 5.0


def test_clean_users_standardises_gender_and_imputes_age():
    u = pd.DataFrame({"user_id": ["U1", "U2", "U3"], "age": [20, None, 40], "gender": ["male", "F", "weird"],
                      "country": [" india ", None, "USA"],
                      "signup_date": pd.to_datetime(["2023-01-01", "2023-01-02", None])})
    out = clean_users(u)
    assert list(out.gender) == ["M", "F", "Other"]
    assert out.country.iloc[0] == "India" and out.country.iloc[1] == "Unknown"
    assert out.age.iloc[1] == 30 and bool(out.age_imputed.iloc[1])
