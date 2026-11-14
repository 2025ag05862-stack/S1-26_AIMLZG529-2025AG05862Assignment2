import pandas as pd

from src.validation import validators as V

CFG = {"rating_min": 1, "rating_max": 5, "min_timestamp": "2022-01-01", "age_min": 10, "age_max": 100,
       "year_min": 1900, "allowed_genders": ["M", "F", "Other"]}


def _inter(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_interactions_rules_reject_bad_rows():
    df = _inter([
        ["U1", "I1", "5", "2023-01-01 10:00:00"],      # valid
        ["U1", "I2", "9", "2023-01-01 10:00:00"],      # rating out of range
        ["U1", "I1", "five", "2023-01-01 10:00:00"],   # wrong type
        ["U1", "I1", "4", "2031-01-01 10:00:00"],      # future
        ["U1", "I1", "4", "garbage"],                  # unparseable
        ["U1", "I1", None, "2023-01-01 10:00:00"],     # missing
        ["U9", "I1", "4", "2023-01-01 10:00:00"],      # orphan user
        ["U1", "I9", "4", "2023-01-01 10:00:00"],      # orphan item
        ["U1", "I1", "5", "2023-01-01 10:00:00"],      # exact duplicate of row 0
    ])
    ok, err, summary = V.validate_interactions(df, CFG, "2024-07-01", {"U1"}, {"I1", "I2"})
    assert len(ok) == 1 and ok.loc[0, "rating"] == 5.0
    assert set(err.source_row) == set(range(1, 9))
    assert str(ok.timestamp.dtype).startswith("datetime64")


def test_user_warnings_keep_record_but_log_it():
    df = pd.DataFrame({"user_id": ["U1", "U2", "U2"], "age": ["abc", "30", "30"], "gender": ["male", "M", "M"],
                       "country": ["India", None, None], "signup_date": ["2023-01-01"] * 3})
    ok, err, summary = V.validate_users(df, CFG)
    assert len(ok) == 2                                   # duplicate U2 rejected, U1 kept despite bad age
    assert pd.isna(ok.loc[ok.user_id == "U1", "age"]).all()
    rules = {s["rule"]: s["rows_failed"] for s in summary}
    assert rules["U02_user_id_unique"] == 1 and rules["U04_age_valid"] == 1


def test_items_year_validation():
    df = pd.DataFrame({"item_id": ["I1", "I2"], "title": ["a", "b"], "category": ["x", None],
                       "genres": ["x", "x"], "release_year": ["2090", "2000"]})
    ok, err, _ = V.validate_items(df, CFG, 2024)
    assert len(ok) == 2 and pd.isna(ok.release_year.iloc[0]) and ok.release_year.iloc[1] == 2000
