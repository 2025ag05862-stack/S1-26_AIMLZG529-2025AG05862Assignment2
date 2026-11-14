"""Rule-based data validation (pure functions, easy to unit-test).

Policy
------
* severity="error"   -> the record is REJECTED and written to data/errors/ (quarantine).
* severity="warning" -> the record is KEPT (bad attribute is coerced to NA and fixed in
                        the preparation stage) but the problem is still logged/reported.
Every rule failure is recorded, so a row that breaks 3 rules appears 3 times in the
error file (one line per rule) but is rejected only once.
"""
import pandas as pd


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


class _Checker:
    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df = table, df
        self.reject = pd.Series(False, index=df.index)
        self.summary, self.frames = [], []

    def check(self, rule, column, mask, detail, severity="error"):
        mask = mask.reindex(self.df.index, fill_value=False).astype(bool)
        n = int(mask.sum())
        self.summary.append({"table": self.table, "rule": rule, "column": column, "severity": severity,
                             "description": detail, "rows_checked": len(self.df), "rows_failed": n,
                             "pct_failed": round(100 * n / max(len(self.df), 1), 3)})
        if n:
            bad = self.df.loc[mask].copy()
            bad.insert(0, "source_row", bad.index)
            bad["rule"], bad["column"], bad["severity"], bad["detail"] = rule, column, severity, detail
            self.frames.append(bad)
        if severity == "error":
            self.reject |= mask

    def errors(self) -> pd.DataFrame:
        return pd.concat(self.frames, ignore_index=True) if self.frames else pd.DataFrame()


def validate_users(df: pd.DataFrame, v: dict):
    c = _Checker("users", df)
    c.check("U01_user_id_not_null", "user_id", _blank(df.user_id), "user_id must be present")
    c.check("U02_user_id_unique", "user_id", df.user_id.duplicated(keep="first") & ~_blank(df.user_id),
            "duplicate user_id (first occurrence kept)")
    age = pd.to_numeric(df.age, errors="coerce")
    c.check("U03_age_missing", "age", _blank(df.age), "age missing (imputed later)", "warning")
    c.check("U04_age_valid", "age", ~_blank(df.age) & (age.isna() | ~age.between(v["age_min"], v["age_max"])),
            f"age must be numeric in [{v['age_min']}, {v['age_max']}]", "warning")
    c.check("U05_country_missing", "country", _blank(df.country), "country missing (filled later)", "warning")
    c.check("U06_gender_standard", "gender", ~_blank(df.gender) & ~df.gender.isin(v["allowed_genders"]),
            f"gender not in {v['allowed_genders']} (standardised later)", "warning")
    signup = pd.to_datetime(df.signup_date, errors="coerce")
    c.check("U07_signup_date_parseable", "signup_date", ~_blank(df.signup_date) & signup.isna(),
            "signup_date must be a valid date", "warning")
    ok = df[~c.reject].copy()
    ok["age"] = pd.to_numeric(ok.age, errors="coerce")
    ok.loc[~ok.age.between(v["age_min"], v["age_max"]), "age"] = pd.NA
    ok["signup_date"] = pd.to_datetime(ok.signup_date, errors="coerce")
    return ok.reset_index(drop=True), c.errors(), c.summary


def validate_items(df: pd.DataFrame, v: dict, ref_year: int):
    c = _Checker("items", df)
    c.check("I01_item_id_not_null", "item_id", _blank(df.item_id), "item_id must be present")
    c.check("I02_item_id_unique", "item_id", df.item_id.duplicated(keep="first") & ~_blank(df.item_id),
            "duplicate item_id (first occurrence kept)")
    c.check("I03_title_missing", "title", _blank(df.title), "title missing", "warning")
    c.check("I04_category_missing", "category", _blank(df.category), "category missing (set to 'unknown')", "warning")
    year = pd.to_numeric(df.release_year, errors="coerce")
    c.check("I05_release_year_valid", "release_year", ~_blank(df.release_year) & (year.isna() | ~year.between(v["year_min"], ref_year)),
            f"release_year must be numeric in [{v['year_min']}, {ref_year}]", "warning")
    ok = df[~c.reject].copy()
    ok["release_year"] = pd.to_numeric(ok.release_year, errors="coerce")
    ok.loc[~ok.release_year.between(v["year_min"], ref_year), "release_year"] = pd.NA
    return ok.reset_index(drop=True), c.errors(), c.summary


def validate_interactions(df: pd.DataFrame, v: dict, ref_date, valid_users, valid_items):
    c = _Checker("interactions", df)
    for col in ["user_id", "item_id", "rating", "timestamp"]:
        c.check(f"X01_{col}_not_null", col, _blank(df[col]), f"{col} must be present")
    rating = pd.to_numeric(df.rating, errors="coerce")
    c.check("X02_rating_numeric", "rating", ~_blank(df.rating) & rating.isna(), "rating must be numeric (data type)")
    c.check("X03_rating_range", "rating", rating.notna() & ~rating.between(v["rating_min"], v["rating_max"]),
            f"rating must be in [{v['rating_min']}, {v['rating_max']}]")
    ts = pd.to_datetime(df.timestamp, errors="coerce", utc=True, format="%Y-%m-%d %H:%M:%S")
    c.check("X04_timestamp_parseable", "timestamp", ~_blank(df.timestamp) & ts.isna(), "timestamp must be a valid datetime")
    lo, hi = pd.Timestamp(v["min_timestamp"], tz="UTC"), pd.Timestamp(ref_date, tz="UTC")
    c.check("X05_timestamp_in_range", "timestamp", ts.notna() & ((ts < lo) | (ts > hi)),
            f"timestamp must be within [{v['min_timestamp']}, {ref_date}] (no future / ancient dates)")
    c.check("X06_exact_duplicate", "all", df.duplicated(keep="first"), "exact duplicate row (first occurrence kept)")
    c.check("X07_user_referential_integrity", "user_id", ~_blank(df.user_id) & ~df.user_id.isin(valid_users),
            "user_id not found in validated users table")
    c.check("X08_item_referential_integrity", "item_id", ~_blank(df.item_id) & ~df.item_id.isin(valid_items),
            "item_id not found in validated items table")
    ok = df[~c.reject].copy()
    ok["rating"] = pd.to_numeric(ok.rating)
    ok["timestamp"] = pd.to_datetime(ok.timestamp, utc=True, format="%Y-%m-%d %H:%M:%S")
    return ok.reset_index(drop=True), c.errors(), c.summary
