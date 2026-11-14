"""Cleaning & standardisation of the validated tables (pure functions + a stage runner)."""
import pandas as pd

GENDER_MAP = {"m": "M", "male": "M", "f": "F", "female": "F", "other": "Other"}


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["gender"] = u.gender.astype("string").str.strip().str.lower().map(GENDER_MAP).fillna("Other")
    u["country"] = u.country.astype("string").str.strip().str.title().fillna("Unknown")
    u["age_imputed"] = u.age.isna()
    u["age"] = u.age.fillna(u.age.median()).astype(int)               # median imputation
    u["signup_date"] = u.signup_date.fillna(u.signup_date.median())
    return u


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    i = items.copy()
    i["category"] = i.category.astype("string").str.strip().str.lower().fillna("unknown")
    i["genres"] = i.genres.astype("string").str.strip().str.lower()
    i["genres"] = i.genres.where(i.genres.notna() & (i.genres != ""), i.category)
    i["title"] = i.title.astype("string").fillna("Untitled")
    i["release_year_imputed"] = i.release_year.isna()
    i["release_year"] = i.release_year.fillna(i.release_year.median()).astype(int)
    return i


def clean_interactions(x: pd.DataFrame):
    """Keep ONE row per (user, item): the latest rating (re-ratings supersede older ones)."""
    x = x.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
    before = len(x)
    x = x.drop_duplicates(["user_id", "item_id"], keep="last").reset_index(drop=True)
    x["rating"] = x.rating.astype(float)
    return x, before - len(x)
