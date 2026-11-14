"""Preparation stage: validated -> cleaned tables + time-based train/test split."""
import json

import pandas as pd

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.preparation.clean import clean_interactions, clean_items, clean_users
from src.preparation.split import time_split

log = get_logger("preparation")


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    proc, reports = path_for(cfg, "processed"), path_for(cfg, "reports")
    users = clean_users(pd.read_parquet(proc / "validated_users.parquet"))
    items = clean_items(pd.read_parquet(proc / "validated_items.parquet"))
    inter, n_reratings = clean_interactions(pd.read_parquet(proc / "validated_interactions.parquet"))

    train, test, cutoff = time_split(inter, cfg["preparation"]["test_fraction"])
    users.to_parquet(proc / "users_clean.parquet", index=False)
    items.to_parquet(proc / "items_clean.parquet", index=False)
    inter.to_parquet(proc / "interactions_clean.parquet", index=False)
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)

    train_users, train_items = set(train.user_id), set(train.item_id)
    summary = {
        "interactions_after_cleaning": len(inter),
        "superseded_re_ratings_dropped": int(n_reratings),
        "split_cutoff_timestamp": str(cutoff),
        "train_rows": len(train), "test_rows": len(test),
        "train_period": [str(train.timestamp.min()), str(train.timestamp.max())],
        "test_period": [str(test.timestamp.min()), str(test.timestamp.max())],
        "train_users": len(train_users), "train_items": len(train_items),
        "test_rows_cold_user": int((~test.user_id.isin(train_users)).sum()),
        "test_rows_cold_item": int((~test.item_id.isin(train_items)).sum()),
        "ages_imputed": int(users.age_imputed.sum()), "release_years_imputed": int(items.release_year_imputed.sum()),
    }
    (reports / "split_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("clean=%d train=%d test=%d cutoff=%s", len(inter), len(train), len(test), cutoff)
    return summary


if __name__ == "__main__":
    run()
