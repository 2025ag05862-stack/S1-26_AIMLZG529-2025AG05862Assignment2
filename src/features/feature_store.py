"""A small custom feature layer (Feast-like API on top of Parquet files).

materialize()  -> offline job, writes data/features/*.parquet
FeatureStore   -> read API used by BOTH training (hybrid model, interaction features for
                  the training pairs) and prediction (candidate scoring). Recency features
                  are recomputed against the requested `as_of` time at read time.
"""
import json

import numpy as np
import pandas as pd

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.features import build_features as B
from src.features.definitions import FEATURES

log = get_logger("features")


def materialize(cfg=None) -> dict:
    cfg = cfg or load_config()
    proc, feat, reports = path_for(cfg, "processed"), path_for(cfg, "features"), path_for(cfg, "reports")
    thr, m = cfg["training"]["positive_threshold"], cfg["features"]["rating_smoothing_m"]
    train = pd.read_parquet(proc / "train.parquet")
    users, items = pd.read_parquet(proc / "users_clean.parquet"), pd.read_parquet(proc / "items_clean.parquet")
    as_of = train.timestamp.max()                       # features are "as of" the end of the training window

    uf = B.compute_user_features(train, users, items, thr)
    itf = B.compute_item_features(train, items, thr, m, as_of)
    aff = B.compute_user_category_affinity(train, items)
    uf.to_parquet(feat / "user_features.parquet", index=False)
    itf.to_parquet(feat / "item_features.parquet", index=False)
    aff.reset_index().to_parquet(feat / "user_category_affinity.parquet", index=False)
    meta = {"as_of": str(as_of), "global_mean_rating": float(train.rating.mean()),
            "n_user_rows": len(uf), "n_item_rows": len(itf)}
    (feat / "feature_meta.json").write_text(json.dumps(meta, indent=2))

    store = FeatureStore(cfg)
    pairs = store.get_interaction_features(train.user_id, train.item_id)     # training-time interaction features
    pairs.insert(0, "item_id", train.item_id.values); pairs.insert(0, "user_id", train.user_id.values)
    pairs.to_parquet(feat / "interaction_features_train.parquet", index=False)

    cat = pd.DataFrame(FEATURES, columns=["feature", "entity", "description", "logic", "dtype"])
    cat.to_csv(reports / "feature_catalog.csv", index=False)
    lines = ["# Feature Catalog", "", f"Features are computed from the TRAIN split only, as of `{as_of}`.", "",
             "| Feature | Entity | Description | Logic | Type |", "|---|---|---|---|---|"]
    lines += [f"| `{r.feature}` | {r.entity} | {r.description} | `{r.logic}` | {r.dtype} |" for r in cat.itertuples()]
    (reports / "feature_catalog.md").write_text("\n".join(lines) + "\n")
    log.info("materialized %d user, %d item, %d interaction-feature rows", len(uf), len(itf), len(pairs))
    return meta


class FeatureStore:
    def __init__(self, cfg=None):
        cfg = cfg or load_config()
        d = path_for(cfg, "features")
        self.meta = json.loads((d / "feature_meta.json").read_text())
        self.as_of = pd.Timestamp(self.meta["as_of"])
        self.gm = self.meta["global_mean_rating"]
        self.users = pd.read_parquet(d / "user_features.parquet").set_index("user_id")
        self.items = pd.read_parquet(d / "item_features.parquet").set_index("item_id")
        self.affinity = pd.read_parquet(d / "user_category_affinity.parquet").set_index("user_id")

    def get_user_features(self, user_ids, as_of=None) -> pd.DataFrame:
        f = self.users.reindex(list(user_ids)).copy()
        f["user_days_since_last_interaction"] = (pd.Timestamp(as_of or self.as_of) - f.user_last_interaction_ts).dt.days
        return f.reset_index()

    def get_item_features(self, item_ids) -> pd.DataFrame:
        return self.items.reindex(list(item_ids)).reset_index()

    def category_affinity_matrix(self, user_ids, item_ids) -> np.ndarray:
        """(n_users, n_items) matrix of ui_category_affinity, used for batch scoring."""
        cats = self.items.reindex(list(item_ids)).item_category.values
        A = self.affinity.reindex(list(user_ids)).reindex(columns=sorted(set(cats))).fillna(0.0)
        return A.reindex(columns=cats).fillna(0.0).to_numpy()

    def get_interaction_features(self, user_ids, item_ids) -> pd.DataFrame:
        u = self.users.reindex(list(user_ids)).reset_index(drop=True)
        i = self.items.reindex(list(item_ids)).reset_index(drop=True)
        cats = i.item_category.values
        aff = self.affinity.reindex(list(user_ids))
        vals = self._vector_affinity(aff, cats)
        out = pd.DataFrame({"ui_category_affinity": vals})
        out["ui_in_preferred_category"] = (cats == u.user_preferred_category.values).astype(int)
        out["ui_baseline_rating"] = (u.user_avg_rating.fillna(self.gm) + i.item_bayes_rating.fillna(self.gm) - self.gm).clip(1, 5).values
        return out

    @staticmethod
    def _vector_affinity(aff, cats):
        col_pos = {c: j for j, c in enumerate(aff.columns)}
        A = aff.to_numpy()
        j = np.array([col_pos.get(c, -1) for c in cats])
        vals = np.where(j >= 0, A[np.arange(len(cats)), np.where(j >= 0, j, 0)], 0.0)
        return np.nan_to_num(vals)


if __name__ == "__main__":
    materialize()
