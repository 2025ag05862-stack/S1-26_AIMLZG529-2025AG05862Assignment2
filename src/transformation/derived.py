"""Transformation stage: derived datasets built from the TRAIN split only
(user activity summary, item popularity, user-item interaction matrix + index maps)."""
import pandas as pd
from scipy import sparse

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger

log = get_logger("transformation")


def user_activity_summary(train: pd.DataFrame, thr: float) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(n_interactions=("item_id", "size"), avg_rating=("rating", "mean"),
                first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max")).reset_index()
    out["n_positive"] = out.user_id.map(train[train.rating >= thr].groupby("user_id").size()).fillna(0).astype(int)
    out["active_span_days"] = (out.last_interaction - out.first_interaction).dt.days
    return out


def item_popularity(train: pd.DataFrame, items: pd.DataFrame, thr: float) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(n_interactions=("user_id", "size"), avg_rating=("rating", "mean")).reset_index()
    out["n_positive"] = out.item_id.map(train[train.rating >= thr].groupby("item_id").size()).fillna(0).astype(int)
    out = items[["item_id", "category"]].merge(out, on="item_id", how="left")
    out[["n_interactions", "n_positive"]] = out[["n_interactions", "n_positive"]].fillna(0).astype(int)
    out["popularity_rank"] = out.n_interactions.rank(ascending=False, method="min").astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)


def build_index(users: pd.DataFrame, items: pd.DataFrame):
    u = pd.DataFrame({"user_id": sorted(users.user_id.unique())}); u["user_idx"] = range(len(u))
    i = pd.DataFrame({"item_id": sorted(items.item_id.unique())}); i["item_idx"] = range(len(i))
    return u, i


def interaction_matrix(train: pd.DataFrame, u_index: pd.DataFrame, i_index: pd.DataFrame) -> sparse.csr_matrix:
    r = train.merge(u_index, on="user_id").merge(i_index, on="item_id")
    return sparse.csr_matrix((r.rating.values, (r.user_idx.values, r.item_idx.values)),
                             shape=(len(u_index), len(i_index)))


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    proc = path_for(cfg, "processed")
    thr = cfg["training"]["positive_threshold"]
    train = pd.read_parquet(proc / "train.parquet")
    users, items = pd.read_parquet(proc / "users_clean.parquet"), pd.read_parquet(proc / "items_clean.parquet")

    uas = user_activity_summary(train, thr); uas.to_parquet(proc / "user_activity_summary.parquet", index=False)
    pop = item_popularity(train, items, thr); pop.to_parquet(proc / "item_popularity.parquet", index=False)
    u_idx, i_idx = build_index(users, items)
    u_idx.to_parquet(proc / "user_index.parquet", index=False); i_idx.to_parquet(proc / "item_index.parquet", index=False)
    M = interaction_matrix(train, u_idx, i_idx)
    sparse.save_npz(proc / "user_item_matrix_train.npz", M)
    density = M.nnz / (M.shape[0] * M.shape[1])
    log.info("matrix %s nnz=%d density=%.4f", M.shape, M.nnz, density)
    return {"matrix_shape": list(M.shape), "nnz": int(M.nnz), "density": float(density)}


if __name__ == "__main__":
    run()
