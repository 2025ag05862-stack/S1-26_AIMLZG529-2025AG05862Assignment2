"""Evaluation stage: Precision/Recall/NDCG@K, Coverage, Diversity, RMSE, MAE for every model.
Metrics are logged to the SAME MLflow runs created in training; results table saved to reports/."""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy import sparse

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.evaluation import metrics as M
from src.features.feature_store import FeatureStore
from src.training.train import setup_mlflow

log = get_logger("evaluation")


def run(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    proc, models_dir, reports = path_for(cfg, "processed"), path_for(cfg, "models"), path_for(cfg, "reports")
    K, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    R = sparse.load_npz(proc / "user_item_matrix_train.npz").tocsr()
    u_idx, i_idx = pd.read_parquet(proc / "user_index.parquet"), pd.read_parquet(proc / "item_index.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet").set_index("item_id").reindex(i_idx.item_id)
    test = pd.read_parquet(proc / "test.parquet").merge(u_idx, on="user_id").merge(i_idx, on="item_id")

    # relevant items per user = test items rated >= threshold; evaluate users who have train history
    rel = test[test.rating >= thr].groupby("user_idx").item_idx.apply(set)
    has_train = np.asarray(R.getnnz(axis=1)) > 0
    eval_users = np.array([u for u in rel.index if has_train[u]])
    relevant = [rel[u] for u in eval_users]
    log.info("evaluating %d users with >=1 relevant test item (K=%d, relevant: rating>=%s)", len(eval_users), K, thr)

    # genre multi-hot for diversity
    genres = items.genres.str.split("|")
    vocab = sorted({g for gs in genres for g in gs})
    G = np.array([[g in gs for g in vocab] for gs in genres])

    store = FeatureStore(cfg)
    run_ids = json.loads((models_dir / "mlflow_run_ids.json").read_text())
    rows = []
    setup_mlflow(cfg)
    for name, run_id in run_ids.items():
        model = joblib.load(models_dir / f"{name}.joblib")
        if hasattr(model, "attach_store"):
            model.attach_store(store)
        recs = model.recommend(eval_users, K, R)
        p, r, n = M.ranking_metrics(recs, relevant)
        pred = model.predict_rating(test.user_idx.values, test.item_idx.values)
        res = {"model": name, f"precision@{K}": p, f"recall@{K}": r, f"ndcg@{K}": n,
               "coverage": M.coverage(recs, R.shape[1]), "diversity": M.intra_list_diversity(recs, G),
               "rmse": M.rmse(test.rating, pred), "mae": M.mae(test.rating, pred), "users_evaluated": len(eval_users)}
        rows.append(res)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in res.items() if k not in ("model", "users_evaluated")})
            mlflow.log_metric("users_evaluated", len(eval_users))
        log.info("%s: %s", name, {k: round(v, 4) for k, v in res.items() if k != "model"})

    table = pd.DataFrame(rows)
    table.to_csv(reports / "model_comparison.csv", index=False)
    fmt = table.copy()
    for c in fmt.columns[1:-1]:
        fmt[c] = fmt[c].map(lambda v: f"{v:.4f}")
    md = ["# Model Evaluation (time-based test split)", "",
          f"K = {K}; relevant item = test rating >= {thr}; users evaluated = {len(eval_users)}; "
          "items already seen in training are excluded from every recommendation list.", "",
          "| " + " | ".join(fmt.columns) + " |", "|" + "---|" * len(fmt.columns)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in fmt.itertuples(index=False)]
    (reports / "model_comparison.md").write_text("\n".join(md) + "\n")

    # export MLflow run table as a portable evidence file (mlflow.db itself is git-ignored)
    exp = mlflow.get_experiment_by_name(cfg["training"]["mlflow"]["experiment_name"])
    mlflow.search_runs([exp.experiment_id]).drop(columns=["tags.mlflow.user", "tags.mlflow.source.name"], errors="ignore") \
        .to_csv(reports / "mlflow_runs.csv", index=False)
    return table


if __name__ == "__main__":
    run()
