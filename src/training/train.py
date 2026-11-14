"""Training stage: fit the baseline + personalised models, track everything in MLflow."""
import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy import sparse

from src.common.config import load_config, path_for
from src.common.logging_utils import get_logger
from src.evaluation.metrics import ranking_metrics, rmse
from src.transformation.derived import interaction_matrix
from src.training.models import HybridRecommender, MFRecommender, PopularityRecommender

log = get_logger("training")


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["training"]["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["training"]["mlflow"]["experiment_name"])


def tune_factors(cfg, train, u_idx, i_idx) -> dict:
    """Pick the number of latent factors using ONLY training data: fit on the earlier 85% of the
    training period, score on the latest 15% (time-based validation). The test split is never touched."""
    t = cfg["training"]
    thr, k_eval = t["positive_threshold"], cfg["evaluation"]["k"]
    cut = train.timestamp.quantile(1 - t["tuning"]["val_fraction"])
    fit, val = train[train.timestamp <= cut], train[train.timestamp > cut]
    Rfit = interaction_matrix(fit, u_idx, i_idx).tocsr()
    v = val.merge(u_idx, on="user_id").merge(i_idx, on="item_id")
    rel = v[v.rating >= thr].groupby("user_idx").item_idx.apply(set)
    users = np.array([u for u in rel.index if Rfit.getnnz(axis=1)[u] > 0])
    relevant = [rel[u] for u in users]
    results = []
    for k in t["tuning"]["factor_grid"]:
        with mlflow.start_run(run_name=f"tune_factors_{k}", nested=False):
            m = MFRecommender(n_factors_rating=k, n_factors_rank=k, bias_reg=t["mf"]["bias_reg"],
                              positive_threshold=thr, seed=cfg["seed"]).fit(Rfit)
            ndcg = ranking_metrics(m.recommend(users, k_eval, Rfit), relevant)[2]
            err = rmse(v.rating, m.predict_rating(v.user_idx.values, v.item_idx.values))
            mlflow.log_params({"stage": "tuning", "n_factors": k})
            mlflow.log_metrics({"val_ndcg_at_10": ndcg, "val_rmse": err})
            results.append((k, ndcg, err))
            log.info("tuning k=%-3d val NDCG@%d=%.4f val RMSE=%.4f", k, k_eval, ndcg, err)
    best_rank = max(results, key=lambda r: r[1])[0]
    best_rating = min(results, key=lambda r: r[2])[0]
    log.info("selected n_factors_rank=%d (best val NDCG), n_factors_rating=%d (best val RMSE)", best_rank, best_rating)
    return {"n_factors_rank": best_rank, "n_factors_rating": best_rating}


def run(cfg=None) -> dict:
    cfg = cfg or load_config()
    proc, models_dir = path_for(cfg, "processed"), path_for(cfg, "models")
    t = cfg["training"]
    R = sparse.load_npz(proc / "user_item_matrix_train.npz")
    u_idx, i_idx = pd.read_parquet(proc / "user_index.parquet"), pd.read_parquet(proc / "item_index.parquet")
    setup_mlflow(cfg)

    chosen = {"n_factors_rank": t["mf"]["n_factors_rank"], "n_factors_rating": t["mf"]["n_factors_rating"]}
    if t["tuning"]["enabled"]:
        chosen = tune_factors(cfg, pd.read_parquet(proc / "train.parquet"), u_idx, i_idx)
    mf_p = dict(n_factors_rating=chosen["n_factors_rating"], n_factors_rank=chosen["n_factors_rank"],
                bias_reg=t["mf"]["bias_reg"], positive_threshold=t["positive_threshold"], seed=cfg["seed"])
    specs = [
        ("popularity_baseline", {"positive_threshold": t["positive_threshold"], "smoothing_m": t["popularity"]["smoothing_m"]},
         lambda: PopularityRecommender(t["positive_threshold"], t["popularity"]["smoothing_m"])),
        ("svd_matrix_factorization", mf_p, lambda: MFRecommender(**mf_p)),
    ]
    run_ids, mf_model = {}, None
    for name, params, make in specs:
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({**params, "model": name, "seed": cfg["seed"], "train_rows": int(R.nnz),
                               "n_users": R.shape[0], "n_items": R.shape[1]})
            t0 = time.time()
            model = make().fit(R)
            mlflow.log_metric("train_seconds", round(time.time() - t0, 3))
            path = models_dir / f"{name}.joblib"
            joblib.dump(model, path)
            mlflow.log_artifact(str(path))
            run_ids[name] = run.info.run_id
            if name == "svd_matrix_factorization":
                mf_model = model
            log.info("trained %s (run %s)", name, run.info.run_id[:8])

    # hybrid re-uses the fitted MF and the feature store (alpha is a config value, not tuned on test data)
    name, hp = "hybrid_mf_plus_features", {"alpha": t["hybrid"]["alpha"], "base": "svd_matrix_factorization"}
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({**hp, "model": name, "seed": cfg["seed"], "features_used": "ui_category_affinity"})
        hyb = HybridRecommender(mf_model, alpha=hp["alpha"], user_ids=list(u_idx.user_id), item_ids=list(i_idx.item_id))
        path = models_dir / f"{name}.joblib"
        joblib.dump(hyb, path)
        mlflow.log_artifact(str(path))
        run_ids[name] = run.info.run_id
        log.info("trained %s (run %s)", name, run.info.run_id[:8])
    (models_dir / "mlflow_run_ids.json").write_text(json.dumps(run_ids, indent=2))
    return run_ids


if __name__ == "__main__":
    run()
