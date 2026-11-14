# Model Evaluation (time-based test split)

K = 10; relevant item = test rating >= 4; users evaluated = 1295; items already seen in training are excluded from every recommendation list.

| model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | rmse | mae | users_evaluated |
|---|---|---|---|---|---|---|---|---|
| popularity_baseline | 0.0863 | 0.2256 | 0.1866 | 0.0583 | 0.8325 | 0.8339 | 0.6757 | 1295 |
| svd_matrix_factorization | 0.1151 | 0.2931 | 0.2463 | 0.1467 | 0.7749 | 0.7547 | 0.5961 | 1295 |
| hybrid_mf_plus_features | 0.1143 | 0.2868 | 0.2462 | 0.1833 | 0.7502 | 0.7547 | 0.5961 | 1295 |
