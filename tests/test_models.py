import numpy as np
from scipy import sparse

from src.training.models import MFRecommender, PopularityRecommender


def _toy():
    rng = np.random.default_rng(0)
    R = (rng.random((30, 12)) < 0.3) * rng.integers(1, 6, (30, 12))
    return sparse.csr_matrix(R.astype(float))


def test_models_never_recommend_seen_items_and_are_deterministic():
    R = _toy()
    for m in (PopularityRecommender().fit(R), MFRecommender(3, 3).fit(R)):
        users = np.arange(R.shape[0])
        recs = m.recommend(users, 5, R)
        assert recs.shape == (30, 5)
        for u, row in zip(users, recs):
            assert not set(row) & set(R[u].indices)
    a = MFRecommender(3, 3, seed=1).fit(R).recommend(np.arange(5), 5, R)
    b = MFRecommender(3, 3, seed=1).fit(R).recommend(np.arange(5), 5, R)
    assert (a == b).all()


def test_rating_predictions_are_clipped():
    R = _toy()
    p = MFRecommender(3, 3).fit(R).predict_rating(np.arange(10), np.arange(10))
    assert p.min() >= 1 and p.max() <= 5
