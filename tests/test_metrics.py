import numpy as np
import pytest

from src.evaluation import metrics as M


def test_perfect_and_zero_ranking():
    recs = np.array([[0, 1, 2], [3, 4, 5]])
    p, r, n = M.ranking_metrics(recs, [{0, 1, 2}, {9}])
    assert p == pytest.approx((1.0 + 0.0) / 2) and r == pytest.approx((1.0 + 0.0) / 2) and n == pytest.approx(0.5)


def test_ndcg_rewards_higher_ranks():
    good, _ = M.ranking_metrics(np.array([[0, 1, 2]]), [{0}])[2], None
    bad = M.ranking_metrics(np.array([[1, 2, 0]]), [{0}])[2]
    assert good == pytest.approx(1.0) and bad == pytest.approx(0.5)      # 1/log2(4) = 0.5


def test_coverage():
    assert M.coverage(np.array([[0, 1], [1, 2]]), 10) == pytest.approx(0.3)


def test_diversity_identical_vs_disjoint_genres():
    G = np.array([[1, 0], [1, 0], [0, 1]])
    assert M.intra_list_diversity(np.array([[0, 1]]), G) == pytest.approx(0.0)    # same genres
    assert M.intra_list_diversity(np.array([[0, 2]]), G) == pytest.approx(1.0)    # disjoint genres


def test_rmse_mae():
    assert M.rmse([1, 3], [2, 5]) == pytest.approx(np.sqrt(2.5))
    assert M.mae([1, 3], [2, 5]) == pytest.approx(1.5)
