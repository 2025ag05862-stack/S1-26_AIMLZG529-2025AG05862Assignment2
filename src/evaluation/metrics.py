"""Ranking, beyond-accuracy and rating metrics (pure numpy; unit-tested)."""
import numpy as np


def ranking_metrics(recs: np.ndarray, relevant: list):
    """recs: (n_users, K) item indices. relevant: list of sets (one per user). Binary relevance."""
    K = recs.shape[1]
    discounts = 1.0 / np.log2(np.arange(2, K + 2))
    p, r, n = [], [], []
    for row, rel in zip(recs, relevant):
        hits = np.array([it in rel for it in row], dtype=float)
        p.append(hits.sum() / K)
        r.append(hits.sum() / len(rel))
        dcg = (hits * discounts).sum()
        idcg = discounts[: min(len(rel), K)].sum()
        n.append(dcg / idcg)
    return float(np.mean(p)), float(np.mean(r)), float(np.mean(n))


def coverage(recs: np.ndarray, n_items: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    return float(len(np.unique(recs)) / n_items)


def intra_list_diversity(recs: np.ndarray, item_genre_matrix: np.ndarray) -> float:
    """Mean over users of (1 - average pairwise Jaccard similarity of genre sets in the list)."""
    G = item_genre_matrix.astype(float)
    sizes = G.sum(axis=1)
    out = []
    for row in recs:
        g = G[row]
        inter = g @ g.T
        union = sizes[row][:, None] + sizes[row][None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        k = len(row)
        out.append(1.0 - (sim.sum() - np.trace(sim)) / (k * (k - 1)))
    return float(np.mean(out))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))
