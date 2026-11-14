"""Recommenders. All share: fit(R) / recommend(user_idx, k, seen) / predict_rating(u, i).

R is the sparse (n_users x n_items) TRAIN rating matrix (0 = unobserved).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


def _mask_seen(scores: np.ndarray, seen: sparse.csr_matrix) -> np.ndarray:
    scores = scores.copy()
    r, c = seen.nonzero()
    scores[r, c] = -np.inf
    return scores


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


class PopularityRecommender:
    """Baseline: same ranking for everyone = items with most positive ratings (minus already seen)."""
    name = "popularity_baseline"

    def __init__(self, positive_threshold=4, smoothing_m=5):
        self.thr, self.m = positive_threshold, smoothing_m

    def fit(self, R):
        R = sparse.csr_matrix(R)
        pos = (R >= self.thr).astype(float)
        self.item_score = np.asarray(pos.sum(axis=0)).ravel()
        self.mu = R.data.mean()
        n = np.asarray((R > 0).sum(axis=0)).ravel()
        s = np.asarray(R.sum(axis=0)).ravel()
        self.item_mean = (s + self.m * self.mu) / (n + self.m)      # smoothed item mean for rating prediction
        return self

    def score_matrix(self, user_idx):
        return np.tile(self.item_score, (len(user_idx), 1)).astype(float)

    def recommend(self, user_idx, k, seen):
        s = _mask_seen(self.score_matrix(user_idx), seen[user_idx])
        return _top_k(s, k)

    def predict_rating(self, u, i):
        return np.clip(self.item_mean[i], 1, 5)


class MFRecommender:
    """Personalised matrix factorisation (truncated SVD), two complementary factorisations:
    * rating model  : biases + SVD of the residual matrix -> predicted ratings (RMSE / MAE)
    * ranking model : PureSVD of the binary 'positive' matrix -> top-N scores
    """
    name = "svd_matrix_factorization"

    def __init__(self, n_factors_rating=20, n_factors_rank=30, bias_reg=10, positive_threshold=4, seed=42):
        self.kr, self.kk, self.reg, self.thr, self.seed = n_factors_rating, n_factors_rank, bias_reg, positive_threshold, seed

    def fit(self, R):
        R = sparse.csr_matrix(R, dtype=float)
        n_u, n_i = R.shape
        mu = R.data.mean()
        obs = R.copy(); obs.data = np.ones_like(obs.data)
        # damped item then user biases
        bi = np.asarray((R - obs.multiply(mu)).sum(axis=0)).ravel() / (np.asarray(obs.sum(axis=0)).ravel() + self.reg)
        resid_i = R - obs.multiply(mu + bi[None, :]).tocsr()
        bu = np.asarray(resid_i.sum(axis=1)).ravel() / (np.asarray(obs.sum(axis=1)).ravel() + self.reg)
        resid = sparse.csr_matrix(R - obs.multiply(mu + bi[None, :] + bu[:, None]))
        k = min(self.kr, min(R.shape) - 1)
        U, s, Vt = svds(resid, k=k, random_state=self.seed)
        self.mu, self.bi, self.bu = mu, bi, bu
        self.Uf, self.Vt = U * s, Vt                                 # (n_users, k), (k, n_items)
        # ranking factorisation on the implicit-positive matrix
        B = (R >= self.thr).astype(float)
        kk = min(self.kk, min(B.shape) - 1)
        _, _, Vr = svds(B, k=kk, random_state=self.seed)
        self.item_proj = Vr.T @ Vr                                   # (n_items, n_items) low-rank item-item
        self.B = B.tocsr()
        return self

    def score_matrix(self, user_idx):
        return np.asarray(self.B[user_idx] @ self.item_proj)

    def recommend(self, user_idx, k, seen):
        return _top_k(_mask_seen(self.score_matrix(user_idx), seen[user_idx]), k)

    def predict_rating(self, u, i):
        u, i = np.asarray(u), np.asarray(i)
        pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("ij,ji->i", self.Uf[u], self.Vt[:, i])
        return np.clip(pred, 1, 5)


class HybridRecommender:
    """MF ranking score blended with a FEATURE-STORE signal (ui_category_affinity):
       score = z(mf_score) + alpha * z(category_affinity)   (z-scored per user)."""
    name = "hybrid_mf_plus_features"

    def __init__(self, mf: MFRecommender, alpha=0.3, user_ids=None, item_ids=None):
        self.mf, self.alpha = mf, alpha
        self.user_ids, self.item_ids = user_ids, item_ids
        self.store = None

    def attach_store(self, store):
        self.store = store
        return self

    def __getstate__(self):                       # the feature store is re-attached at load time, not pickled
        d = self.__dict__.copy(); d["store"] = None
        return d

    @staticmethod
    def _z(a):
        return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-9)

    def score_matrix(self, user_idx):
        mf = self._z(self.mf.score_matrix(user_idx))
        aff = self.store.category_affinity_matrix([self.user_ids[u] for u in user_idx], self.item_ids)
        return mf + self.alpha * self._z(aff)

    def recommend(self, user_idx, k, seen):
        return _top_k(_mask_seen(self.score_matrix(user_idx), seen[user_idx]), k)

    def predict_rating(self, u, i):
        return self.mf.predict_rating(u, i)
