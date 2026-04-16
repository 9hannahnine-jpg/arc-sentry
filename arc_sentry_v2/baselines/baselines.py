import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from sklearn.mixture import GaussianMixture

# τ* = √(3/2) ≈ 1.2247 — Landauer threshold, Fisher manifold H²×H²
# Distances below this are below the geometric noise floor and treated
# as behaviorally equivalent. Derived in Nine (2026c), not tuned.
TAU_STAR = math.sqrt(3.0 / 2.0)


def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def fisher_rao_dist(u, v):
    """
    Fisher-Rao geodesic distance between two L2-normalized vectors.
    d(u,v) = arccos(u·v)

    Cosine distance (1 - u·v) is an approximation that diverges from
    the true geodesic at large angles — exactly when drift is occurring.
    TAU_STAR is meaningful only under this metric.
    """
    cos_sim = float(np.clip(np.dot(u, v), -1.0 + 1e-7, 1.0 - 1e-7))
    return float(np.arccos(cos_sim))


class BaseBaseline(ABC):
    @abstractmethod
    def fit(self, vectors: List[np.ndarray]): pass

    @abstractmethod
    def score(self, vector: np.ndarray) -> float: pass

    @abstractmethod
    def threshold(self) -> float: pass


class CentroidBaseline(BaseBaseline):
    """
    Single centroid + Fisher-Rao geodesic distance.
    Threshold = max(mean + safety_factor * std, TAU_STAR).
    TAU_STAR serves as a hard geometric noise floor — the threshold
    can never be set below the Landauer threshold of the manifold.
    """

    def __init__(self, safety_factor: float = 2.0):
        self.safety_factor = safety_factor
        self._centroid = None
        self._threshold = None
        self._warmup_mean = None
        self._warmup_std = None

    def fit(self, vectors):
        stack = np.stack(vectors)
        cn = l2_normalize(stack.mean(axis=0))
        self._centroid = cn
        dists = [fisher_rao_dist(v, cn) for v in vectors]
        mu  = float(np.mean(dists))
        sig = float(np.std(dists)) + 1e-8
        self._warmup_mean = mu
        self._warmup_std  = sig
        # Hard floor at TAU_STAR — threshold never below manifold noise floor
        self._threshold = max(mu + self.safety_factor * sig, TAU_STAR)

    def score(self, vector):
        return fisher_rao_dist(vector, self._centroid)

    def threshold(self):
        return self._threshold

    def snr(self, score):
        """Signal-to-noise ratio relative to TAU_STAR."""
        return round(score / TAU_STAR, 3)


class MultiCentroidBaseline(BaseBaseline):
    """
    GMM-based multi-centroid baseline.
    For multimodal normal traffic (e.g. mixed-domain deployments).
    Score = negative log likelihood under GMM.
    """

    def __init__(self, n_components: int = 2,
                 safety_factor: float = 3.0):
        self.n_components = n_components
        self.safety_factor = safety_factor
        self._gmm = None
        self._threshold = None
        self._mean_vec = None

    def fit(self, vectors):
        self._mean_vec = l2_normalize(np.stack(vectors).mean(axis=0))
        scores_1d = np.array([fisher_rao_dist(v, self._mean_vec)
                               for v in vectors]).reshape(-1, 1)
        k = min(self.n_components, len(vectors))
        self._gmm = GaussianMixture(n_components=k, random_state=42)
        self._gmm.fit(scores_1d)
        warmup_scores = [-self._gmm.score_samples([[s]])[0] for s in scores_1d]
        self._threshold = max(
            np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores),
            TAU_STAR)

    def score(self, vector):
        fr = fisher_rao_dist(vector, self._mean_vec)
        return float(-self._gmm.score_samples([[fr]])[0])

    def threshold(self):
        return self._threshold


class KNNBaseline(BaseBaseline):
    """
    k-nearest neighbor baseline using Fisher-Rao distance.
    Use when warmup > 20 samples.
    """

    def __init__(self, k: int = 3, safety_factor: float = 2.0):
        self.k = k
        self.safety_factor = safety_factor
        self._refs = None
        self._threshold = None

    def fit(self, vectors):
        self._refs = list(vectors)
        warmup_scores = [self._knn_dist(v) for v in vectors]
        self._threshold = max(
            np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores),
            TAU_STAR)

    def _knn_dist(self, vector):
        dists = sorted([fisher_rao_dist(vector, r) for r in self._refs])
        return float(np.mean(dists[:self.k]))

    def score(self, vector):
        return self._knn_dist(vector)

    def threshold(self):
        return self._threshold
