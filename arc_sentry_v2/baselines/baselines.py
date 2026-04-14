
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from sklearn.mixture import GaussianMixture


def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


class BaseBaseline(ABC):
    @abstractmethod
    def fit(self, vectors: List[np.ndarray]): pass

    @abstractmethod
    def score(self, vector: np.ndarray) -> float: pass

    @abstractmethod
    def threshold(self) -> float: pass


class CentroidBaseline(BaseBaseline):
    """Single centroid + cosine distance. For homogeneous routes."""

    def __init__(self, safety_factor: float = 2.0):
        self.safety_factor = safety_factor
        self._centroid = None
        self._threshold = None

    def fit(self, vectors):
        stack = np.stack(vectors)
        cn = stack.mean(axis=0)
        self._centroid = l2_normalize(cn)
        dists = [float(1.0 - np.dot(v, self._centroid)) for v in vectors]
        self._threshold = max(dists) * self.safety_factor

    def score(self, vector):
        return float(1.0 - np.dot(vector, self._centroid))

    def threshold(self):
        return self._threshold


class MultiCentroidBaseline(BaseBaseline):
    """
    GMM-based multi-centroid baseline.
    Default for v2 -- handles multimodal normal traffic.
    Score = negative log likelihood under GMM.
    """

    def __init__(self, n_components: int = 2,
                 safety_factor: float = 3.0):
        self.n_components = n_components
        self.safety_factor = safety_factor
        self._gmm = None
        self._threshold = None

    def fit(self, vectors):
        # Use norm of vectors as 1D signal for GMM
        scores_1d = np.array([np.linalg.norm(v) for v in vectors]).reshape(-1, 1)
        k = min(self.n_components, len(vectors))
        self._gmm = GaussianMixture(n_components=k, random_state=42)
        self._gmm.fit(scores_1d)
        warmup_scores = [-self._gmm.score_samples([[s]])[0] for s in scores_1d]
        self._threshold = np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores)

    def score(self, vector):
        s = np.linalg.norm(vector)
        return float(-self._gmm.score_samples([[s]])[0])

    def threshold(self):
        return self._threshold


class KNNBaseline(BaseBaseline):
    """
    k-nearest neighbor baseline.
    Use when warmup > 20 samples.
    """

    def __init__(self, k: int = 3, safety_factor: float = 2.0):
        self.k = k
        self.safety_factor = safety_factor
        self._refs = None
        self._threshold = None

    def fit(self, vectors):
        self._refs = np.stack(vectors)
        warmup_scores = [self._knn_dist(v) for v in vectors]
        self._threshold = np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores)

    def _knn_dist(self, vector):
        dists = np.linalg.norm(self._refs - vector, axis=1)
        return float(np.sort(dists)[:self.k].mean())

    def score(self, vector):
        return self._knn_dist(vector)

    def threshold(self):
        return self._threshold
