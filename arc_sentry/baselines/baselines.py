import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from sklearn.mixture import GaussianMixture

# τ* = √(3/2) ≈ 1.2247 — Landauer threshold, Fisher manifold H²×H²
# Used for SNR reporting only in activation space.
# Not used as a hard threshold floor — activation layer deltas live on
# the unit hypersphere S^n, not H²×H², so TAU_STAR is not a valid floor here.
TAU_STAR = math.sqrt(3.0 / 2.0)


def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def fisher_rao_dist(u, v):
    """
    Fisher-Rao geodesic distance between two L2-normalized vectors.
    d(u,v) = arccos(u·v)
    """
    cos_sim = float(np.clip(np.dot(u, v), -1.0 + 1e-7, 1.0 - 1e-7))
    return float(np.arccos(cos_sim))


class BaseBaseline(ABC):
    @abstractmethod
    def fit(self, vectors: List[np.ndarray]): pass

    @abstractmethod
    def fit_with_probes(self, warmup_vectors: List[np.ndarray],
                        probe_vectors: List[np.ndarray]): pass

    @abstractmethod
    def score(self, vector: np.ndarray) -> float: pass

    @abstractmethod
    def threshold(self) -> float: pass


class CentroidBaseline(BaseBaseline):
    """
    Single centroid + Fisher-Rao geodesic distance.
    Threshold set from actual probe separation when probes available.
    Falls back to mean + safety_factor * std from warmup only.
    """

    def __init__(self, safety_factor: float = 2.0):
        self.safety_factor = safety_factor
        self._centroid = None
        self._threshold = None
        self._warmup_mean = None
        self._warmup_std = None

    def fit(self, vectors):
        """Fit from warmup only — threshold from variance."""
        stack = np.stack(vectors)
        cn = l2_normalize(stack.mean(axis=0))
        self._centroid = cn
        dists = [fisher_rao_dist(v, cn) for v in vectors]
        mu  = float(np.mean(dists))
        sig = float(np.std(dists)) + 1e-8
        self._warmup_mean = mu
        self._warmup_std  = sig
        self._threshold = mu + self.safety_factor * sig

    def fit_with_probes(self, warmup_vectors, probe_vectors):
        """
        Fit threshold using actual probe separation.
        Threshold = midpoint between max(warmup_dists) and min(probe_dists).
        This is the geometrically correct approach — set threshold where
        the two distributions are furthest apart.
        """
        stack = np.stack(warmup_vectors)
        cn = l2_normalize(stack.mean(axis=0))
        self._centroid = cn

        warmup_dists = [fisher_rao_dist(v, cn) for v in warmup_vectors]
        probe_dists  = [fisher_rao_dist(v, cn) for v in probe_vectors]

        mu  = float(np.mean(warmup_dists))
        sig = float(np.std(warmup_dists)) + 1e-8
        self._warmup_mean = mu
        self._warmup_std  = sig

        max_warmup = float(np.max(warmup_dists))
        min_probe  = float(np.min(probe_dists))

        if min_probe > max_warmup:
            # Clean separation — set threshold at midpoint
            self._threshold = (max_warmup + min_probe) / 2.0
            print(f"    probe separation clean: warmup_max={max_warmup:.4f} "
                  f"probe_min={min_probe:.4f} → threshold={self._threshold:.4f}")
        else:
            # Overlap — fall back to safety_factor on warmup std
            self._threshold = mu + self.safety_factor * sig
            print(f"    probe overlap: falling back to mean+{self.safety_factor}*std "
                  f"→ threshold={self._threshold:.4f}")

    def score(self, vector):
        return fisher_rao_dist(vector, self._centroid)

    def threshold(self):
        return self._threshold

    def snr(self, score):
        """Signal-to-noise ratio relative to TAU_STAR — for reporting."""
        return round(score / TAU_STAR, 3)


class MultiCentroidBaseline(BaseBaseline):
    def __init__(self, n_components: int = 2, safety_factor: float = 3.0):
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
        self._threshold = np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores)

    def fit_with_probes(self, warmup_vectors, probe_vectors):
        self.fit(warmup_vectors)

    def score(self, vector):
        fr = fisher_rao_dist(vector, self._mean_vec)
        return float(-self._gmm.score_samples([[fr]])[0])

    def threshold(self):
        return self._threshold


class KNNBaseline(BaseBaseline):
    def __init__(self, k: int = 3, safety_factor: float = 2.0):
        self.k = k
        self.safety_factor = safety_factor
        self._refs = None
        self._threshold = None

    def fit(self, vectors):
        self._refs = list(vectors)
        warmup_scores = [self._knn_dist(v) for v in vectors]
        self._threshold = np.mean(warmup_scores) + self.safety_factor * np.std(warmup_scores)

    def fit_with_probes(self, warmup_vectors, probe_vectors):
        self.fit(warmup_vectors)

    def _knn_dist(self, vector):
        dists = sorted([fisher_rao_dist(vector, r) for r in self._refs])
        return float(np.mean(dists[:self.k]))

    def score(self, vector):
        return self._knn_dist(vector)

    def threshold(self):
        return self._threshold
