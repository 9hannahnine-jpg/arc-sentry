"""
D(t) stability scalar applied to residual stream layer depth.

From Nine (2026b,c):
  D(t) = λ(τ)·(Δt − T)
  λ(τ) = 3/τ² − 2
  τ* = √(3/2) ≈ 1.2247

τ is estimated from the residual stream trajectory:
  δx(L) = FR distance between test trajectory and warmup centroid at layer L
  D(L) = log S_global(L) − log S_local(L)
  λ fitted from D(L) = λ·(ΔL − T)
  τ = √(3/(λ + 2))

If τ < τ*: below Landauer threshold → unstable → BLOCK
If τ > τ*: above Landauer threshold → stable → ALLOW

The threshold is τ* itself — geometrically forced, not tuned.
"""

import math
import numpy as np
from typing import List, Optional
from arc_sentry_v2.models.base_adapter import BaseModelAdapter

TAU_STAR = math.sqrt(3.0 / 2.0)  # ≈ 1.2247, Landauer threshold

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)

def fisher_rao_dist(u: np.ndarray, v: np.ndarray) -> float:
    cos_sim = float(np.clip(np.dot(u, v), -1.0 + 1e-7, 1.0 - 1e-7))
    return float(np.arccos(cos_sim))

def lambda_from_tau(tau: float) -> float:
    """Manifold eigenvalue: λ(τ) = 3/τ² − 2."""
    return 3.0 / (tau ** 2) - 2.0

def tau_from_lambda(lam: float) -> float:
    """Invert λ(τ) = 3/τ² − 2 → τ = √(3/(λ+2))."""
    denom = lam + 2.0
    if denom <= 0:
        return 0.0  # unphysical — below manifold floor
    return math.sqrt(3.0 / denom)


class TrajectoryExtractor:
    """
    Extracts the residual stream trajectory for a prompt.
    Returns L2-normalized Δh vectors at each scanned layer.
    """

    def __init__(self, scan_start: float = 0.30):
        self.scan_start = scan_start

    def extract(self, adapter: BaseModelAdapter,
                prompt: str,
                system_prompt: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Returns array shape (n_layers, hidden_dim) — the sequence of
        L2-normalized Δh directions from scan_start to final layer.
        """
        n = adapter.n_layers
        start = int(n * self.scan_start)
        layer_pairs = sorted(set(
            idx for L in range(start, n) for idx in [L-1, L]
        ))
        hidden = adapter.extract_hidden(prompt, layer_pairs, system_prompt)

        trajectory = []
        for L in range(start, n):
            if (L-1) not in hidden or L not in hidden:
                continue
            delta = hidden[L] - hidden[L-1]
            trajectory.append(l2_normalize(delta))

        if not trajectory:
            return None
        return np.stack(trajectory)  # shape: (n_layers_scanned, hidden_dim)


class DtStabilityDetector:
    """
    Implements D(t) stability scalar from Nine (2026b,c).

    Applied to layer depth instead of time:
      - Builds centroid trajectory from warmup prompts
      - For each test prompt, computes δx(L) = FR dist to centroid at layer L
      - Computes D(L) across the trajectory
      - Fits λ from D(L) = λ·(ΔL − T)
      - Estimates τ = √(3/(λ+2))
      - Blocks if τ < τ* (below Landauer threshold)

    No tuned threshold. The decision boundary is τ* itself.
    """

    def __init__(self, short_horizon: int = 2, long_horizon: int = 8):
        """
        Args:
            short_horizon: ΔL for local stability (Δt in Paper 2)
            long_horizon:  T  for global stability
        """
        self.short_horizon = short_horizon
        self.long_horizon  = long_horizon
        self._centroid_trajectory = None
        self._n_layers = None
        self._warmup_taus = None

    def fit(self, trajectories: List[np.ndarray]):
        """Build centroid trajectory from warmup prompts."""
        min_len = min(t.shape[0] for t in trajectories)
        self._n_layers = min_len
        aligned = np.stack([t[:min_len] for t in trajectories])
        self._centroid_trajectory = np.stack([
            l2_normalize(aligned[:, L, :].mean(axis=0))
            for L in range(min_len)
        ])

        # Compute τ for each warmup prompt — should all be ≥ τ*
        self._warmup_taus = [self._estimate_tau(t[:min_len])
                             for t in trajectories]
        mu  = float(np.mean(self._warmup_taus))
        std = float(np.std(self._warmup_taus))
        print(f"    [D(t)] warmup τ: mean={mu:.4f} std={std:.4f} "
              f"min={min(self._warmup_taus):.4f} max={max(self._warmup_taus):.4f}")
        print(f"    [D(t)] τ* = {TAU_STAR:.6f}")
        print(f"    [D(t)] warmup prompts above τ*: "
              f"{sum(1 for t in self._warmup_taus if t >= TAU_STAR)}/{len(self._warmup_taus)}")

    def _compute_separations(self, trajectory: np.ndarray) -> np.ndarray:
        """
        δx(L) = FR distance between test trajectory and warmup centroid at layer L.
        Returns array of shape (n_layers,).
        """
        n = min(trajectory.shape[0], self._n_layers)
        return np.array([
            fisher_rao_dist(trajectory[L], self._centroid_trajectory[L])
            for L in range(n)
        ])

    def _compute_D(self, separations: np.ndarray) -> np.ndarray:
        """
        D(L) = log S_global(L) − log S_local(L)
        S_local(L)  = δx(L) / δx(L + ΔL)   (short horizon)
        S_global(L) = δx(L) / δx(L + T)    (long horizon)
        """
        n = len(separations)
        dL = self.short_horizon
        T  = self.long_horizon
        D_vals = []

        for L in range(n):
            if L + T >= n or L + dL >= n:
                break
            dx_L    = separations[L] + 1e-10
            dx_Ldl  = separations[L + dL] + 1e-10
            dx_LT   = separations[L + T] + 1e-10

            log_S_local  = math.log(dx_L / dx_Ldl)
            log_S_global = math.log(dx_L / dx_LT)
            D_vals.append(log_S_global - log_S_local)

        return np.array(D_vals) if D_vals else np.array([0.0])

    def _fit_lambda(self, D_vals: np.ndarray) -> float:
        """
        Fit λ from D(L) = λ·(ΔL − T).
        Since ΔL − T = short_horizon − long_horizon is constant,
        λ = mean(D) / (ΔL − T).
        """
        delta = self.short_horizon - self.long_horizon  # negative
        if abs(delta) < 1e-8:
            return 0.0
        return float(np.mean(D_vals)) / delta

    def _estimate_tau(self, trajectory: np.ndarray) -> float:
        """Full pipeline: trajectory → δx → D → λ → τ."""
        separations = self._compute_separations(trajectory)
        D_vals      = self._compute_D(separations)
        lam         = self._fit_lambda(D_vals)
        return tau_from_lambda(lam)

    def score(self, trajectory: np.ndarray) -> dict:
        """
        Returns dict with:
          tau:     estimated manifold position
          lambda:  estimated eigenvalue
          blocked: True if τ < τ* (below Landauer threshold)
          snr:     τ / τ* (above 1.0 = stable, below 1.0 = injection)
        """
        n = min(trajectory.shape[0], self._n_layers)
        tau = self._estimate_tau(trajectory[:n])
        lam = lambda_from_tau(tau) if tau > 0 else 999.0
        blocked = tau < TAU_STAR
        snr = round(tau / TAU_STAR, 4)
        return {
            "tau":     round(tau, 6),
            "lambda":  round(lam, 6),
            "blocked": blocked,
            "snr":     snr,
        }


class SessionMonitor:
    """
    Applies D(t) stability scalar (Nine 2026b, Theorem 3) to inference
    request history rather than training steps.

    Each request contributes a FR distance δx(t) from the warmup centroid.
    D(t) is computed over a rolling window of these distances.

    D(t) = log S_global - log S_local
         = log δx(t-T) - log δx(t-Δt)

    Normal traffic:    distances stable     → D ≈ 0
    Injection campaign: distances escalating → D < 0 (diverging)

    This catches gradual social engineering campaigns that produce
    individually normal-looking FR distances but diverging trajectories.
    """

    def __init__(self, short_horizon: int = 2, long_horizon: int = 8,
                 min_history: int = 10):
        self.short_horizon = short_horizon
        self.long_horizon  = long_horizon
        self.min_history   = min_history
        self._history      = []   # rolling FR distances
        self._D_history    = []   # D(t) values
        self._threshold    = None
        self._warmup_D     = []

    def push(self, fr_distance: float) -> dict:
        """
        Add one FR distance to history and compute D(t).
        Returns dict with D, lambda_est, tau_est, blocked.
        """
        self._history.append(fr_distance)

        if len(self._history) < self.long_horizon + 1:
            return {"D": None, "tau": None, "lambda": None,
                    "session_blocked": False, "reason": "insufficient_history"}

        # Causal D(t): look back into history
        # S_local  = δx(t - Δt) / δx(t)  → log = log δx(t-Δt) - log δx(t)
        # S_global = δx(t - T)  / δx(t)  → log = log δx(t-T)  - log δx(t)
        # D = log S_global - log S_local
        #   = log δx(t-T) - log δx(t-Δt)
        dx_now   = self._history[-1]            + 1e-10
        dx_short = self._history[-self.short_horizon - 1] + 1e-10
        dx_long  = self._history[-self.long_horizon - 1]  + 1e-10

        D = math.log(dx_long) - math.log(dx_short)

        # Estimate λ from D(t) = λ(τ)·(Δt − T)
        # λ = D / (short_horizon - long_horizon)
        delta = self.short_horizon - self.long_horizon  # negative
        lam = D / delta if abs(delta) > 1e-8 else 0.0

        # Estimate τ = sqrt(3/(λ+2))
        denom = lam + 2.0
        tau = math.sqrt(3.0 / denom) if denom > 1e-8 else 0.0

        self._D_history.append(D)

        # Block if D < threshold (session diverging)
        threshold = self._threshold if self._threshold is not None else 0.0
        session_blocked = (len(self._D_history) >= self.min_history and
                           D < threshold)

        return {
            "D": round(D, 6),
            "tau": round(tau, 6),
            "lambda": round(lam, 6),
            "session_blocked": session_blocked,
            "threshold": round(threshold, 6),
            "history_len": len(self._history),
        }

    def calibrate(self, warmup_distances: list):
        """
        Compute D(t) over warmup distances to set normal-traffic baseline.
        Threshold = mean(D_warmup) - 2*std(D_warmup).
        """
        self._history = list(warmup_distances)
        D_vals = []
        for i in range(self.long_horizon, len(warmup_distances)):
            dx_short = warmup_distances[i - self.short_horizon] + 1e-10
            dx_long  = warmup_distances[i - self.long_horizon]  + 1e-10
            D_vals.append(math.log(dx_long) - math.log(dx_short))

        if D_vals:
            mu  = float(np.mean(D_vals))
            std = float(np.std(D_vals)) + 1e-8
            self._threshold = mu - 2.0 * std
            self._warmup_D  = D_vals
            print(f"    [session D(t)] warmup D: mean={mu:.4f} std={std:.4f} "
                  f"threshold={self._threshold:.4f}")
        self._history = []  # reset for live traffic

    def reset(self):
        self._history  = []
        self._D_history = []
