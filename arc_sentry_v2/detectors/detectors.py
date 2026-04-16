import numpy as np
from typing import List, Optional, Dict
from arc_sentry_v2.core.types import DetectionResult
from arc_sentry_v2.baselines.baselines import BaseBaseline


class OneSidedDetector:
    """Block if score > upper threshold. Used for injection."""

    def __init__(self, baseline: BaseBaseline, name: str = "injection"):
        self.baseline = baseline
        self.name = name

    def detect(self, vector: np.ndarray) -> DetectionResult:
        score = self.baseline.score(vector)
        fired = score > self.baseline.threshold()
        return DetectionResult(
            score=round(score, 6),
            label="block" if fired else "allow",
            reason=f"high-side deviation on {self.name}" if fired else "normal",
            detector=self.name,
            fired_by=[self.name] if fired else [],
            blocked=fired)


class TwoSidedDetector:
    """
    Flag high-side (injection) and low-side (policy drift).
    Used for behavioral state monitoring.
    """

    def __init__(self, baseline: BaseBaseline,
                 upper_factor: float = 2.0,
                 lower_factor: float = 0.3,
                 name: str = "behavior"):
        self.baseline = baseline
        self.upper = baseline.threshold()
        self.lower = baseline.threshold() * lower_factor
        self.name = name

    def detect(self, vector: np.ndarray) -> DetectionResult:
        score = self.baseline.score(vector)
        if score > self.upper:
            return DetectionResult(score=round(score,6), label="block",
                reason="high-side deviation (injection-like)",
                detector=self.name, fired_by=[self.name], blocked=True)
        elif score < self.lower:
            return DetectionResult(score=round(score,6), label="flag",
                reason="low-side deviation (policy-drift-like)",
                detector=self.name, fired_by=[self.name], blocked=False)
        return DetectionResult(score=round(score,6), label="allow",
            reason="normal", detector=self.name, blocked=False)


class MultiLayerInjectionDetector:
    """
    Runs injection detection across multiple residual stream layers simultaneously.
    Blocks if ANY layer fires (OR logic).

    Each layer gets its own baseline calibrated on warmup + probe vectors.
    Different attack types localize at different depths:
      - Direct injection:  ~93% depth (late layers)
      - Subtle/indirect:   may localize at earlier layers

    Using multiple layers catches attacks that are invisible at any single layer.
    """

    def __init__(self, layer_detectors: Dict[int, OneSidedDetector]):
        """
        Args:
            layer_detectors: dict mapping layer_index -> OneSidedDetector
        """
        self.layer_detectors = layer_detectors
        self.name = "multi_layer_injection"

    def detect(self, layer_vectors: Dict[int, np.ndarray]) -> DetectionResult:
        """
        Args:
            layer_vectors: dict mapping layer_index -> extracted vector
        """
        fired_layers = []
        all_scores = {}

        for layer_idx, det in self.layer_detectors.items():
            if layer_idx not in layer_vectors:
                continue
            r = det.detect(layer_vectors[layer_idx])
            all_scores[f"L{layer_idx}"] = r.score
            if r.blocked:
                fired_layers.append(f"L{layer_idx}")

        blocked = len(fired_layers) > 0  # OR logic — any layer fires = block
        max_score = max(all_scores.values()) if all_scores else 0.0

        return DetectionResult(
            score=round(max_score, 6),
            label="block" if blocked else "allow",
            reason=f"fired at layers: {fired_layers}" if fired_layers else "normal",
            detector=self.name,
            fired_by=fired_layers,
            evidence=all_scores,
            blocked=blocked)


class EnsembleDetector:
    """
    Combines multiple detectors.
    Blocks if ANY injection detector fires.
    Flags if ANY drift detector fires.
    """

    def __init__(self, injection_detectors: List, drift_detectors: List):
        self.injection_detectors = injection_detectors
        self.drift_detectors = drift_detectors

    def detect(self, vectors: dict) -> DetectionResult:
        fired_by = []
        all_scores = {}

        for det in self.injection_detectors:
            # MultiLayerInjectionDetector takes a layer_vectors dict
            if isinstance(det, MultiLayerInjectionDetector):
                layer_vecs = {k: v for k, v in vectors.items()
                              if isinstance(k, int)}
                if layer_vecs:
                    r = det.detect(layer_vecs)
                    all_scores.update(r.evidence or {})
                    if r.blocked:
                        fired_by.extend(r.fired_by)
            else:
                if det.name in vectors:
                    r = det.detect(vectors[det.name])
                    all_scores[det.name] = r.score
                    if r.blocked:
                        fired_by.append(det.name)

        for det in self.drift_detectors:
            if det.name in vectors:
                r = det.detect(vectors[det.name])
                all_scores[det.name] = r.score
                if r.label == "flag":
                    fired_by.append(f"{det.name}_drift")

        blocked = len(fired_by) > 0 and any(
            f.startswith("L") or f in [d.name for d in self.injection_detectors
                                        if not isinstance(d, MultiLayerInjectionDetector)]
            for f in fired_by)
        label = "block" if blocked else ("flag" if fired_by else "allow")

        return DetectionResult(
            score=round(max(all_scores.values()) if all_scores else 0, 6),
            label=label,
            reason=f"fired: {fired_by}" if fired_by else "normal",
            detector="ensemble",
            fired_by=fired_by,
            evidence=all_scores,
            blocked=blocked)
