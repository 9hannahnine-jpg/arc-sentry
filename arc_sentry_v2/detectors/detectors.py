
import numpy as np
from typing import List, Optional
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

        blocked = any(d.name in fired_by for d in self.injection_detectors)
        label = "block" if blocked else ("flag" if fired_by else "allow")

        return DetectionResult(
            score=round(max(all_scores.values()) if all_scores else 0, 6),
            label=label,
            reason=f"fired: {fired_by}" if fired_by else "normal",
            detector="ensemble",
            fired_by=fired_by,
            evidence=all_scores,
            blocked=blocked)
