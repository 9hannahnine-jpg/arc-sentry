"""
Hallucination detector — output-side residual stream monitoring.
"""
import math
import numpy as np
from typing import List, Optional, Tuple
from arc_sentry_v2.models.base_adapter import BaseModelAdapter
from arc_sentry_v2.representations.trajectory import (
    TAU_STAR, l2_normalize,
    DtStabilityDetector,
)


class GenerationTrajectoryExtractor:
    def __init__(self, scan_start: float = 0.30):
        self.scan_start = scan_start

    def extract(self, adapter: BaseModelAdapter,
                prompt: str, response: str,
                system_prompt: Optional[str] = None):
        n = adapter.n_layers
        start = int(n * self.scan_start)
        layer_pairs = sorted(set(idx for L in range(start, n) for idx in [L-1, L]))
        combined = prompt + "\n" + response
        hidden = adapter.extract_hidden(combined, layer_pairs, system_prompt)
        trajectory = []
        for L in range(start, n):
            if (L-1) not in hidden or L not in hidden:
                continue
            delta = hidden[L] - hidden[L-1]
            trajectory.append(l2_normalize(delta))
        if not trajectory:
            return None
        return np.stack(trajectory)


class HallucinationDetector:
    def __init__(self, scan_start=0.30, short_horizon=2, long_horizon=8):
        self.extractor = GenerationTrajectoryExtractor(scan_start)
        self.detector = DtStabilityDetector(short_horizon, long_horizon)
        self._calibrated = False

    def calibrate(self, adapter, warmup_pairs, system_prompt=None):
        trajectories = []
        for prompt, response in warmup_pairs:
            traj = self.extractor.extract(adapter, prompt, response, system_prompt)
            if traj is not None:
                trajectories.append(traj)
        if len(trajectories) < 3:
            raise ValueError(f"Need >=3 valid warmup pairs, got {len(trajectories)}")
        self.detector.fit(trajectories)
        self._calibrated = True
        print(f"    [hallucination] calibrated on {len(trajectories)} pairs")

    def score(self, adapter, prompt, response, system_prompt=None):
        if not self._calibrated:
            return {"tau": None, "hallucinated": False, "snr": None, "reason": "not_calibrated"}
        traj = self.extractor.extract(adapter, prompt, response, system_prompt)
        if traj is None:
            return {"tau": None, "hallucinated": False, "snr": None, "reason": "extraction_failed"}
        result = self.detector.score(traj)
        return {
            "tau": result["tau"],
            "lambda": result["lambda"],
            "hallucinated": result["blocked"],
            "snr": result["snr"],
        }

    def generate_and_check(self, adapter, prompt, max_new_tokens=60, system_prompt=None):
        response = adapter.generate(prompt, system_prompt, max_new_tokens)
        score = self.score(adapter, prompt, response, system_prompt)
        return {"response": response, **score}
