import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from arc_sentry.models.base_adapter import BaseModelAdapter


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def fisher_rao_dist(u, v):
    cos_sim = float(np.clip(np.dot(u, v), -1.0 + 1e-7, 1.0 - 1e-7))
    return float(np.arccos(cos_sim))


class BaseRepresentation(ABC):
    @abstractmethod
    def extract(self, adapter: BaseModelAdapter,
                prompt: str,
                system_prompt: Optional[str] = None) -> Optional[np.ndarray]:
        pass


class LayerDeltaNormRep(BaseRepresentation):
    """
    Δh norm at a single late layer.
    Best for injection / loud control-flow violations.
    """
    def __init__(self, layer: int):
        self.layer = layer

    def extract(self, adapter, prompt, system_prompt=None):
        hidden = adapter.extract_hidden(
            prompt, [self.layer - 1, self.layer], system_prompt)
        if (self.layer - 1) not in hidden or self.layer not in hidden:
            return None
        delta = hidden[self.layer] - hidden[self.layer - 1]
        return delta


class FirstTokenDeltaDirectionRep(BaseRepresentation):
    """
    L2-normalized Δh direction at a single late layer.
    Best for policy / behavioral drift.
    """
    def __init__(self, layer: int):
        self.layer = layer

    def extract(self, adapter, prompt, system_prompt=None):
        hidden = adapter.extract_hidden(
            prompt, [self.layer - 1, self.layer], system_prompt)
        if (self.layer - 1) not in hidden or self.layer not in hidden:
            return None
        delta = hidden[self.layer] - hidden[self.layer - 1]
        return l2_normalize(delta)


class LayerScanRep:
    """
    Utility to scan multiple layers and find best separation.
    Uses Fisher-Rao geodesic distance — consistent with TAU_STAR noise floor.
    Used during calibration only.
    """

    @staticmethod
    def _compute_layer_ratios(adapter, warmup_prompts, probe_prompts,
                               probe_system, scan_start=0.40):
        """Shared computation — returns dict of layer -> separation ratio."""
        n = adapter.n_layers
        start = int(n * scan_start)
        layer_ratios = {}

        for layer in range(start, n):
            warmup_vecs = []
            for p in warmup_prompts:
                h = adapter.extract_hidden(p, [layer-1, layer])
                if (layer-1) in h and layer in h:
                    d = h[layer] - h[layer-1]
                    warmup_vecs.append(l2_normalize(d))
            if not warmup_vecs:
                continue

            cn = l2_normalize(np.stack(warmup_vecs).mean(axis=0))
            ref = max(np.median([fisher_rao_dist(v, cn)
                                 for v in warmup_vecs]), 1e-8)

            probe_scores = []
            for p in probe_prompts:
                h = adapter.extract_hidden(p, [layer-1, layer], probe_system)
                if (layer-1) in h and layer in h:
                    d = l2_normalize(h[layer] - h[layer-1])
                    probe_scores.append(fisher_rao_dist(d, cn))

            ratio = np.mean(probe_scores) / ref if probe_scores else 0
            layer_ratios[layer] = ratio

        return layer_ratios

    @staticmethod
    def find_best_layer(adapter: BaseModelAdapter,
                        warmup_prompts: List[str],
                        probe_prompts: List[str],
                        probe_system: Optional[str],
                        scan_start: float = 0.40) -> int:
        """Find single best layer by Fisher-Rao separation ratio."""
        layer_ratios = LayerScanRep._compute_layer_ratios(
            adapter, warmup_prompts, probe_prompts, probe_system, scan_start)
        if not layer_ratios:
            return int(adapter.n_layers * 0.93)
        return max(layer_ratios, key=layer_ratios.get)

    @staticmethod
    def find_top_n_layers(adapter: BaseModelAdapter,
                          warmup_prompts: List[str],
                          probe_prompts: List[str],
                          probe_system: Optional[str],
                          n: int = 3,
                          scan_start: float = 0.40) -> List[int]:
        """
        Find top N layers by Fisher-Rao separation ratio.
        Returns list sorted ascending by layer index.
        Used for multi-layer injection detection.
        """
        layer_ratios = LayerScanRep._compute_layer_ratios(
            adapter, warmup_prompts, probe_prompts, probe_system, scan_start)
        if not layer_ratios:
            nl = adapter.n_layers
            return sorted([int(nl * 0.6), int(nl * 0.8), nl - 1])
        top = sorted(layer_ratios.items(), key=lambda x: -x[1])[:n]
        return sorted([layer for layer, _ in top])
