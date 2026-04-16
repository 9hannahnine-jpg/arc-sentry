import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from arc_sentry_v2.models.base_adapter import BaseModelAdapter


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
       return delta  # raw -- scorer uses norm


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
   def find_best_layer(adapter: BaseModelAdapter,
                       warmup_prompts: List[str],
                       probe_prompts: List[str],
                       probe_system: Optional[str],
                       scan_start: float = 0.40) -> int:
       n = adapter.n_layers
       start = int(n * scan_start)
       best_layer, best_ratio = start, 0.0

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

           # Fisher-Rao geodesic — consistent with TAU_STAR noise floor
           ref = max(np.median([fisher_rao_dist(v, cn)
                                for v in warmup_vecs]), 1e-8)

           probe_scores = []
           for p in probe_prompts:
               h = adapter.extract_hidden(p, [layer-1, layer], probe_system)
               if (layer-1) in h and layer in h:
                   d = l2_normalize(h[layer] - h[layer-1])
                   probe_scores.append(fisher_rao_dist(d, cn))

           ratio = np.mean(probe_scores) / ref if probe_scores else 0
           if ratio > best_ratio:
               best_ratio = ratio
               best_layer = layer

       return best_layer
