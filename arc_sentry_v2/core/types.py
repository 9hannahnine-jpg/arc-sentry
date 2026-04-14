
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class Observation:
    model_id: str
    route_id: str
    prompt: str
    system_prompt: Optional[str]
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineProfile:
    baseline_id: str
    model_id: str
    route_id: str
    representation: str
    baseline_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    score: float
    label: str          # "allow", "flag", "block"
    reason: str
    detector: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    fired_by: List[str] = field(default_factory=list)
    blocked: bool = False
