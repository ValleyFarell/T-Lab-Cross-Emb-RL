from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class EpisodeResult:
    scenario_id: str
    method: str
    seed: int
    success: bool
    steps: int
    path_length: float
    final_distance: float
    observations: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    intentions: np.ndarray
    diagnostics: dict[str, Any]
