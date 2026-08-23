"""Офлайн-проверка геометрии оценок замороженного FB-критика."""

from .analysis import affine_probe, regression_metrics
from .data import PairSplit, split_state_indices
from .models import TrainingConfig, fit_value_model

__all__ = [
    "PairSplit",
    "TrainingConfig",
    "affine_probe",
    "fit_value_model",
    "regression_metrics",
    "split_state_indices",
]
