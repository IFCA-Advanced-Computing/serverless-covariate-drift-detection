"""Schemas __init__."""

from .model import (
    ModelInputData,
    PredictResponse,
)
from .health import HealthResponse

__all__ = [
    "HealthResponse",
    "ModelInputData",
    "PredictResponse",
]
