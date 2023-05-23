"""Drift schemas."""

import numpy as np
from frouros.detectors.data_drift.batch.distance_based.base import DistanceResult
from pydantic import BaseModel, validator


class NoFoundResponse(BaseModel):
    """No detector found response class.

    Message showing that no detector was found
    """

    msg: str


class DetectorInputData(BaseModel):
    """Detector input data class.

    Set input detector values and parse them
    """

    alpha: float | None
    values: list[list[int | float]]
    return_input_values: bool = False

    @validator("values", pre=False)
    def parse_values(
        cls: "DetectorInputData",  # noqa: N805
        values: list[list[int | float]],
    ) -> np.ndarray:
        """Parse values to numpy array."""
        return np.array(values, dtype=float)


class BaseCheckDriftResponse(BaseModel):
    """Check drift response class.

    Ensures that the response has
    the defined format.
    """

    alpha: float
    datetime: str
    is_drift: bool
    p_value: float
    values: list[list[int | float]] | None = None


class DistanceBasedResponse(BaseCheckDriftResponse):
    """Check drift response class.

    Ensures that the response has
    the defined format.
    """

    distance: DistanceResult

    class Config:
        """Detector input class config."""

        arbitrary_types_allowed = True
