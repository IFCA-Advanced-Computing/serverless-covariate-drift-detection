"""Detector module."""

import datetime
import logging
import pickle
from pathlib import Path

import numpy as np
from frouros.detectors.base import DetectorBase
from settings import DetectorSettings
from utils import SingletonMeta


class Detector(metaclass=SingletonMeta):
    """Detector class."""

    def __init__(self: "Detector") -> None:
        """Init method."""
        self.detector = None

    def load(self: "Detector", settings: DetectorSettings) -> DetectorBase:
        """Load drift detector.

        :return detector metadata
        :rtype: dict[str, str | None]
        """
        logging.info("Loading drift detector...")
        self.detector = self._load(settings=settings)
        logging.info("Drift detector loaded.")
        return self

    @staticmethod
    def _load(settings: DetectorSettings) -> DetectorBase:
        with Path.open(settings.FILE_PATH, mode="rb") as f:
            data = f.read()
        return pickle.loads(data)  # noqa: S301

    def check_drift(
        self: "Detector",
        values: np.ndarray,
        alpha: float = 0.05,
    ) -> dict[str, str | bool | float | np.ndarray]:
        """Check if drift is present.

        :param values: input sample
        :type values: numpy.ndarray
        :param alpha: significance level, defaults to 0.05
        :type alpha: float, optional
        :return: drift data information
        :rtype: dict[str, str]
        """
        distance, callback_logs = self.detector.compare(X=values)

        return {
            "alpha": alpha,
            "datetime": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%d/%m/%Y %H:%M:%S.%f",
            ),
            "distance": distance,
            "is_drift": callback_logs["permutation_test"]["p_value"] < alpha,
            "p_value": callback_logs["permutation_test"]["p_value"],
        }


def load_detector(settings: DetectorSettings) -> Detector:
    """Load drift detector.

    :param settings: detector settings
    :type settings: DetectorSettings
    :return detector
    :rtype: Detector
    """
    return Detector().load(settings=settings)
