"""Settings module."""

from pathlib import Path

from pydantic import BaseSettings


class APISettings(BaseSettings):
    """Settings class.

    Set variables to be used
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Data drift detection service"
    VERSION = "1.0.0"


class DetectorSettings(BaseSettings):
    """Detector settings class.

    Set detector variables to be used
    """

    FILE_PATH: Path = Path("detector/detector.pkl")


api_settings = APISettings()
detector_settings = DetectorSettings()
