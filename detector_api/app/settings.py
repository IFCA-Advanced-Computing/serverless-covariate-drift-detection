"""Settings module."""

from pathlib import Path

from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """Settings class.

    Set variables to be used
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Covariate drift detection service"
    VERSION: str = "1.0.0"


class DetectorSettings(BaseSettings):
    """Detector settings class.

    Set detector variables to be used
    """

    FILE_PATH: Path = Path("objects/detector.pkl")


api_settings = APISettings()
detector_settings = DetectorSettings()
