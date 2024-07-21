"""Settings module."""

from pathlib import Path

from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """Settings class.

    Set variables to be used
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML inference service"
    VERSION: str = "1.0.0"


class ModelSettings(BaseSettings):
    """Model settings class.

    Set detector variables to be used
    """

    FILE_PATH: Path = Path("objects/cnn.pt")


class TransformSettings(BaseSettings):
    """Transform settings class."""

    FILE_PATH: Path = Path("objects/transform.pt")


api_settings = APISettings()
model_settings = ModelSettings()
transform_settings = TransformSettings()
