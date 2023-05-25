"""Settings module."""

from pathlib import Path

from pydantic import BaseSettings


class APISettings(BaseSettings):
    """Settings class.

    Set variables to be used
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ML inference service"
    VERSION = "1.0.0"


class ModelSettings(BaseSettings):
    """Model settings class.

    Set detector variables to be used
    """

    FILE_PATH: Path = Path("objects/model.pt")


class TransformerSettings(BaseSettings):
    """Transformer settings class."""

    FILE_PATH: Path = Path("objects/transformer.pt")


api_settings = APISettings()
model_settings = ModelSettings()
transformer_settings = TransformerSettings()
