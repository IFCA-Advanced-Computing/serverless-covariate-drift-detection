"""Settings module."""

from pathlib import Path

from pydantic import BaseSettings


class APISettings(BaseSettings):
    """Settings class.

    Set variables to be used
    """

    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Dimensionality reduction service"
    VERSION = "1.0.0"


class EncoderSettings(BaseSettings):
    """Encoder settings class."""

    FILE_PATH: Path = Path("objects/encoder.pt")


class TransformerSettings(BaseSettings):
    """Transformer settings class."""

    FILE_PATH: Path = Path("objects/transformer.pt")


api_settings = APISettings()
encoder_settings = EncoderSettings()
transformer_settings = TransformerSettings()
