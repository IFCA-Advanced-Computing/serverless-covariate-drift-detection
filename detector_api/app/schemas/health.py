"""Health schema."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health schema."""

    name: str
    api_version: str
