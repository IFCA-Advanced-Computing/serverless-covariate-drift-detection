"""Dimensionality reduction schemas."""

from io import BytesIO

from PIL import Image
from litestar.datastructures import UploadFile
from pydantic import BaseConfig, BaseModel, field_validator


class DimensionalityReductionInputData(BaseModel):
    """Dimensionality reduction input data class."""

    image: UploadFile

    @field_validator("image", mode="after")
    async def parse_image(cls, data: UploadFile) -> Image:
        """Parse image.

        :param data: data
        :type data: UploadFile
        :return: image
        :rtype: Image
        """
        data = await data.read()
        image = Image.open(
            fp=BytesIO(
                initial_bytes=data,
            ),
        )
        return image

    class Config(BaseConfig):
        arbitrary_types_allowed = True


class DimensionalityReductionResponse(BaseModel):
    """Dimensionality reduction response class."""

    reduced_image: list[float]
