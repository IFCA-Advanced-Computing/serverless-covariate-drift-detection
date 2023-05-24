"""Dimensionality reduction schemas."""

from io import BytesIO

from PIL import Image, ImageOps
from litestar.datastructures import UploadFile
from pydantic import BaseConfig, BaseModel, validator


class DimensionalityReductionInputData(BaseModel):
    """Dimensionality reduction input data class."""

    image: UploadFile

    @validator("image", pre=False)
    async def parse_image(cls, data: UploadFile) -> Image:
        """Parse image.

        :param data: data
        :type data: UploadFile
        :return: image
        :rtype: Image
        """
        data = await data.read()
        image = ImageOps.grayscale(
            image=Image.open(
                fp=BytesIO(
                    initial_bytes=data,
                ),
            ),
        ).resize((28, 28))
        return image

    class Config(BaseConfig):
        arbitrary_types_allowed = True


class DimensionalityReductionResponse(BaseModel):
    """Dimensionality reduction response class."""

    reduced_image: list[float]
