"""API module."""

import logging

import numpy as np
from litestar import Router, get, post
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
)

from dr import DimensionalityReduction
from schemas import (
    DimensionalityReductionInputData,
    DimensionalityReductionResponse,
    HealthResponse,
)
from settings import (
    api_settings,
    encoder_settings,
    transformer_settings,
)

dr = DimensionalityReduction(
    settings_encoder=encoder_settings,
    settings_transformer=transformer_settings,
)


@post(
    path="/dimensionality_reduction",
    status_code=HTTP_200_OK,
)
async def dimensionality_reduction(
        data: DimensionalityReductionInputData = Body(
            media_type=RequestEncodingType.MULTI_PART,
        ),
) -> DimensionalityReductionResponse:
    """Reduce image.

    :param data: input data
    :type data: EncoderInputData
    :return: reduced image
    :rtype: BaseDimensionalityReductionResponse
    """
    image = await data.image
    logging.info("Transforming image...")
    transformed_image = dr.transform(
        data=image,
    )
    logging.info("Image transformed.")
    logging.info("Encoding image...")
    reduced_image = dr.encode(
        data=transformed_image,
    )
    logging.info("Image encoded.")
    return DimensionalityReductionResponse(
        reduced_image=np.squeeze(reduced_image).tolist(),
    )


@get(
    path="/health",
    status_code=HTTP_200_OK,
)
async def health() -> HealthResponse:
    """Health check function.

    :return: Health check response
    :rtype: HealthResponse
    """
    return HealthResponse(
        name=api_settings.PROJECT_NAME,
        api_version=api_settings.VERSION,
    )


api_router = Router(
    path="/api",
    tags=["API"],
    route_handlers=[
        dimensionality_reduction,
        health,
    ],
)
