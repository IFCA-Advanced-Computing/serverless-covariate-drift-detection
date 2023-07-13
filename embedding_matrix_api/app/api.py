"""API module."""

import logging

from litestar import Router, get, post
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
)

from model import Model
from schemas import (
    HealthResponse,
    ModelInputData,
    PredictResponse,
)
from settings import (
    api_settings,
    model_settings,
    transformer_settings,
)

model = Model(
    settings_model=model_settings,
    settings_transformer=transformer_settings,
)


@post(
    path="/predict",
    status_code=HTTP_200_OK,
)
async def predict(
        data: ModelInputData = Body(
            media_type=RequestEncodingType.MULTI_PART,
        ),
) -> PredictResponse:
    """Predict function.

    :param data: model input data
    :type data: ModelInputData
    :return: prediction response
    :rtype: BasePredictionResponse
    """
    image = await data.image
    logging.info("Transforming image...")
    transformed_image = model.transform(
        data=image,
    )
    logging.info("Image transformed.")
    logging.info("Predicting...")
    prediction = model.predict(
        data=transformed_image,
    )
    logging.info("Prediction made.")
    return PredictResponse(
        **prediction,
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
        predict,
        health,
    ],
)
