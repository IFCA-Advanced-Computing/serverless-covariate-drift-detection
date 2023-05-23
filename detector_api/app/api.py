"""API module."""

import logging

from detector import load_detector
from litestar import Router, get, post
from litestar.status_codes import (
    HTTP_200_OK,
)
from schemas import (
    BaseCheckDriftResponse,
    DetectorInputData,
    HealthResponse,
)
from settings import api_settings, detector_settings

detector = load_detector(settings=detector_settings)


@post(
    path="/check_drift",
    status_code=HTTP_200_OK,
)
async def check_drift(data: DetectorInputData) -> BaseCheckDriftResponse:
    """Check if drift is present.

    :param data: input data
    :type data: DetectorInputData
    :return: drift data information
    :rtype: BaseCheckDriftResponse
    """
    logging.info("Checking drift...")
    check_drift_result = detector.check_drift(
        values=data.values,
        alpha=data.alpha,
    )
    if data.return_input_values:
        check_drift_result["values"] = data.values.tolist()  # noqa: PD011

    return BaseCheckDriftResponse(
        **check_drift_result,
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
        check_drift,
        health,
    ],
)
