"""Main module."""

from litestar import Litestar
from litestar.app import OpenAPIConfig
from litestar.logging import LoggingConfig
from litestar.openapi import OpenAPIController

from api import api_router
from settings import api_settings


class CustomOpenAPIController(OpenAPIController):
    """Custom OpenAPI controller."""

    path = "/docs"


openapi_config = OpenAPIConfig(
    title=api_settings.PROJECT_NAME,
    version=api_settings.VERSION,
    openapi_controller=CustomOpenAPIController,
)

logging_config = LoggingConfig(
    loggers={
        "app": {
            "level": "INFO",
            "handlers": ["queue_listener"],
        },
    },
)

app = Litestar(
    route_handlers=[api_router],
    openapi_config=openapi_config,
    logging_config=logging_config,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=5002,
        log_level="debug",
    )
