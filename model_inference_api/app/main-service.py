"""API module."""

import logging
import sys
import numpy as np
from PIL import Image
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

def predict(
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
    image = data
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
    

#upload image from bucket
img = Image.open(sys.argv[1])

#execution of inference model using the predict function
result=predict(img)

#save the results to a txt file
f = open(str(sys.argv[2])+".txt", "a")
f.write(str(result))
f.close()
