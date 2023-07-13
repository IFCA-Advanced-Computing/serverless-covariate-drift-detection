# main-service.py for execute DRS (Demensionality Reduction Service) service.

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

def dim_red(image, out):
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
    
    #Write algorithm results to output bucket
    f = open(str(out)+".json", "a")
    f.write(str(np.squeeze(reduced_image).tolist()))
    f.close()

#Capture the image from the input bucket (upload/input) and pass as parameters to the dimensionality
#reduction function the content of the image and the output bucket to put the results of the algorithm
img = Image.open(sys.argv[1])
dim_red(img,sys.argv[2])
