"""API module."""
# main-service.py for execute DDS (Drift Detection Service) service.

import logging
import sys
import json
import numpy as np
from detector import Detector
from litestar import Router, get, post
from litestar.status_codes import (
    HTTP_200_OK,
)
from schemas import (
    DetectorInputData,
    DistanceBasedResponse,
    HealthResponse,
)
from settings import api_settings, detector_settings

detector = Detector(
    settings=detector_settings,
)

class Variable:
    def __init__(self, d1,d2,d3):
        self.alpha = d1
        self.values = d2
        self.return_input_values=d3
        

def check_drift(data: DetectorInputData,out) -> DistanceBasedResponse:
     #async
    """Check if drift is present.

    :param data: input data
    :type data: DetectorInputData
    :return: drift data information
    :rtype: BaseCheckDriftResponse
    """
    
        
    logging.info("Checking drift...")
    check_drift_result = detector.check_drift(
        #The value parameter has to be in an array with the N embedding
        values=np.array(data['values']),
        alpha=data['alpha'],
    )
    return_input_values=data['return_input_values']
    value=np.array(data['values'])
    check_drift_result["distance"] = check_drift_result["distance"].distance
    if return_input_values:
        check_drift_result["values"] = value.tolist()  # noqa: PD011
    result = DistanceBasedResponse(
        **check_drift_result,
    )
    #Save the results of the algorithm in the output bucket (dds/output)
    f = open(str(out)+".txt", "a")
    f.write(str(result))
    f.close()

#Read json object with data from EMC service
with open(sys.argv[1]) as f:
    data = json.load(f)
print(data)

#Pass parameters (data and output bucket) to the drift detection algorithm
check_drift(data,sys.argv[2])

