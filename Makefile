.ONESHELL:
SHELL := /bin/bash

VENV=.venv

install:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	pip3 install --upgrade pip &&\
				 pip3 install -r api/requirements/requirements.txt \
				              -r api/requirements/tox_requirements.txt

tox:
	$(VENV)/bin/tox

build-data-drift-detector:
	docker build -t ifcacomputing/data-drift-detection-api detector_api

build-dimensionality-reduction:
	docker build -t ifcacomputing/dimensionality-reduction-api dimensionality_reduction_api

build-model-inference:
	docker build -t ifcacomputing/model-inference-api model_inference_api

run-data-drift-detector:
	docker run --name data-drift-detection -p 5001:8000 ifcacomputing/data-drift-detection-api

run-dimensionality-reduction:
	docker run --name dimensionality-reduction -p 5002:8000 ifcacomputing/dimensionality-reduction-api

run-model-inference:
	docker run --name model-inference -p 5003:8000 ifcacomputing/model-inference-api

push-data-drift-detector:
	docker push ifcacomputing/data-drift-detection-api

push-dimensionality-reduction:
	docker push ifcacomputing/dimensionality-reduction-api

push-model-inference:
	docker push ifcacomputing/model-inference-api
