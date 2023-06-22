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

add-multi-arch-builder:
	docker run --privileged --rm tonistiigi/binfmt --install all
	docker buildx create --name drift-detection-builder --driver docker-container --bootstrap
	docker buildx use drift-detection-builder

remove-multi-arch-builder:
	docker buildx rm drift-detection-builder

build-push-data-drift-detector:
	docker buildx build --platform linux/amd64,linux/arm64 -t ifcacomputing/data-drift-detection-api --push detector_api

build-push-dimensionality-reduction:
	docker buildx build --platform linux/amd64,linux/arm64 -t ifcacomputing/dimensionality-reduction-api --push dimensionality_reduction_api

build-push-model-inference:
	docker buildx build --platform linux/amd64,linux/arm64 -t ifcacomputing/model-inference-api:latest --push model_inference_api

run-data-drift-detector:
	docker run --name data-drift-detection -p 5001:8000 ifcacomputing/data-drift-detection-api

run-dimensionality-reduction:
	docker run --name dimensionality-reduction -p 5002:8000 ifcacomputing/dimensionality-reduction-api

run-model-inference:
	docker run --name model-inference -p 5003:8000 ifcacomputing/model-inference-api