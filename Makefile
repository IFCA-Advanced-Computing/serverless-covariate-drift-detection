#multi-platform execution of services
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
	docker build -t ghcr.io/grycap/mls-arm-api model_inference_api
	docker push ghcr.io/grycap/mls-arm-api
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/grycap/mls-arm-api --push model_inference_api

run-data-drift-detector:
	docker run --name data-drift-detection -p 5001:8000 ifcacomputing/data-drift-detection-api

run-dimensionality-reduction:
	docker run --name dimensionality-reduction -p 5002:8000 ifcacomputing/dimensionality-reduction-api

run-model-inference:
	docker run --name model-inference -p 5003:8000 ifcacomputing/model-inference-api
    
mls:
	docker build -t ghcr.io/grycap/mls-arm-api model_inference_api
	docker push ghcr.io/grycap/mls-arm-api
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/grycap/mls-arm-api --push model_inference_api
	    
dds:
	docker build -t ghcr.io/grycap/dds-arm-api detector_api
	docker push ghcr.io/grycap/dds-arm-api
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/grycap/dds-arm-api --push detector_api
	    
emc:
	docker build -t ghcr.io/grycap/emc-arm-api embedding_matrix
	docker push ghcr.io/grycap/emc-arm-api
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/grycap/emc-arm-api --push embedding_matrix
drs:
	docker build -t ghcr.io/grycap/drs-arm-api dimensionality_reduction_api
	docker push ghcr.io/grycap/drs-arm-api
	docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/grycap/drs-arm-api --push dimensionality_reduction_api
	
