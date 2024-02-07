.ONESHELL:
SHELL := /bin/bash

repository = ifcacomputing

add-multi-arch-builder:
	docker run --privileged --rm tonistiigi/binfmt --install all
	docker buildx create --name drift-detection-builder --driver docker-container --bootstrap
	docker buildx use drift-detection-builder

remove-multi-arch-builder:
	docker buildx rm drift-detection-builder

build-data-drift-detector:
	for dataset in mnist fashion_mnist cifar10;
	do
		docker buildx build --platform linux/amd64,linux/arm64 -t $(repository)/data-drift-detection-api-$${dataset} --build-arg DATASET=$${dataset} --build-arg PARENT_DIR=detector_api -f detector_api/Dockerfile --push .
	done

build-dimensionality-reduction:
	for dataset in mnist fashion_mnist cifar10;
	do
		docker buildx build --platform linux/amd64,linux/arm64 -t $(repository)/dimensionality-reduction-api-$${dataset} --build-arg DATASET=$${dataset} --build-arg PARENT_DIR=dimensionality_reduction_api -f dimensionality_reduction_api/Dockerfile --push .
	done

build-model-inference:
	for dataset in mnist fashion_mnist cifar10;
	do
		docker buildx build --platform linux/amd64,linux/arm64 -t $(repository)/model-inference-api-$${dataset} --build-arg DATASET=$${dataset} --build-arg PARENT_DIR=model_inference_api -f model_inference_api/Dockerfile --push .
	done

run-data-drift-detector-mnist:
	docker run --name data-drift-detection-mnist -p 5001:8000 $(repository)/data-drift-detection-api-mnist

run-data-drift-detector-fashion_mnist:
	docker run --name data-drift-detection-fashion_mnist -p 5001:8000 $(repository)/data-drift-detection-api-fashion_mnist

run-data-drift-detector-cifar10:
	docker run --name data-drift-detection-cifar10 -p 5001:8000 $(repository)/data-drift-detection-api-cifar10

run-dimensionality-reduction-mnist:
	docker run --name dimensionality-reduction-mnist -p 5002:8000 $(repository)/dimensionality-reduction-api-mnist

run-dimensionality-reduction-fashion_mnist:
	docker run --name dimensionality-reduction-fashion_mnist -p 5002:8000 $(repository)/dimensionality-reduction-api-fashion_mnist

run-dimensionality-reduction-cifar10:
	docker run --name dimensionality-reduction-cifar10 -p 5002:8000 $(repository)/dimensionality-reduction-api-cifar10

run-model-inference-mnist:
	docker run --name model-inference-mnist -p 5003:8000 $(repository)/model-inference-api-mnist

run-model-inference-fashion_mnist:
	docker run --name model-inference-fashion_mnist -p 5003:8000 $(repository)/model-inference-api-fashion_mnist

run-model-inference-cifar10:
	docker run --name model-inference-cifar10 -p 5003:8000 $(repository)/model-inference-api-cifar10
