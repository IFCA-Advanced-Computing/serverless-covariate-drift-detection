# Efficient and scalable covariate drift detection in machine learning systems with serverless computing

This repository contains the code to reproduce the proposed serverless architecture for covariate drift detection of the paper [Efficient and scalable covariate drift detection in machine learning systems with serverless computing](https://doi.org/10.1016/j.future.2024.07.010).

## Requirements

The code has been tested with the following environment:

- **OS**: Ubuntu 22.04.03 LTS
- **Python**: Version 3.10

## Installation

Create a virtual environment and install the required packages for training and testing the models:
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt
```

## Training

Run the following commands for each dataset to train models and generate the necessary files:

- **MNIST**
```bash
python ml/training.py -d MNIST
```

- **Fashion MNIST**
```bash
python ml/training.py -d FashionMNIST
```

- **CIFAR-10**
```bash
python ml/training.py -d CIFAR10
```

For each dataset, this will create a `cnn.pt` (inference model), `detector.pkl` (covariate drift detector), `encoder.pt` (dimensionality reduction model) and `transformer.pt` (preprocessing object) files in the `ml/objects` directory.

In addition, `cnn.pt` and `transformer.pt` are stored in the `model_inference_api/app/objects` directory. `encoder.pt` and `transformer.pt` are stored in the `dimensionality_reduction_api/app/objects` directory. `detector.pkl` is stored in the `detector_api/app/objects` directory.

## Testing

Evaluate the models to replicate paper results:

- **MNIST**
```bash
python ml/mnist/testing.py -d MNIST
```

- **Fashion MNIST**
```bash
python ml/mnist/testing.py -d FashionMNIST
```

- **CIFAR-10**
```bash
python ml/mnist/testing.py -d CIFAR10
```

**_NOTE:_**
The results may vary slightly due to the randomness and hardware differences even when using the same seed. See [PyTorch reproducibility](https://pytorch.org/docs/2.1/notes/randomness.html#reproducibility).


## Architecture overview

The architecture comprises three key services:

- **ML Inference Service (MLIS)**: Handles model inference requests.
- **Dimensionality Reduction Service (DRS)**: Processes data for dimensionality reduction.
- **Drift Detection Service (DDS)**: Detects covariate drift in data batches.

![alt architecture](./images/architecture.png)


### Building and pushing service images

**_NOTE:_**
Ensure models are trained before proceeding. See the [Training](#training) section.
Set `registry_name` in `Makefile` to your registry.

To allow to build the services images for multiple architectures (linux/arm64 and linux/amd64), we use [buildx](https://docs.docker.com/buildx/working-with-buildx/). To enable it, run the following command:
```bash
make add-multi-arch-builder
```

Build and push the services images with the following commands:

- **ML Inference Service (MLIS)**
```bash
make build-model-inference
``` 

- **Dimensionality Reduction Service (DRS)**
```bash
make build-dimensionality-reduction
``` 

- **Drift Detection Service (DDS)**
```bash
make build-covariate-drift-detector
```


## Citation

Please cite our paper if you use this code or architecture:

```@article{CESPEDESSISNIEGA2024174,
title = {Efficient and scalable covariate drift detection in machine learning systems with serverless computing},
journal = {Future Generation Computer Systems},
volume = {161},
pages = {174-188},
year = {2024},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2024.07.010},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X24003716},
author = {Jaime {Céspedes Sisniega} and Vicente Rodríguez and Germán Moltó and Álvaro {López García}},
keywords = {Serverless computing, Machine learning, Drift detection, Covariate drift, Covariate shift},
abstract = {As machine learning models are increasingly deployed in production, robust monitoring and detection of concept and covariate drift become critical. This paper addresses the gap in the widespread adoption of drift detection techniques by proposing a serverless-based approach for batch covariate drift detection in ML systems. Leveraging the open-source OSCAR framework and the open-source Frouros drift detection library, we develop a set of services that enable parallel execution of two key components: the ML inference pipeline and the batch covariate drift detection pipeline. To this end, our proposal takes advantage of the elasticity and efficiency of serverless computing for ML pipelines, including scalability, cost-effectiveness, and seamless integration with existing infrastructure. We evaluate this approach through an edge ML use case, showcasing its operation on a simulated batch covariate drift scenario. Our research highlights the importance of integrating drift detection as a fundamental requirement in developing robust and trustworthy AI systems and encourages the adoption of these techniques in ML deployment pipelines. In this way, organizations can proactively identify and mitigate the adverse effects of covariate drift while capitalizing on the benefits offered by serverless computing.}
}
```