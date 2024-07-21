import torchvision


DATASETS = {
    "MNIST": {
        "name": torchvision.datasets.MNIST,
        "num_classes": 10,
        "input_channels": 1,
        "cmap": "gray",
        "lowercase": "mnist",
    },
    "FashionMNIST": {
        "name": torchvision.datasets.FashionMNIST,
        "num_classes": 10,
        "input_channels": 1,
        "cmap": "gray",
        "lowercase": "fashion_mnist",
    },
    "CIFAR10": {
        "name": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "input_channels": 3,
        "cmap": None,
        "lowercase": "cifar10",
    },
}

IMAGE_SIZE = (28, 28)
