import torchvision


DATASETS = {
    "MNIST": {
        "name": torchvision.datasets.MNIST,
        "num_classes": 10,
        "input_channels": 1,
        "cmap": "gray",
    },
    "FashionMNIST": {
        "name": torchvision.datasets.FashionMNIST,
        "num_classes": 10,
        "input_channels": 1,
        "cmap": "gray",
    },
    "CIFAR10": {
        "name": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "input_channels": 3,
        "cmap": None,
    },
}

IMAGE_SIZE = (28, 28)
