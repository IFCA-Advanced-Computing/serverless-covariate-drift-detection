from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from ml.mnist.utils.dataset import CustomMNIST


def make_transformed_dataset(
        subset: Dataset,
        transform: torch.nn.Module | torchvision.transforms.Compose | None = None,
) -> Dataset:
    return CustomMNIST(
        subset=subset,
        transform=transform,
    )


def save_images(data_loader, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    batch_size = data_loader.batch_size
    for i, batch in enumerate(data_loader):
        batch = (np.squeeze(batch[0].numpy(), axis=1) * 255).astype("uint8")
        for image, file_path in zip(batch, data_loader.dataset.samples[i*batch_size: i*batch_size + batch_size]):
            file_image_path = Path(target_dir, str(file_path[1]))
            file_image_path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(Path(file_image_path, f"{Path(file_path[0]).name}"))


transformations = [
    (
        "GaussianBlur(kernel_size=(5, 9), sigma=0.25)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=0.25),
            ],
        ),
    ),
    (
        "GaussianBlur(kernel_size=(5, 9), sigma=0.5)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=0.5),
            ],
        ),
    ),
    (
        "GaussianBlur(kernel_size=(5, 9), sigma=1.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=1.0),
            ],
        ),
    ),
    (
        "GaussianBlur(kernel_size=(5, 9), sigma=2.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=2.0),
            ],
        ),
    ),
    (
        "GaussianBlur(kernel_size=(5, 9), sigma=4.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.GaussianBlur(
                    kernel_size=(5, 9),
                    sigma=4.0),
            ],
        ),
    ),
    (
        "ElasticTransform(alpha=12.5, sigma=5.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.ElasticTransform(
                    alpha=12.5,
                    sigma=5.0,
                ),
            ],
        ),
    ),
    (
        "ElasticTransform(alpha=25.0, sigma=5.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.ElasticTransform(
                    alpha=25.0,
                    sigma=5.0,
                ),
            ],
        ),
    ),
    (
        "ElasticTransform(alpha=50.0, sigma=5.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.ElasticTransform(
                    alpha=50.0,
                    sigma=5.0,
                ),
            ],
        ),
    ),
    (
        "ElasticTransform(alpha=100.0, sigma=5.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.ElasticTransform(
                    alpha=100.0,
                    sigma=5.0,
                ),
            ],
        ),
    ),
    (
        "ElasticTransform(alpha=200.0, sigma=5.0)",
        torchvision.transforms.Compose(
            [
                torchvision.transforms.ElasticTransform(
                    alpha=200.0,
                    sigma=5.0,
                ),
            ],
        ),
    ),
]
