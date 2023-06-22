from typing import Dict
import gzip
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
from tqdm import tqdm


resources = {
    "train": {
        "images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "number": 60000,
    },
    "test": {
        "images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "number": 10000,
    },
}


def download(target_dir: str, url: str) -> Path:
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(target_dir, urlparse(url).path.split("/")[-1])
    urllib.request.urlretrieve(
        url=url,
        filename=filename,
    )
    return filename


def extract(filename_images: Path, filename_labels: Path, target_dir: Path, number: int) -> None:
    with gzip.open(filename=filename_images, mode="rb") as f:
        f.read(16)  # skip header
        size = 28
        buf = f.read(size * size * number)
        data = np.frombuffer(buf, dtype=np.uint8)
        images = data.reshape(number, size, size)
        target_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(filename=filename_labels, mode="rb") as f:
        f.read(8)  # skip header
        size = 1
        buf = f.read(size * size * number)
        labels = np.frombuffer(buf, dtype=np.uint8)

    for i, (image, label) in tqdm(enumerate(zip(images, labels))):
        target_dir_image = Path(target_dir, str(label))
        target_dir_image.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(f"{target_dir_image}/{i}.png")


def get_resource(target_dir: str, resource: Dict[str, str], number: int, name: str) -> None:
    filename_images = download(
        target_dir=target_dir,
        url=resource["images"],
    )
    filename_labels = download(
        target_dir=target_dir,
        url=resource["labels"],
    )
    extract(
        filename_images=filename_images,
        filename_labels=filename_labels,
        target_dir=Path(target_dir, name),
        number=number,
    )


if __name__ == '__main__':
    target_dir = "data"

    for name, resource in resources.items():
        get_resource(
            target_dir=target_dir,
            resource=resource,
            number=resource["number"],
            name=name,
        )
