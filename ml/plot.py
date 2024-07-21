import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from utils.constants import DATASETS, IMAGE_SIZE
from utils.drift import make_transformed_dataset, transformations

# Set the font size for the plots
plt.rcParams.update({"font.size": 20})


# Set pytorch seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seed(seed=31)

for dataset in DATASETS:
    transform_steps = (
        [torchvision.transforms.Grayscale()]
        if DATASETS[dataset]["input_channels"] == 1
        else []
    )
    transform = torchvision.transforms.Compose(
        transform_steps
        + [
            torchvision.transforms.Resize(size=IMAGE_SIZE),
            torchvision.transforms.ToTensor(),  # Convert images to the range [0.0, 1.0] (normalize)
        ],
    )

    test_dataset = DATASETS[dataset]["name"](
        root="/tmp/test",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
    )

    fig, ax = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(10, 5),
        dpi=300,
    )

    # Print the first occurrence of each class
    for batch in test_dataloader:
        for i, class_name in enumerate(test_dataset.classes):
            class_idx = (batch[1] == i).nonzero(as_tuple=False)
            ax[i // 5, i % 5].imshow(
                batch[0][class_idx[0].item()].permute(1, 2, 0),
                cmap=DATASETS[dataset]["cmap"],
            )
            ax[i // 5, i % 5].set_title(f"{class_name}")
            ax[i // 5, i % 5].axis("off")
        break

    plt.tight_layout()
    # Save the plot
    fig_output_path = Path(f"ml/plots/{dataset.lower()}-reference.png")
    # Create the plots directory if it does not exist
    fig_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"ml/plots/{dataset.lower()}-reference.png", dpi=300, bbox_inches="tight")

    for type_, transformation in transformations:
        transformed_dataset = make_transformed_dataset(
            subset=test_dataset,
            transform=transformation,
        )
        transformed_dataloader = torch.utils.data.DataLoader(
            dataset=transformed_dataset,
            batch_size=128,
            shuffle=False,
        )

        fig, ax = plt.subplots(
            nrows=2,
            ncols=5,
            figsize=(10, 5),
            dpi=300,
        )

        # Print the first occurrence of each class
        for batch in transformed_dataloader:
            for i, class_name in enumerate(test_dataset.classes):
                class_idx = (batch[1] == i).nonzero(as_tuple=False)
                ax[i // 5, i % 5].imshow(
                    batch[0][class_idx[0].item()].permute(1, 2, 0),
                    cmap=DATASETS[dataset]["cmap"],
                )
                ax[i // 5, i % 5].set_title(f"{class_name}")
                ax[i // 5, i % 5].axis("off")
            break

        plt.tight_layout()
        # Save the plot
        fig_output_path = Path(f"ml/plots/{dataset.lower()}-{type_.lower()}.png")
        # Create the plots directory if it does not exist
        fig_output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"ml/plots/{dataset.lower()}-{type_.lower()}.png", dpi=300, bbox_inches="tight"
        )
