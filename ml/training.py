import argparse
import logging
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from frouros.callbacks import PermutationTestDistanceBased
from frouros.detectors.data_drift import MMD
from frouros.utils.kernels import rbf_kernel
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.spatial.distance import pdist

from utils.autoencoder import (
    AutoEncoder,
    AutoEncoderSystem,
    encode_data,
)
from utils.cnn import CNN, CNNSystem
from utils.constants import DATASETS, IMAGE_SIZE


def train_cnn(
    cnn: pl.LightningModule,
    epochs: int,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
) -> tuple[CNNSystem, tuple[list[float], list[float]]]:
    cnn_system = CNNSystem(
        cnn=cnn,
    )
    cnn_checkpoint_callback = ModelCheckpoint(
        monitor="val_cnn_loss",
    )
    cnn_trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            cnn_checkpoint_callback,
        ],
        deterministic=True,
        num_sanity_val_steps=0,
    )
    cnn_trainer.fit(
        model=cnn_system,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
    train_losses = cnn_system.train_losses
    val_losses = cnn_system.val_losses
    # Load best model
    logging.info(
        f"Best model: {cnn_checkpoint_callback.best_model_path}",
    )
    cnn_system = CNNSystem.load_from_checkpoint(
        checkpoint_path=cnn_checkpoint_callback.best_model_path,
        cnn=cnn,
    )
    return cnn_system, (train_losses, val_losses)


def train_autoencoder(
    autoencoder: pl.LightningModule,
    epochs: int,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
) -> tuple[AutoEncoderSystem, tuple[list[float], list[float]]]:
    autoencoder_system = AutoEncoderSystem(
        autoencoder=autoencoder,
    )
    autoencoder_checkpoint_callback = ModelCheckpoint(
        monitor="val_autoencoder_loss",
    )
    autoencoder_trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            autoencoder_checkpoint_callback,
        ],
        deterministic=True,
        num_sanity_val_steps=0,
    )
    autoencoder_trainer.fit(
        model=autoencoder_system,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
    train_losses = autoencoder_system.train_losses
    val_losses = autoencoder_system.val_losses
    # Load best model
    logging.info(
        f"Best model: {autoencoder_checkpoint_callback.best_model_path}",
    )
    autoencoder_system = AutoEncoderSystem.load_from_checkpoint(
        checkpoint_path=autoencoder_checkpoint_callback.best_model_path,
        autoencoder=autoencoder,
    )
    return autoencoder_system, (train_losses, val_losses)


def save_obj(
    obj: Any,
    path: Path,
) -> None:
    with open(path, "wb") as file:
        pickle.dump(
            obj=obj,
            file=file,
        )


def fit_detector(
    X_ref: np.ndarray,
    chunk_size: int,
    permutation_test_num_jobs: int,
    permutation_test_number: int,
    random_state: int,
) -> MMD:
    sigma = np.median(
        pdist(
            X=X_ref,
            metric="euclidean",
        )
    )
    print(f"sigma: {sigma}")

    detector = MMD(
        kernel=partial(
            rbf_kernel,
            sigma=sigma,
        ),
        callbacks=[
            PermutationTestDistanceBased(
                num_permutations=permutation_test_number,
                random_state=random_state,
                num_jobs=permutation_test_num_jobs,
                name="permutation_test",
                verbose=True,
            ),
        ],
        chunk_size=chunk_size,
    )
    _ = detector.fit(X=X_ref)
    return detector


def plot_losses(
    losses: tuple[list[float], list[float]],
    name: str,
) -> None:
    # Set font size
    plt.rcParams.update({"font.size": 20})
    # Plot the figure
    epochs = list(range(1, len(losses[0]) + 1))
    plt.figure(figsize=(12, 8))
    plt.plot(
        epochs, losses[0], marker="o", linestyle="-", color="blue", label="Training"
    )
    plt.plot(
        epochs, losses[1], marker="^", linestyle="--", color="red", label="Validation"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    # Mark best validation epoch
    best_epoch = np.argmin(losses[1])
    plt.axvline(
        x=best_epoch + 1,
        color="green",
        linestyle="--",
        label=f"Best Validation Epoch: {best_epoch + 1}",
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    # Save the figure to a file (optional)
    plt.tight_layout()
    plt.savefig(f"train_val_{name}.png", dpi=300, bbox_inches="tight")
    # Show the plot
    plt.show()


def main(
    dataset: str,
    train_images_dir: str,
    autoencoder_batch_size: int,
    autoencoder_epochs: int,
    detector_batch_size: int,
    detector_chunk_size: int,
    cnn_batch_size: int,
    cnn_epochs: int,
    permutation_test_number: int,
    permutation_test_num_jobs: int,
    latent_dim: int,
    random_state: int,
    objects_path: Path,
    model_inference_objects_path: Path,
    detector_objects_path: Path,
    dimensionality_reduction_objects_path: Path,
) -> None:
    pl.seed_everything(
        seed=random_state,
        workers=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_lowercase = DATASETS[dataset]["lowercase"]

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

    transform_name = "transform"
    transform_file_name = f"{transform_name}.pt"

    transform_output_paths = [
        model_inference_objects_path / transform_file_name,
        dimensionality_reduction_objects_path / transform_file_name,
        objects_path / f"{dataset_lowercase}/{transform_file_name}",
    ]

    for output_path in transform_output_paths:
        logger.info(f"Saving transform to {output_path}...")
        # Create the directory if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=transform,
            f=output_path,
        )

    train_all_dataset = DATASETS[dataset]["name"](
        root=train_images_dir,
        train=True,
        download=True,
        transform=transform,
    )

    train_cnn_fraction = 0.6
    val_cnn_fraction = 0.1
    train_autoencoder_fraction = 0.15
    val_autoencoder_dataset_fraction = 0.05
    detector_fraction = 0.1

    (
        train_cnn_dataset,
        val_cnn_dataset,
        train_autoencoder_dataset,
        val_autoencoder_dataset,
        detector_dataset,
    ) = torch.utils.data.random_split(
        dataset=train_all_dataset,
        lengths=[
            train_cnn_fraction,
            val_cnn_fraction,
            train_autoencoder_fraction,
            val_autoencoder_dataset_fraction,
            detector_fraction,
        ],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_cnn_data_loader = torch.utils.data.DataLoader(
        dataset=train_cnn_dataset,
        batch_size=cnn_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() - 1,
    )
    val_cnn_data_loader = torch.utils.data.DataLoader(
        dataset=val_cnn_dataset,
        batch_size=cnn_batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
    )

    cnn = CNN(
        input_size=(DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        num_classes=DATASETS[dataset]["num_classes"],
    )
    logger.info("Training CNN...")
    cnn_system, cnn_losses = train_cnn(
        cnn=cnn,
        epochs=cnn_epochs,
        train_data_loader=train_cnn_data_loader,
        val_data_loader=val_cnn_data_loader,
    )

    plot_losses(
        losses=cnn_losses,
        name="cnn",
    )

    cnn_name = "cnn"

    # Save the trained CNN model to ONNX format
    onnx_output_path = objects_path / f"{dataset_lowercase}/{cnn_name}.onnx"
    logger.info(f"Saving CNN to {onnx_output_path}...")
    # Create the directory if it does not exist
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=cnn_system.cnn,
        args=torch.randn(1, DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        f=onnx_output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    cnn_file_name = f"{cnn_name}.pt"

    cnn_output_paths = [
        model_inference_objects_path / cnn_file_name,
        objects_path / f"{dataset_lowercase}/{cnn_file_name}",
    ]

    for output_path in cnn_output_paths:
        logger.info(f"Saving CNN to {output_path}...")
        # Create the directory if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=cnn_system.cnn.state_dict(),
            f=output_path,
        )

    train_autoencoder_data_loader = torch.utils.data.DataLoader(
        dataset=train_autoencoder_dataset,
        batch_size=autoencoder_batch_size,
        shuffle=True,
        num_workers=multiprocessing.cpu_count() - 1,
    )
    val_autoencoder_data_loader = torch.utils.data.DataLoader(
        dataset=val_autoencoder_dataset,
        batch_size=autoencoder_batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
    )

    autoencoder = AutoEncoder(
        input_size=(DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        latent_dim=latent_dim,
    )
    logger.info("Training dimensionality reduction autoencoder...")
    autoencoder_system, autoencoder_losses = train_autoencoder(
        autoencoder=autoencoder,
        epochs=autoencoder_epochs,
        train_data_loader=train_autoencoder_data_loader,
        val_data_loader=val_autoencoder_data_loader,
    )

    plot_losses(
        losses=autoencoder_losses,
        name="autoencoder",
    )

    encoder_name = "encoder"

    # Save the trained encoder model to ONNX format
    onnx_output_path = objects_path / f"{dataset_lowercase}/{encoder_name}.onnx"
    logger.info(f"Saving encoder to {onnx_output_path}...")
    # Create the directory if it does not exist
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=autoencoder_system.autoencoder.encoder,
        args=torch.randn(1, DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        f=onnx_output_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )

    decoder_name = "decoder"

    # Save the trained decoder model to ONNX format
    onnx_output_path = objects_path / f"{dataset_lowercase}/{decoder_name}.onnx"
    logger.info(f"Saving decoder to {onnx_output_path}...")
    # Create the directory if it does not exist
    onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=autoencoder_system.autoencoder.decoder,
        args=torch.randn(1, latent_dim),
        f=onnx_output_path,
        input_names=["embedding"],
        output_names=["input"],
        dynamic_axes={
            "embedding": {0: "batch_size"},
            "input": {0: "batch_size"},
        },
    )

    encoder_name = "encoder"
    encoder_file_name = f"{encoder_name}.pt"

    encoder_output_paths = [
        dimensionality_reduction_objects_path / encoder_file_name,
        objects_path / f"{dataset_lowercase}/{encoder_file_name}",
    ]

    encoder = autoencoder_system.autoencoder.encoder
    for output_path in encoder_output_paths:
        logger.info(f"Saving encoder to {output_path}...")
        # Create the directory if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=encoder.state_dict(),
            f=output_path,
        )

    detector_data_loader = torch.utils.data.DataLoader(
        dataset=detector_dataset,
        batch_size=detector_batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
    )

    logger.info("Encoding reference data...")
    X_ref_encoded, _ = encode_data(
        encoder=encoder.to(device),
        data_loader=detector_data_loader,
    )

    logger.info("Fitting covariate drift detector...")
    detector = fit_detector(
        X_ref=X_ref_encoded,
        chunk_size=detector_chunk_size,
        permutation_test_num_jobs=permutation_test_num_jobs,
        permutation_test_number=permutation_test_number,
        random_state=random_state,
    )

    detector_name = "detector"
    detector_file_name = f"{detector_name}.pkl"

    detector_output_paths = [
        detector_objects_path / detector_file_name,
        objects_path / f"{dataset_lowercase}/{detector_file_name}",
    ]

    for output_path in detector_output_paths:
        logger.info(f"Saving covariate drift detector to {output_path}...")
        # Create the directory if it does not exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_obj(
            obj=detector,
            path=output_path,
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger = logging.getLogger(__name__)

    current_file_path = Path(__file__).parent.resolve()
    objects_path = Path(current_file_path, "objects/")

    root_path = Path(__file__).parent.parent.resolve()
    model_inference_objects_path = Path(root_path, "model_inference_api/app/objects/")
    detector_objects_path = Path(root_path, "detector_api/app/objects/")
    dimensionality_reduction_objects_path = Path(
        root_path, "dimensionality_reduction_api/app/objects/"
    )

    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "-d",
        "--Dataset",
        type=str,
        help="Dataset",
        choices=["MNIST", "FashionMNIST", "CIFAR10"],
        default="MNIST",
    )
    parser.add_argument(
        "-ti",
        "--TrainImagesDir",
        type=str,
        help="Train images directory",
        default="/tmp/train/",
    )
    parser.add_argument(
        "-cb", "--CNNBatchSize", type=int, help="CNN batch size", default=64
    )
    parser.add_argument("-ce", "--CNNEpochs", type=int, help="CNN epochs", default=50)
    parser.add_argument(
        "-db", "--DetectorBatchSize", type=int, help="Detector batch size", default=64
    )
    parser.add_argument(
        "-dc", "--DetectorChunkSize", type=int, help="Detector chunk size", default=200
    )
    parser.add_argument(
        "-ab",
        "--AutoencoderBatchSize",
        type=int,
        help="Autoencoder batch size",
        default=64,
    )
    parser.add_argument(
        "-ae", "--AutoencoderEpochs", type=int, help="Autoencoder epochs", default=50
    )
    parser.add_argument(
        "-ld", "--LatentDim", type=int, help="Latent dimension", default=2
    )
    parser.add_argument(
        "-pn",
        "--PermutationTestNumber",
        type=int,
        help="Number of permutation tests",
        default=100,
    )
    parser.add_argument(
        "-pp",
        "--PermutationTestNumJobs",
        type=int,
        help="Number of permutation test jobs",
        default=multiprocessing.cpu_count() - 1,
    )
    parser.add_argument(
        "-rs", "--RandomState", type=int, help="Random state", default=31
    )

    args = parser.parse_args()

    main(
        dataset=args.Dataset,
        train_images_dir=args.TrainImagesDir,
        cnn_batch_size=args.CNNBatchSize,
        cnn_epochs=args.CNNEpochs,
        detector_batch_size=args.DetectorBatchSize,
        detector_chunk_size=args.DetectorChunkSize,
        autoencoder_batch_size=args.AutoencoderBatchSize,
        autoencoder_epochs=args.AutoencoderEpochs,
        latent_dim=args.LatentDim,
        permutation_test_number=args.PermutationTestNumber,
        permutation_test_num_jobs=args.PermutationTestNumJobs,
        random_state=args.RandomState,
        objects_path=objects_path,
        model_inference_objects_path=model_inference_objects_path,
        detector_objects_path=detector_objects_path,
        dimensionality_reduction_objects_path=dimensionality_reduction_objects_path,
    )
