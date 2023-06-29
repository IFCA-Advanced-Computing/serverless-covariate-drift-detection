import argparse
import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from frouros.callbacks import PermutationTestDistanceBased
from frouros.detectors.data_drift import MMD
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.autoencoder import (
    AutoEncoder,
    AutoEncoderSystem,
    encode_data,
)
from utils.cnn import CNN, CNNSystem


def train_cnn(
        cnn: pl.LightningModule,
        epochs: int,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
) -> torch.nn.Module:
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
    )
    cnn_trainer.fit(
        model=cnn_system,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
    # Load best model
    logging.info(
        f"Best model: {cnn_checkpoint_callback.best_model_path}",
    )
    cnn_system = CNNSystem.load_from_checkpoint(
        checkpoint_path=cnn_checkpoint_callback.best_model_path,
        cnn=cnn,
    )
    return cnn_system


def train_autoencoder(
        autoencoder: pl.LightningModule,
        epochs: int,
        train_data_loader: torch.utils.data.DataLoader,
        val_data_loader: torch.utils.data.DataLoader,
) -> AutoEncoderSystem:
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
    )
    autoencoder_trainer.fit(
        model=autoencoder_system,
        train_dataloaders=train_data_loader,
        val_dataloaders=val_data_loader,
    )
    # Load best model
    logging.info(
        f"Best model: {autoencoder_checkpoint_callback.best_model_path}",
    )
    autoencoder_system = AutoEncoderSystem.load_from_checkpoint(
        checkpoint_path=autoencoder_checkpoint_callback.best_model_path,
        autoencoder=autoencoder,
    )
    return autoencoder_system


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
    detector = MMD(
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


def main(
        train_images_dir: str,
        autoencoder_batch_size: int,
        autoencoder_epochs: int,
        encoder_output_path: Path,
        detector_batch_size: int,
        detector_chunk_size: int,
        detector_output_path: Path,
        cnn_batch_size: int,
        cnn_epochs: int,
        cnn_output_path: Path,
        transform_output_path: Path,
        permutation_test_number: int,
        permutation_test_num_jobs: int,
        latent_dim: int,
        random_state: int,
) -> None:
    pl.seed_everything(
        seed=31,
        workers=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(),  # Convert to grayscale (1 channel)
            torchvision.transforms.ToTensor(),  # Convert images to the range [0.0, 1.0] (normalize)
        ],
    )
    logger.info(f"Saving transform to {transform_output_path}...")
    torch.save(
        obj=transform,
        f=transform_output_path,
    )

    train_all_dataset = torchvision.datasets.MNIST(
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

    train_cnn_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=train_cnn_dataset,
        batch_size=cnn_batch_size,
        shuffle=True,
    )
    val_cnn_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=val_cnn_dataset,
        batch_size=cnn_batch_size,
        shuffle=False,
    )

    cnn = CNN(
        num_classes=10,
    )
    logger.info("Training CNN...")
    cnn_system = train_cnn(
        cnn=cnn,
        epochs=cnn_epochs,
        train_data_loader=train_cnn_data_loader,
        val_data_loader=val_cnn_data_loader,
    )

    logger.info(f"Saving CNN to {cnn_output_path}...")
    torch.save(
        obj=cnn_system.cnn.state_dict(),
        f=cnn_output_path,
    )

    train_autoencoder_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=train_autoencoder_dataset,
        batch_size=autoencoder_batch_size,
        shuffle=True,
    )
    val_autoencoder_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=val_autoencoder_dataset,
        batch_size=autoencoder_batch_size,
        shuffle=False,
    )

    autoencoder = AutoEncoder(
        latent_dim=latent_dim,
    )
    logger.info("Training dimensionality reduction autoencoder...")
    autoencoder_system = train_autoencoder(
        autoencoder=autoencoder,
        epochs=autoencoder_epochs,
        train_data_loader=train_autoencoder_data_loader,
        val_data_loader=val_autoencoder_data_loader,
    )

    encoder = autoencoder_system.autoencoder.encoder
    logger.info(f"Saving encoder to {encoder_output_path}...")
    torch.save(
        obj=encoder.state_dict(),
        f=encoder_output_path,
    )

    detector_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=detector_dataset,
        batch_size=detector_batch_size,
        shuffle=False,
    )

    logger.info("Encoding reference data...")
    X_ref_encoded, _ = encode_data(
        encoder=encoder.to(device),
        data_loader=detector_data_loader,
    )

    logger.info("Fitting detector...")
    detector = fit_detector(
        X_ref=X_ref_encoded,
        chunk_size=detector_chunk_size,
        permutation_test_num_jobs=permutation_test_num_jobs,
        permutation_test_number=permutation_test_number,
        random_state=random_state,
    )

    logger.info(f"Saving detector to {detector_output_path}...")
    save_obj(
        obj=detector,
        path=detector_output_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    current_file_path = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="MNIST training.")
    parser.add_argument("-ti", "--TrainImagesDir", type=str, help="Train images directory", default="/tmp/mnist/train/")
    parser.add_argument("-cb", "--CNNBatchSize", type=int, help="CNN batch size", default=128)
    parser.add_argument("-ce", "--CNNEpochs", type=int, help="CNN epochs", default=10)
    parser.add_argument("-co", "--CNNOutputPath", type=str, help="CNN output path", default=Path(current_file_path, "objects/cnn.pt"))
    parser.add_argument("-db", "--DetectorBatchSize", type=int, help="Detector batch size", default=8)
    parser.add_argument("-dc", "--DetectorChunkSize", type=int, help="Detector chunk size", default=100)
    parser.add_argument("-do", "--DetectorOutputPath", type=str, help="Detector output path", default=Path(current_file_path, "objects/detector.pkl"))
    parser.add_argument("-ab", "--AutoencoderBatchSize", type=int, help="Autoencoder batch size", default=128)
    parser.add_argument("-ae", "--AutoencoderEpochs", type=int, help="Autoencoder epochs", default=10)
    parser.add_argument("-eo", "--EncoderOutputPath", type=str, help="Encoder output path", default=Path(current_file_path, "objects/encoder.pt"))
    parser.add_argument("-to", "--TransformOutputPath", type=str, help="Transform output path", default=Path(current_file_path, "objects/transformer.pt"))
    parser.add_argument("-ld", "--LatentDim", type=int, help="Latent dimension", default=5)
    parser.add_argument("-pn", "--PermutationTestNumber", type=int, help="Number of permutation tests", default=5)
    parser.add_argument("-pp", "--PermutationTestNumJobs", type=int, help="Number of permutation test jobs", default=multiprocessing.cpu_count())
    parser.add_argument("-rs", "--RandomState", type=int, help="Random state", default=31)

    args = parser.parse_args()

    main(
        train_images_dir=args.TrainImagesDir,
        cnn_batch_size=args.CNNBatchSize,
        cnn_epochs=args.CNNEpochs,
        cnn_output_path=Path(args.CNNOutputPath),
        detector_batch_size=args.DetectorBatchSize,
        detector_chunk_size=args.DetectorChunkSize,
        detector_output_path=Path(args.DetectorOutputPath),
        autoencoder_batch_size=args.AutoencoderBatchSize,
        autoencoder_epochs=args.AutoencoderEpochs,
        encoder_output_path=Path(args.EncoderOutputPath),
        transform_output_path=Path(args.TransformOutputPath),
        latent_dim=args.LatentDim,
        permutation_test_number=args.PermutationTestNumber,
        permutation_test_num_jobs=args.PermutationTestNumJobs,
        random_state=args.RandomState,
    )
