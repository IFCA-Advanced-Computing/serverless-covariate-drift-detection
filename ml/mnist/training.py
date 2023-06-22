import argparse
import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from frouros.callbacks import PermutationTestOnBatchData
from frouros.detectors.data_drift import MMD
from torch import nn

from ml.mnist.utils.dr import Autoencoder, encode_data
from ml.mnist.utils.model import CNN


def set_seed(seed=31):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(
        epochs: int,
        data_loader: torch.utils.data.DataLoader,
        device: str,
) -> torch.nn.Module:
    model = torch.compile(
        model=CNN(), options={},
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.001,
    )

    for epoch in range(1, epochs+1):
        logger.info(f"Epoch {epoch}:")
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader, start=1):
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                logger.info(f"  Batch {i} loss: {running_loss / 100:.4f}")  # loss per batch
                running_loss = 0.0
    return model


def contractive_loss(outputs_e, outputs, inputs, lamda = 1e-4):
    assert outputs.shape == inputs.shape, f"outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}"
    criterion = nn.MSELoss()
    loss1 = criterion(outputs, inputs)

    outputs_e.backward(torch.ones(outputs_e.size()), retain_graph=True)
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()

    loss = loss1 + (lamda*loss2)
    return loss


def main(
        train_images_dir: str,
        autoencoder_batch_size: int,
        autoencoder_epochs: int,
        encoder_output_path: Path,
        detector_batch_size: int,
        detector_chunk_size: int,
        detector_output_path: Path,
        model_batch_size: int,
        model_epochs: int,
        model_output_path: Path,
        transform_output_path: Path,
        permutation_test_number: int,
        permutation_test_num_jobs: int,
        latent_dim: int,
        random_state: int,
) -> None:
    set_seed(seed=random_state)
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

    train_all_dataset_size = len(train_all_dataset)
    train_dataset_size = int(train_all_dataset_size * 0.5)
    ref_dataset_size = train_all_dataset_size - train_dataset_size

    train_dataset, ref_dataset = torch.utils.data.random_split(
        dataset=train_all_dataset,
        lengths=[
            train_dataset_size,
            ref_dataset_size,
        ],
        generator=torch.Generator().manual_seed(random_state),
    )

    train_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=train_dataset,
        batch_size=model_batch_size,
        shuffle=True,
    )

    logger.info("Training model...")
    model = train(
        epochs=model_epochs,
        data_loader=train_data_loader,
        device=device,
    )

    logger.info(f"Saving model to {model_output_path}...")
    torch.save(
        obj=model.state_dict(),
        f=model_output_path,
    )

    ref_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=ref_dataset,
        batch_size=autoencoder_batch_size,
        shuffle=False,
    )

    logger.info("Training dimensionality reduction autoencoder...")
    autoencoder = train_dr(
        autoencoder_epochs=autoencoder_epochs,
        latent_dim=latent_dim,
        data_loader=ref_data_loader,
        device=device,
    )

    logger.info(f"Saving encoder to {encoder_output_path}...")
    torch.save(
        obj=autoencoder.encoder.state_dict(),
        f=encoder_output_path,
    )

    ref_data_loader = torch.utils.data.DataLoader(  # 30000 samples
        dataset=ref_dataset,
        batch_size=detector_batch_size,
        shuffle=False,
    )

    logger.info("Encoding reference data...")
    X_ref_encoded, _ = encode_data(
        encoder=autoencoder.encoder,
        data_loader=ref_data_loader,
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


def train_dr(
        autoencoder_epochs: int,
        latent_dim: int,
        data_loader: torch.utils.data.DataLoader,
        device: str,
) -> Autoencoder:
    autoencoder = torch.compile(
        Autoencoder(
            latent_dim=latent_dim,
        ),
    ).to(device)
    optimizer = torch.optim.Adam(
        params=autoencoder.parameters(),
        lr=0.001,
    )
    for _epoch in range(1, autoencoder_epochs + 1):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(data_loader, start=1):
            inputs.requires_grad = True
            inputs.retain_grad()

            outputs_e, outputs = autoencoder(inputs)
            loss = contractive_loss(outputs_e, outputs, inputs)

            inputs.requires_grad = False

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                logger.info(f"  Batch {i} loss: {running_loss / 100:.4f}")  # loss per batch
                running_loss = 0.0
    return autoencoder


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
            PermutationTestOnBatchData(
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


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="MNIST training.")
    parser.add_argument("-ti", "--TrainImagesDir", type=str, help="Train images directory", default="/tmp/mnist/train/")
    parser.add_argument("-mb", "--ModelBatchSize", type=int, help="Model batch size", default=64)
    parser.add_argument("-me", "--ModelEpochs", type=int, help="Model epochs", default=5)
    parser.add_argument("-mo", "--ModelOutputPath", type=str, help="Model output path", default="objects/model.pt")
    parser.add_argument("-db", "--DetectorBatchSize", type=int, help="Detector batch size", default=8)
    parser.add_argument("-dc", "--DetectorChunkSize", type=int, help="Detector chunk size", default=100)
    parser.add_argument("-do", "--DetectorOutputPath", type=str, help="Detector output path", default="objects/detector.pkl")
    parser.add_argument("-ab", "--AutoencoderBatchSize", type=int, help="Autoencoder batch size", default=64)
    parser.add_argument("-ae", "--AutoencoderEpochs", type=int, help="Autoencoder epochs", default=5)
    parser.add_argument("-eo", "--EncoderOutputPath", type=str, help="Encoder output path", default="objects/encoder.pt")
    parser.add_argument("-to", "--TransformOutputPath", type=str, help="Transform output path", default="objects/transformer.pt")
    parser.add_argument("-ld", "--LatentDim", type=int, help="Latent dimension", default=5)
    parser.add_argument("-pn", "--PermutationTestNumber", type=int, help="Number of permutation tests", default=100)
    parser.add_argument("-pp", "--PermutationTestNumJobs", type=int, help="Number of permutation test jobs", default=multiprocessing.cpu_count())
    parser.add_argument("-rs", "--RandomState", type=int, help="Random state", default=31)

    args = parser.parse_args()

    main(
        train_images_dir=args.TrainImagesDir,
        model_batch_size=args.ModelBatchSize,
        model_epochs=args.ModelEpochs,
        model_output_path=Path(args.ModelOutputPath),
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
