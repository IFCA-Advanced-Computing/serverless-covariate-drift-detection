import argparse
import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.autoencoder import Encoder, encode_data
from utils.cnn import CNN, test_model
from utils.constants import DATASETS, IMAGE_SIZE
from utils.drift import make_transformed_dataset, save_images
from utils.drift import transformations


def check_drift(detector, X: np.ndarray, alpha: float) -> dict[str, Any]:
    distance, callback_logs = detector.compare(X=X)
    return {
        "is_drift": callback_logs["permutation_test"]["p_value"] <= alpha,
        "distance": distance,
        "p_value": callback_logs["permutation_test"]["p_value"],
    }


def load_obj(path: Path) -> Any:
    with open(path, "rb") as file:
        obj = pickle.load(
            file=file,
        )
    return obj


def main(
    dataset: str,
    test_images_dir: str,
    cnn_batch_size: int,
    alpha: float,
    save_transformed_images: bool,
    objects_path: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_lowercase = DATASETS[dataset]["lowercase"]

    transform_file_path = Path(objects_path, f"{dataset_lowercase}/transform.pt")

    logger.info("Loading transform...")
    transform = torch.load(
        f=transform_file_path,
    )

    test_dataset = DATASETS[dataset]["name"](
        root=test_images_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loaders = []
    test_data_loader = torch.utils.data.DataLoader(  # 10000 samples
        dataset=test_dataset,
        batch_size=cnn_batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() - 1,
    )
    data_loaders.append(("Reference", test_data_loader))

    for type_, transformation in transformations:
        transformed_dataset = make_transformed_dataset(
            subset=test_dataset,
            transform=transformation,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=transformed_dataset,
            batch_size=cnn_batch_size,
            shuffle=False,
        )
        data_loaders.append((type_, data_loader))

    for type_, data_loader in data_loaders:
        if save_transformed_images:
            logger.info(f"Saving {type_} images...")
            save_images(
                data_loader=data_loader,
                target_dir=Path("data", type_),
            )

    encoder_file_path = Path(objects_path, f"{dataset_lowercase}/encoder.pt")

    logger.info("Loading encoder...")
    encoder_state_dict = torch.load(
        f=encoder_file_path,
    )
    latent_dim = [*encoder_state_dict.items()][-1][-1].size(dim=0)
    encoder = Encoder(
        input_size=(DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        latent_dim=latent_dim,
    ).to(device)
    encoder.load_state_dict(
        state_dict=encoder_state_dict,
    )
    logger.info("Applying dimensionality reduction...")

    X_encoded = []
    for type_, data_loader in data_loaders:
        X, y = encode_data(
            encoder=encoder,
            data_loader=data_loader,
        )
        X_encoded.append((type_, X))

    detector_file_path = Path(objects_path, f"{dataset_lowercase}/detector.pkl")

    logger.info("Loading covariate drift detector...")
    detector = load_obj(
        path=detector_file_path,
    )

    for type_, X in X_encoded:
        covariate_drift_check = check_drift(
            detector=detector,
            X=X,
            alpha=alpha,
        )
        logger.info(f"{type_} covariate drift check: {covariate_drift_check}")

    cnn_file_path = Path(objects_path, f"{dataset_lowercase}/cnn.pt")

    logger.info("Loading CNN...")
    cnn = CNN(
        input_size=(DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        num_classes=DATASETS[dataset]["num_classes"],
    ).to(device)
    cnn.eval()
    cnn_state_dict = torch.load(
        f=cnn_file_path,
    )
    for k, _v in cnn_state_dict.copy().items():
        cnn_state_dict[k.removeprefix("_orig_mod.")] = cnn_state_dict.pop(k)
    cnn.load_state_dict(state_dict=cnn_state_dict)

    logger.info("Testing CNN...")
    for type_, data_loader in data_loaders:
        accuracy = test_model(
            model=cnn,
            data_loader=data_loader,
        )
        logger.info(f"{type_} accuracy on test set: {accuracy:.4f}")


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

    parser = argparse.ArgumentParser(description="Testing model")
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
        "--TestImagesDir",
        type=str,
        help="Test images directory",
        default="/tmp/test/",
    )
    parser.add_argument(
        "-mb", "--CNNBatchSize", type=int, help="CNN batch size", default=64
    )
    parser.add_argument("-a", "--Alpha", type=float, help="Alpha", default=0.01)
    parser.add_argument(
        "-st",
        "--SaveTransformedImages",
        type=bool,
        help="Save transformed images",
        default=False,
    )

    args = parser.parse_args()

    main(
        dataset=args.Dataset,
        test_images_dir=args.TestImagesDir,
        cnn_batch_size=args.CNNBatchSize,
        alpha=args.Alpha,
        save_transformed_images=args.SaveTransformedImages,
        objects_path=objects_path,
    )
