import argparse
import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.utils.autoencoder import Encoder, encode_data
from ml.utils.cnn import CNN, test_model
from ml.utils.drift import make_transformed_dataset, save_images
from ml.utils.drift import transformations
from utils.constants import DATASETS, IMAGE_SIZE


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


def add_suffix_to_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def main(
        dataset: str,
        test_images_dir: str,
        encoder_file_path: Path,
        detector_file_path: Path,
        cnn_batch_size: int,
        cnn_file_path: Path,
        transform_file_path: Path,
        alpha: float,
        save_transformed_images: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_suffix = dataset.lower()

    logger.info("Loading transform...")
    transform = torch.load(
        f=add_suffix_to_path(
            path=transform_file_path,
            suffix=dataset_suffix,
        )
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

    logger.info("Loading encoder...")
    encoder_state_dict = torch.load(
        f=add_suffix_to_path(
            path=encoder_file_path,
            suffix=dataset_suffix,
        )
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

    logger.info("Loading drift detector...")
    detector = load_obj(
        path=add_suffix_to_path(
            path=detector_file_path,
            suffix=dataset_suffix,
        )
    )
    # FIXME: This is a hack to change the number of permutations
    # detector.callbacks[0].num_permutations = 20

    # bonferroni_alpha = alpha / detector.callbacks[0].num_permutations
    #
    # logger.info(f"Checking for drift using Bonferroni correction with alpha={bonferroni_alpha}...")

    for type_, X in X_encoded:
        data_drift_check = check_drift(
            detector=detector,
            X=X,
            alpha=alpha,
        )
        logger.info(f"{type_} data drift check: {data_drift_check}")

    logger.info("Loading CNN...")
    cnn = CNN(
        input_size=(DATASETS[dataset]["input_channels"], *IMAGE_SIZE),
        num_classes=DATASETS[dataset]["num_classes"],
    ).to(device)
    cnn.eval()
    cnn_state_dict = torch.load(
        f=add_suffix_to_path(
            path=cnn_file_path,
            suffix=dataset_suffix,
        )
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
    root_path = Path(__file__).parent.parent.parent.resolve()
    model_inference_objects_path = Path(root_path, "model_inference_api/app/objects/")
    detector_objects_path = Path(root_path, "detector_api/app/objects/")
    dimensionality_reduction_objects_path = Path(root_path, "dimensionality_reduction_api/app/objects/")

    parser = argparse.ArgumentParser(description="Testing model")
    parser.add_argument("-d", "--Dataset", type=str, help="Dataset", choices=["MNIST", "FashionMNIST", "CIFAR10"], default="CIFAR10")
    parser.add_argument("-ti", "--TestImagesDir", type=str, help="Test images directory", default="/tmp/test/")
    parser.add_argument("-mb", "--CNNBatchSize", type=int, help="CNN batch size", default=64)
    parser.add_argument("-mf", "--CNNFilePath", type=str, help="CNN file path", default=Path(current_file_path, "objects/cnn.pt"))
    parser.add_argument("-df", "--DetectorFilePath", type=str, help="Detector file path", default=Path(current_file_path, "objects/detector.pkl"))
    parser.add_argument("-ef", "--EncoderFilePath", type=str, help="Encoder file path", default=Path(current_file_path, "objects/encoder.pt"))
    parser.add_argument("-tf", "--TransformFilePath", type=str, help="Transform file path", default=Path(current_file_path, "objects/transformer.pt"))
    parser.add_argument("-a", "--Alpha", type=float, help="Alpha", default=0.01)
    parser.add_argument("-st", "--SaveTransformedImages", type=bool, help="Save transformed images", default=False)

    args = parser.parse_args()

    main(
        dataset=args.Dataset,
        test_images_dir=args.TestImagesDir,
        cnn_batch_size=args.CNNBatchSize,
        cnn_file_path=Path(args.CNNFilePath),
        detector_file_path=Path(args.DetectorFilePath),
        encoder_file_path=Path(args.EncoderFilePath),
        transform_file_path=Path(args.TransformFilePath),
        alpha=args.Alpha,
        save_transformed_images=args.SaveTransformedImages,
    )
