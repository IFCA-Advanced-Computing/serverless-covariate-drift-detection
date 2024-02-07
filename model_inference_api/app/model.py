"""Model module."""

import datetime
import logging
from typing import Tuple

import PIL
import numpy as np
import torch
import torch.nn as nn
import torchvision

from settings import ModelSettings, TransformerSettings
from utils import SingletonMeta


class CNN(nn.Module):
    def __init__(self, input_size: Tuple[int, int, int], num_classes: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_size[0],
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
            ),
        )
        # Compute the size of the feature maps dynamically
        self._feature_size = self._get_feature_size(
            input_channels=self.input_size[0],
            input_size=self.input_size[1:],
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(
            in_features=self._feature_size,
            out_features=self.num_classes,
        )

    def _get_feature_size(self, input_channels: int, input_size: Tuple[int, int]) -> int:
        # Function to compute the size of the feature maps after convolutional layers
        x = torch.randn(1, input_channels, *input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Model(metaclass=SingletonMeta):
    """Model class."""

    def __init__(
        self: "Model",
        settings_model: ModelSettings,
        settings_transformer: TransformerSettings,
    ) -> None:
        """Init method."""
        logging.info("Loading model...")
        self.model = self.load_model(
            settings=settings_model,
        )
        logging.info("Model loaded.")
        logging.info("Loading transformer...")
        self.transformer = self.load_transformer(
            settings=settings_transformer,
        )
        logging.info("Transformer loaded.")

    def load_model(
        self: "Model",
        settings: ModelSettings,
    ) -> CNN:
        """Load model.

        :return model
        :rtype: CNN
        """
        model = self._load_model(
            settings=settings,
        )
        return model

    def load_transformer(
        self: "Model",
        settings: TransformerSettings,
    ) -> torchvision.transforms.Compose:
        """Load transformer.

        :return transformer
        :rtype: torchvision.transforms.Compose
        """
        transformer = self._load_transformer(
            settings=settings,
        )
        return transformer

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data.

        :param data: data
        :type data: np.ndarray
        :return: encoded data
        :rtype: np.ndarray
        """
        with torch.no_grad():
            encoded = self.encoder(data).numpy()
        return encoded

    def predict(
        self: "Model",
        data: torch.Tensor,
    ) -> dict[str, str | int | float]:
        """Predict.

        :param data: data tensor
        :type data: torch.Tensor
        :return: predict information
        :rtype: dict[str, str | int | float]
        """
        with torch.no_grad():
            pred = self.model(data)
        class_pred = torch.max(pred, dim=1).indices.item()

        return {
            "datetime": datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%d/%m/%Y %H:%M:%S.%f",
            ),
            "prediction": class_pred,
        }

    def transform(self, data: PIL.Image) -> torch.Tensor:
        """Transform data.

        :param data: data
        :type data: PIL.Image
        :return: transformed data
        :rtype: torch.Tensor
        """
        transformed = self.transformer(data).unsqueeze(0)
        return transformed

    @staticmethod
    def _load_model(settings: ModelSettings) -> CNN:
        model_state_dict = torch.load(
            f=settings.FILE_PATH,
        )
        input_channels = [*model_state_dict.values()][0].shape[1]
        model = CNN(
            input_size=(input_channels, 28, 28),
            num_classes=10,
        )
        model.eval()
        for k, _v in model_state_dict.copy().items():
            model_state_dict[k.removeprefix("_orig_mod.")] = model_state_dict.pop(k)
        model.load_state_dict(state_dict=model_state_dict)
        return model

    @staticmethod
    def _load_transformer(
        settings: TransformerSettings,
    ) -> torchvision.transforms.Compose:
        transformer = torch.load(
            f=settings.FILE_PATH,
        )
        return transformer
