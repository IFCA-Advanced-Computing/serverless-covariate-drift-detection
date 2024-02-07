"""Encoder module."""

import logging
from typing import Tuple

import numpy as np
import torch
import torchvision
from torch import nn

from settings import EncoderSettings, TransformerSettings
from utils import SingletonMeta


class Encoder(nn.Module):

    def __init__(self, input_size: Tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        self.input_size = input_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_size[0],
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten(
            start_dim=1,
        )

        self.feature_size = self._get_feature_size(
            input_channels=self.input_size[0],
            input_size=self.input_size[1:],
        )

        # Convert tuple to the number of features
        self._feature_size = np.prod(self.feature_size)

        self.encoder_lin = nn.Sequential(
            nn.Linear(
                in_features=self._feature_size,
                out_features=128,
            ),
            nn.ReLU(True),
            nn.Linear(
                in_features=128,
                out_features=latent_dim,
            ),
        )

    def _get_feature_size(self, input_channels: int, input_size: Tuple[int, int]) -> Tuple[int, int, int]:
        # Function to compute the size of the feature maps after convolutional layers
        x = torch.randn(1, input_channels, *input_size)
        x = self.encoder_conv(x)
        return tuple(x.squeeze(dim=0).shape)

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class DimensionalityReduction(metaclass=SingletonMeta):
    """Encoder class."""

    def __init__(
        self: "DimensionalityReduction",
        settings_encoder: EncoderSettings,
        settings_transformer: TransformerSettings,
    ) -> None:
        """Init method."""
        logging.info("Loading image encoder...")
        self.encoder = self.load_encoder(
            settings=settings_encoder,
        )
        logging.info("Image encoder loaded.")
        logging.info("Loading transformer...")
        self.transformer = self.load_transformer(
            settings=settings_transformer,
        )
        logging.info("Transformer loaded.")

    def load_encoder(
        self: "DimensionalityReduction",
        settings: EncoderSettings,
    ) -> Encoder:
        """Load image encoder.

        :return encoder
        :rtype: dict[str, str | None]
        """
        encoder = self._load_encoder(
            settings=settings,
        )
        return encoder

    def load_transformer(
        self: "DimensionalityReduction",
        settings: EncoderSettings,
    ) -> torchvision.transforms.Compose:
        """Load image encoder.

        :return encoder
        :rtype: dict[str, str | None]
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

    def transform(self, data: np.ndarray) -> torch.Tensor:
        """Transform data.

        :param data: data
        :type data: np.ndarray
        :return: transformed data
        :rtype: np.ndarray
        """

        transformed = self.transformer(data).unsqueeze(0)
        return transformed

    @staticmethod
    def _load_encoder(settings: EncoderSettings) -> Encoder:
        encoder_state_dict = torch.load(
            f=settings.FILE_PATH,
        )
        latent_dim = [*encoder_state_dict.items()][-1][-1].size(dim=0)
        input_channels = [*encoder_state_dict.items()][0][-1].size()[1]
        encoder = Encoder(
            input_size=(input_channels, 28, 28),
            latent_dim=latent_dim,
        )
        encoder.eval()
        encoder.load_state_dict(
            state_dict=encoder_state_dict,
        )
        return encoder

    @staticmethod
    def _load_transformer(
        settings: TransformerSettings,
    ) -> torchvision.transforms.Compose:
        transformer = torch.load(
            f=settings.FILE_PATH,
        )
        return transformer
