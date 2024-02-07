"""Autoencoder model for MNIST dataset."""

from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


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


class Decoder(nn.Module):

    def __init__(self, output_size: Tuple[int, int, int], feature_size: Tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        self.output_size = output_size

        self.decoder_lin = nn.Sequential(
            nn.Linear(
                in_features=latent_dim,
                out_features=128,
            ),
            nn.ReLU(True),
            nn.Linear(
                in_features=128,
                out_features=np.prod(feature_size),
            ),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(
            dim=1,
            unflattened_size=feature_size,
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                output_padding=0,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=self.output_size[0],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, input_size: Tuple[int, int, int], latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_size=input_size,
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
            output_size=input_size,
            feature_size=self.encoder.feature_size,
            latent_dim=latent_dim,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoEncoderSystem(pl.LightningModule):

    def __init__(self, autoencoder: nn.Module) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        output = self.autoencoder(x)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_autoencoder_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        _, x_hat = self.autoencoder(x)
        loss = self.loss_fn(x_hat, x)
        self.log(
            name="train_autoencoder_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self):
        # Collect training losses at the end of each epoch
        train_loss = self.trainer.callback_metrics["train_autoencoder_loss"].item()
        self.train_losses.append(train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, x_hat = self.autoencoder(x)
        loss = self.loss_fn(x_hat, x)
        self.log(
            name="val_autoencoder_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_end(self):
        # Collect validation losses at the end of each epoch
        val_loss = self.trainer.callback_metrics["val_autoencoder_loss"].item()
        self.val_losses.append(val_loss)


def encode_data(encoder, data_loader) -> tuple[np.ndarray, np.ndarray]:
    X_encoded = []
    y = []
    encoder.eval()
    with torch.no_grad():
        for X_sample, y_sample in data_loader:
            X_encoded.append(encoder(X_sample))
            y.append(y_sample)
    X_encoded = torch.cat(X_encoded, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    return X_encoded, y
