"""Autoencoder model for MNIST dataset."""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, latent_dim: int) -> None:
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.encoder = Encoder(
            latent_dim=latent_dim,
        )
        self.decoder = Decoder(
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
        self.log("train_autoencoder_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        _, x_hat = self.autoencoder(x)
        val_loss = self.loss_fn(x_hat, x)
        self.log("val_autoencoder_loss", val_loss)


def encode_data(encoder, data_loader) -> tuple[np.ndarray, np.ndarray]:
    X_encoded = []
    y = []
    with torch.no_grad():
        for X_sample, y_sample in data_loader:
            X_encoded.append(encoder(X_sample))
            y.append(y_sample)
    X_encoded = torch.cat(X_encoded, dim=0).cpu().numpy()
    y = torch.cat(y, dim=0).cpu().numpy()
    return X_encoded, y
