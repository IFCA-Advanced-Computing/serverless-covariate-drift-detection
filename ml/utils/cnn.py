"""CNN model for MNIST dataset."""

from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn


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


class CNNSystem(pl.LightningModule):
    def __init__(self, cnn: nn.Module) -> None:
        super().__init__()
        self.cnn = cnn
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []

    def forward(self, x):
        output = self.cnn(x)
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
                "monitor": "val_cnn_loss",
            },
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_pred = self.cnn(x)
        loss = self.loss_fn(y_pred, y)
        self.log(
            name="train_cnn_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_train_epoch_end(self):
        # Collect training losses at the end of each epoch
        train_loss = self.trainer.callback_metrics["train_cnn_loss"].item()
        self.train_losses.append(train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.cnn(x)
        loss = self.loss_fn(y_pred, y)
        self.log(
            name="val_cnn_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def on_validation_epoch_end(self):
        # Collect validation losses at the end of each epoch
        val_loss = self.trainer.callback_metrics["val_cnn_loss"].item()
        self.val_losses.append(val_loss)


def test_model(model, data_loader):
    y_pred = []
    y = []
    with torch.no_grad():
        for X_sample, y_sample in data_loader:
            y_pred.append(model(X_sample))
            y.append(y_sample)
    y_pred = torch.max(torch.cat(y_pred, dim=0).cpu(), dim=1).indices
    y = torch.cat(y, dim=0).cpu()
    accuracy = torch.sum(y_pred == y) / len(y)
    return accuracy
