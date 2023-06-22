from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


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
