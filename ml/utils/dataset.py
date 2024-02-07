import torch
import torchvision
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(
        self,
        subset: Dataset,
        transform: torch.nn.Module | torchvision.transforms.Compose | None = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.subset)
