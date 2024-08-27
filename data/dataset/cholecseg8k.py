from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

__all__ = ["CholecSeg8kDataset"]


class CholecSeg8kDataset(Dataset):
    def __init__(self, root: str, phase: str) -> None:
        super().__init__()
        data = np.load(f"{root}/data_{phase}.npz", allow_pickle=True)
        self.images = data["image"]
        self.labels = data["label"]
        self.fnames = data["name"]
        self.transforms = Compose(
            [
                ToTensorV2()
            ],
            is_check_shapes=True
        )

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        transformed = self.transforms(image=self.images[idx], mask=self.labels[idx])
        return transformed["image"], transformed["mask"].long(), self.fnames[idx]
