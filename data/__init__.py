from torch.utils.data import DataLoader
from .dataset import *
from .preprocessing import *

__all__ = ["get_dataloader"]


def get_dataloader(root: str, name: str, phase: str, batch_size: int, shuffle: bool = True, num_workers: int = 1) -> DataLoader:
    if name == "cholecseg8k":
        dataset_ = CholecSeg8kDataset(root, phase=phase)
    else:
        raise ValueError(f"Dataset not supported! Got dataset name: {name}.")
    return DataLoader(dataset_, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=True)
