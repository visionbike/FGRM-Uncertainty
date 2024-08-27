import torch.nn as nn
from .dice import *
from .dice_evidence_u import *
from .kl import *

__all__ = ["get_criterion"]


def get_criterion(name: str, **kwargs) -> nn.Module:
    if name == "dice_evidence_u":
        criterion = DiceEvidenceULoss(**kwargs)
    else:
        raise NotImplementedError
    return criterion
