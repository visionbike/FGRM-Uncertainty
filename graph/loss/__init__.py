import torch.nn as nn
from .dice import *
from .kl import *
from .dice_evidence_u import *
from .policy_gradient import *

__all__ = [
    "get_criterion",
    "DiceLoss",
    "KLLoss",
    "DiceEvidenceULoss",
    "PolicyGradientECECalibrationLoss"
]


def get_criterion(name: str, **kwargs) -> nn.Module:
    criterion = None
    if name == "dice_evidence_u":
        criterion = DiceEvidenceULoss(**kwargs)
    elif name == "policy_gradient":
        criterion = PolicyGradientECECalibrationLoss(**kwargs)
    else:
        raise NotImplementedError
    return criterion
