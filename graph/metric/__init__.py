import numpy as np
import torch
from .dice import DiceScore
from .mi import MI
from .ece import ECE


__all__ = ["Metrics"]


class Metrics:
    def __init__(self, num_classes: int, edl_uncertainty: bool, device: torch.device | str):
        self.metric_dice = DiceScore(num_classes=num_classes)
        self.metric_mi = MI(sigma=0.4, num_bins=256, normalize=True, device=device)
        self.metric_ece = ECE(edl_uncertainty=edl_uncertainty, num_bins=5, sample_wise=False, estimator="plugin")

    def get_evaluations(self, s: torch.Tensor, l: torch.Tensor, evidence: np.ndarray, soft_output: np.ndarray, hard_labels: np.ndarray, edl_u: np.ndarray) -> dict:
        """
        Helper function to compute all the evaluation metrics:

        - Segmentation volume, number of regions, min and max vol for each region
        - TPF (segmentation and detection)
        - FPF (segmentation and detection)
        - DSC (segmentation and detection)
        - PPV (segmentation and detection)
        - Volume difference
        - Haursdoff distance (standard and modified)
        - Custom f-score

        Inputs:
        - gt: 3D np.ndarray, reference image (ground truth)
        - mask: 3D np.ndarray, input MRI mask
        - spacing: sets the input resolution (def: (1, 1, 1))

        Output:
        - (dict) containing each of the evaluated results

        """
        return {"dsc_seg": self.metric_dice(s, l),
                "ece": self.metric_ece(edl_u, soft_output, hard_labels),
                "mi": self.metric_mi(evidence, hard_labels)}
