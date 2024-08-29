import numpy as np
import torch
import torch.nn as nn
from calibration import get_calibration_error

__all__ = ["ECE"]


class ECE(nn.Module):
    def __init__(self, edl_uncertainty: bool, num_bins: int = 5, sample_wise: bool = False, estimator: str = "plugin") -> None:
        super().__init__()
        self.edl_uncertainty = edl_uncertainty
        self.num_bins = num_bins
        self.sample_wise = sample_wise
        self.estimator = estimator

    def forward(self, edl_u: np.ndarray, softmax: np.ndarray, label: np.ndarray) -> torch.Tensor | float:
        ece = None
        if self.estimator == 'plugin':
            bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            softmax = torch.tensor(softmax)
            labels = torch.tensor(label)
            if self.edl_uncertainty:
                edl_u = torch.tensor(edl_u)
            softmax_max, predictions = torch.max(softmax, 1)
            correctness = predictions.eq(labels)
            batch_size = softmax.shape[0]
            # num_classes = softmax.shape[1]
            plugin_ece = torch.zeros(batch_size)
            for i in range(batch_size):
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    if self.edl_uncertainty:
                        in_bin = edl_u[i].gt(bin_lower.item()) * edl_u[i].le(bin_upper.item())
                    else:
                        in_bin = softmax_max[i].gt(bin_lower.item()) * softmax_max[i].le(bin_upper.item())
                    prop_in_bin = in_bin.float().mean()

                    if prop_in_bin.item() > 0.0:
                        accuracy_in_bin = correctness[i][in_bin].float().mean()
                        if self.edl_uncertainty:
                            avg_confidence_in_bin = edl_u[i][in_bin].mean()
                        else:
                            avg_confidence_in_bin = softmax_max[i][in_bin].mean()
                        plugin_ece[i] += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            ece = plugin_ece
        elif self.estimator == 'debiased':
            batch_size = softmax.shape[0]
            num_classes = softmax.shape[1]
            debiased_ece = torch.zeros(batch_size)
            for i in range(batch_size):
                zs = np.reshape(torch.tensor(softmax[i]).permute(1, 2, 0).numpy(), (-1, num_classes))
                ys = np.reshape(label[i], (-1, ))
                debiased_ece[i] = get_calibration_error(zs, ys)
            ece = debiased_ece
        if self.sample_wise:
            return ece
        return ece.mean().item()
