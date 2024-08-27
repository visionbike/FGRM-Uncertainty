from typing import Tuple
import einops
import numpy as np
import torch
import torch.nn as nn

__all__ = ["MI"]


class MI(nn.Module):
    def __init__(self, sample_wise: bool = False, sigma: float = 0.1, num_bins: int = 256, normalize: bool = True, device: None | torch.device | str = None):
        super().__init__()
        self.sample_wise = sample_wise
        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.device = device
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=device).float(), requires_grad=False)

    def marginal_pdf(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Marginal probability density function.

        :param values: input tensor.
        :return: PDF tensor, kernel tensor.
        """
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values

    def joint_pdf(self, kernel_values1: torch.Tensor, kernel_values2: torch.Tensor) -> torch.Tensor:
        """
        Joint probability density function.

        :param kernel_values1: kernel tensor 1.
        :param kernel_values2: kernel tensor 2.
        :return: PDF tensor.
        """
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf

    def get_mutual_information(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """

        :param x1: input tensor 1 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :param x2: input tensor 2 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :return: mutual information value.
        """
        assert (x1.shape == x2.shape)
        # Torch tensors for images between (0, 1)
        x1 *= 255.
        x2 *= 255.
        x1 = einops.rearrange(x1, "b c h w -> b (h w) c")
        x2 = einops.rearrange(x2, "b c h w -> b (h w) c")
        # compute PDF
        pdf_x1, kernel_x1 = self.marginal_pdf(x1)
        pdf_x2, kernel_x2 = self.marginal_pdf(x2)
        pdf_x1x2 = self.joint_pdf(kernel_x1, kernel_x2)
        # compute entropy
        H_x1 = -torch.sum(pdf_x1 * torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2 * torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2 * torch.log2(pdf_x1x2 + self.epsilon), dim=(1, 2))
        #
        mutual_information = H_x1 + H_x2 - H_x1x2
        if self.normalize:
            mutual_information = 2. * mutual_information / (H_x1 + H_x2)
        return mutual_information

    def forward(self, evidence: np.ndarray, labels: np.ndarray) -> torch.Tensor | float:
        """

        :param evidence: input tensor 1 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :param labels: input tensor 2 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :return: mutual information value.
        """
        evidence = torch.from_numpy(evidence)
        labels = torch.from_numpy(labels)
        _, predicts = torch.max(evidence, 1)
        match = torch.eq(predicts, labels)
        match = match.unsqueeze(1)
        alpha = evidence + 1
        expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
        uncertainty, _ = torch.max(expected_prob, dim=1, keepdim=True)
        match, uncertainty = match.to(self.device), uncertainty.to(self.device)
        score = (self.get_mutual_information(match, uncertainty) + self.get_mutual_information(uncertainty, match)) / 2.
        if self.sample_wise:
            return score
        return score.mean().item()
