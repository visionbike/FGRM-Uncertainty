from typing import Tuple
import numpy as np
import einops
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import normalized_mutual_info_score


__all__ = [
    "MutualInformation"
]


class MutualInformation(nn.Module):
    """
    Mutual Information between two sets of images.
    """

    def __init__(self, sigma: float = 0.1, num_bins: int = 256, normalize: bool = True, device: None | torch.device | str = None) -> None:
        """

        :param sigma:
        :param num_bins:
        :param normalize:
        :param device:
        """
        super(MutualInformation, self).__init__()
        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=device).float(), requires_grad=False)

    def marginal_pdf(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Marginal probability density function.

        :param values: input tensor.
        :return: PDF tensor, kernel tensor.
        """
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))
        #
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        #
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
        #
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
        #
        return mutual_information

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """

        :param x1: input tensor 1 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :param x2: input tensor 2 with size of (B, C, H, W) and the values are normalized in range [0, 1].
        :return: mutual information value.
        """
        return self.get_mutual_information(x1, x2)


if __name__ == "__main__":
    device = "cuda:0"
    # for testing
    img1 = Image.open("grad.jpg").convert("L")
    img2 = img1.rotate(10)
    #
    mi_true_1 = normalized_mutual_info_score(np.array(img1).ravel(), np.array(img2).ravel())
    mi_true_2 = normalized_mutual_info_score(np.array(img1).ravel(), np.array(img2).ravel())
    #
    img1 = transforms.ToTensor()(img1).unsqueeze(dim=0).to(device)
    img2 = transforms.ToTensor()(img2).unsqueeze(dim=0).to(device)
    # a pair of different images, a pair of same images
    x1 = torch.cat([img1, img2])
    x2 = torch.cat([img2, img2])
    #
    MI = MutualInformation(num_bins=256, sigma=0.1, normalize=True).to(device)
    mi_test = MI(x1, x2)
    #
    mi_test_1 = mi_test[0].cpu().numpy()
    mi_test_2 = mi_test[1].cpu().numpy()
    #
    print(f"Image Pair 1 | sklearn MI: {mi_true_1}, this MI: {mi_test_1}")
    print(f"Image Pair 2 | sklearn MI: {mi_true_2}, this MI: {mi_test_2}")
    #
    assert (np.abs(mi_test_1 - mi_true_1) < 0.05)
    assert (np.abs(mi_test_2 - mi_true_2) < 0.05)
