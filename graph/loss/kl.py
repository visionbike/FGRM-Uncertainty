import torch
import torch.nn as nn

__all__ = ["KLLoss"]


class KLLoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        beta = torch.ones((1, self.num_classes)).to(alpha.device)
        # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
        M_beta = torch.ones(alpha.shape).to(alpha)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        # print(beta.shape, lnB_uni.shape)
        kl = torch.sum((alpha - M_beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl
