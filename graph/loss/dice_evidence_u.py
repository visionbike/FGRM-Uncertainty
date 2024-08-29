import torch
import torch.nn as nn
import torch.nn.functional as nfn
from graph.loss import DiceLoss, KLLoss

__all__ = ["DiceEvidenceULoss"]


class DiceEvidenceULoss(nn.Module):
    def __init__(self, num_classes: int, step_lambda: int, step_total: int, eps: bool, disentangle: bool):
        super(DiceEvidenceULoss, self).__init__()
        self.dice = DiceLoss()
        self.kl = KLLoss(num_classes)
        self.num_classes = num_classes
        self.step_lambda = step_lambda
        self.step_total = step_total
        self.disentangle = disentangle
        self.eps = eps

    def forward(self, p: torch.Tensor, alpha: torch.Tensor, evidence: torch.Tensor, step_current: int) -> torch.Tensor:
        # soft_p = get_soft_label(soft_p, c)
        if alpha.ndim == 5:
            soft_p = p.unsqueeze(1)
        else:
            soft_p = p

        L_dice = self.dice(evidence, soft_p)

        alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
        alpha = alpha.transpose(1, 2)  # [N, HW, C]
        alpha = alpha.contiguous().view(-1, alpha.size(2))
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        label = nfn.one_hot(p, num_classes=self.num_classes)
        label = label.view(-1, self.num_classes)
        # digama loss
        L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
        # log loss
        # labelK = label * (torch.log(S) -  torch.log(alpha))
        # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

        # KL loss
        annealing_coef = min(1, int(step_current / self.step_lambda))
        annealing_start = torch.tensor(0.01, dtype=torch.float32)
        annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / self.step_total * step_current)
        annealing_AU = min(1., annealing_AU)
        alp = E * (1 - label) + 1
        L_KL = annealing_coef * self.kl(alp)
        # AU Loss
        pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
        uncertainty = self.num_classes / S
        target = p.view(-1, 1)
        acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
        if self.disentangle:
            acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + self.eps)
            inacc_certain = - torch.log((1 - pred_scores) * uncertainty + self.eps)
        else:
            acc_uncertain = - pred_scores * torch.log(1 - uncertainty + self.eps)
            inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + self.eps)
        L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain

        return L_ace + L_KL + (1 - annealing_AU) * L_dice + L_AU
