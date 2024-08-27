import torch
import torch.nn as nn
import torch.nn.functional as nfn


__all__ = ["DiceLoss"]


class DiceLoss(nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, size_average: bool = True, reduce: bool = True) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor, weight: None | torch.Tensor = None) -> torch.Tensor:
        N = predicts.size(0)
        C = predicts.size(1)
        if predicts.ndim == 5:
            predicts = predicts.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        else:
            predicts = predicts.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        log_P = nfn.log_softmax(predicts, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(predicts, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(predicts.shape).to(predicts.device) + 1e-8
        class_mask.scatter_(1, targets, 1.)

        ones = torch.ones(predicts.shape).to(predicts.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(predicts.device)
        self.alpha = FP.sum(dim=0) / ((FP.sum(dim=0) + FN.sum(dim=0)) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        # print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=0).float()
        den = num + self.alpha * torch.sum(FP, dim=0).float() + self.beta * torch.sum(FN, dim=0).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not None:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not None:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C
        return loss
