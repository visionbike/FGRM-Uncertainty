import torch
import torch.nn as nn

__all__ = ["DiceScore"]


class DiceScore(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1., activation: str = "sigmoid") -> None:
        super().__init__()
        self.smooth = smooth
        self.activation = activation
        self.num_classes = num_classes

    def forward(self, predicts: torch.Tensor, targets: torch.Tensor) -> float:
        batch_size = predicts.shape[0]
        all_dice = torch.zeros(batch_size)
        seg_pred = torch.from_numpy(predicts)
        gt = torch.from_numpy(targets)
        label_ratio_ones = torch.ones_like(seg_pred)
        label_ratio_sum = torch.sum(label_ratio_ones.view(batch_size, -1), dim=1)
        # label_ratio = []
        # dice_ones = torch.ones(batch_size)
        for i in range(self.num_classes):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            A = torch.sum(each_pred.view(batch_size, -1), dim=1)
            B = torch.sum(each_gt.view(batch_size, -1), dim=1)
            union = A + B
            mask = union > 0
            mask = mask.to(torch.int32)
            dice = (2. * intersection) / (union + 1e-6)
            dice = mask * dice + (1 - mask) * dice
            label_ratio = (B * 1.0) / label_ratio_sum
            all_dice += dice * label_ratio

        return torch.mean(all_dice * 1.0).item()
