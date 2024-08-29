import numpy as np
import torch
from torch.distributions.dirichlet import Dirichlet
from graph.metric.ece import ECE
from calibration import PlattBinnerMarginalCalibrator

__all__ = ["PolicyGradientECECalibrationLoss"]


class PolicyGradientECECalibrationLoss(torch.nn.Module):
    def __init__(self, edl_uncertainty: bool, num_bins: int = 5, sample_wise: bool = True, estimator: str = "plugin", device: None | str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.ece_softmax = ECE(edl_uncertainty, num_bins, sample_wise, estimator)

    def forward(self, edl_u: torch.Tensor, p: torch.Tensor, alpha: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        batch_size = evidence.shape[0]
        num_classes = evidence.shape[1]
        width = evidence.shape[2]
        height = evidence.shape[3]
        expected_prob = torch.nn.functional.normalize(alpha, p=1, dim=1)
        ece = self.ece_softmax(
            edl_u.detach().cpu().numpy(),
            expected_prob.detach().cpu().numpy(),
            p.detach().cpu().numpy()
        )
        #
        calibrated_probs = []
        for i in range(batch_size):
            zs = np.reshape(expected_prob[i].permute(1, 2, 0).detach().cpu().numpy(), (-1, num_classes))
            ys = np.reshape(p[i].detach().cpu().numpy(), (-1, ))
            calibrator = PlattBinnerMarginalCalibrator(zs.shape[0], num_bins=10)
            calibrator.train_calibration(zs, ys)
            calibrated_zs = calibrator.calibrate(zs)
            calibrated_zs = calibrated_zs / np.expand_dims(np.linalg.norm(calibrated_zs, ord=1, axis=1), axis=1)
            calibrated_zs = np.reshape(calibrated_zs, (width, height, num_classes))
            calibrated_probs.append(calibrated_zs)
        calibrated_probs = np.stack(calibrated_probs, axis=0) # (batch, 256, 256, num_classes)
        calibrated_probs = torch.tensor(calibrated_probs).to(self.device)
        expected_prob = expected_prob.permute(0, 2, 3, 1)
        #
        log_probs = torch.zeros(batch_size)
        entropy = torch.zeros(batch_size)
        for i in range(batch_size):
            m = Dirichlet(alpha[i].permute(1, 2, 0).view(-1, num_classes))
            log_probs[i] = torch.sum(m.log_prob(calibrated_probs[i].view(-1, num_classes))) / (
                    width * height)
            entropy[i] = torch.sum(m.entropy()) / (
                    width * height)
        ece = - torch.log(ece)
        loss_ece = - log_probs.to(self.device) * ece.to(self.device)
        # entropy = entropy.to(self.device)
        loss_ece = torch.mean(loss_ece)
        # loss_entropy = torch.mean(entropy)
        expected_prob = expected_prob.permute(0, 3, 1, 2)
        prob_entropy = -torch.sum(expected_prob * torch.log(expected_prob + 1e-6), dim=1)
        prob_entropy = torch.mean(prob_entropy)

        return 0.1 * loss_ece - 0.01 * prob_entropy
