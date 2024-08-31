from typing import Any, Tuple
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as nfn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from graph.network import get_network
from graph.loss import get_criterion
from graph.scheduler import EarlyStopping
from graph.metric import Metrics
from data import get_dataloader

__all__ = ["FGRMModel"]


class FGRMModel:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.ExpConfig.log == "wandb":
            self.run = wandb.init(
                project="FGRM_uncertainty_estimation",
                name=self.configs.ExpConfig.exp_name,
                config={
                    "method": self.configs.LossConfig.name,
                    "total_epochs": self.configs.ExpConfig.num_epochs,
                    "experiment_name": self.configs.ExpConfig.exp_name,
                    "lr": self.configs.OptimConfig.lr,
                }
            )

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.metrics = None
        self.early_stopping = None
        self.precision_matrices = None

    def prepare_model(self, model_path: None | str = None):
        model = get_network(
            self.configs.ExpConfig.task,
            self.configs.DataConfig.num_classes,
            encoder_weights=None
        )
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        model.to(self.configs.ExpConfig.device)
        return model

    def prepare_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self.configs.OptimConfig.lr)

    def prepare_criterion(self) -> nn.Module:
        return get_criterion(
            name=self.configs.LossConfig.name,
            edl_uncertainty=self.configs.ExpConfig.edl_uncertainty,
            num_bins=5,
            sample_wise=True,
            estimator="plugin",
            device=self.configs.ExpConfig.device
        )

    def prepare_early_stopping(self) -> Any:
        return EarlyStopping(patience=self.configs.SchedulerConfig.patience, verbose=True)

    def prepare_metrics(self) -> Any:
        return Metrics(
            num_classes=self.configs.DataConfig.num_classes,
            edl_uncertainty=self.configs.ExpConfig.edl_uncertainty,
            device=self.configs.ExpConfig.device
        )

    def diag_fisher(self, model: nn.Module, loader_val: DataLoader) -> dict:
        params = {n: p for n, p in model.named_parameters()}
        precision_matrices = {}
        for n, p in copy.deepcopy(params).items():
            p.data.zero_()
            precision_matrices[n] = torch.autograd.Variable(p.data.to(self.configs.ExpConfig.device))
        training = model.training
        model.train()
        for batch_idx, (image, label, _) in enumerate(loader_val):
            image, label = image.to(self.configs.ExpConfig.device), label.to(self.configs.ExpConfig.device)
            model.zero_grad()
            outputs = model(image)
            evidence = nfn.softplus(outputs, beta=20)
            alpha = evidence + 1
            edl_u = self.configs.DataConfig.num_classes / torch.sum(alpha, dim=1, keepdim=False)
            reward = torch.mean(self.criterion(edl_u, label, alpha, evidence))
            total_loss = reward
            total_loss = torch.mean(total_loss)
            total_loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n].data += p.grad.data ** 2
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        model.train(training)
        return precision_matrices

    def run_FGRM(self):
        loader_val = get_dataloader(
            self.configs.DataConfig.path,
            self.configs.DataConfig.name,
            "val",
            self.configs.DataConfig.batch_size,
            True,
            self.configs.DataConfig.num_workers
        )
        loader_test = get_dataloader(
            self.configs.DataConfig.path,
            self.configs.DataConfig.name,
            "test",
            self.configs.DataConfig.batch_size,
            False,
            self.configs.DataConfig.num_workers
        )

        self.model = self.prepare_model(model_path=self.configs.ExpConfig.edl_model_path)
        self.criterion = self.prepare_criterion()
        self.optimizer = self.prepare_optimizer(self.model)
        self.early_stopping = self.prepare_early_stopping()
        self.metrics = self.prepare_metrics()

        self.train_FGRM(loader_val, loader_test)

    def train_FGRM(self, loader_val: DataLoader, loader_test: DataLoader) -> None:
        best_val_dice = 0.
        best_val_ece = 10.
        best_val_mi = 0.
        val_loss, val_dice, val_ece, val_mi = self.validate_FGRM_epoch(self.model, loader_test)
        print(f"val dice : {val_dice:.6f} val ece : {val_ece:.6f} val mi : {val_mi:.6f}.")
        for _, epoch in zip(np.linspace(0.01, 0.99, num=self.configs.ExpConfig.num_epochs), range(self.configs.ExpConfig.num_epochs)):
            # Train
            train_loss, all_head_losses = self.train_FGRM_epoch(self.model, loader_val, self.optimizer)

            print(f"EPOCH {epoch}/{self.configs.ExpConfig.num_epochs}: Loss: {all_head_losses}")

            val_loss, val_dice, val_ece, val_mi = self.validate_FGRM_epoch(self.model, loader_test)
            if val_dice > best_val_dice:
                best_val_dice = val_dice
            if val_mi > best_val_mi:
                best_val_mi = val_mi
            if val_ece < best_val_ece:
                best_val_ece = val_ece

            if self.configs.ExpConfig.log == "wandb":
                wandb.log({"training_loss": all_head_losses, "val dice": val_dice,
                           "val ece": val_ece, "val mi": val_mi})

            print(f"val dice : {val_dice:.6f} val ece : {val_ece:.6f} val mi : {val_mi:.6f}.")
            print(f"Best val dice : {best_val_dice:.6f} Best val ece : {best_val_ece:.6f} Best val mi : {best_val_mi:.6f}.")

    def train_FGRM_epoch(self, model: nn.Module, loader_val: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, np.ndarray]:
        model.train()
        train_loss = 0.
        losses = []

        self.precision_matrices = self.diag_fisher(copy.deepcopy(model), loader_val)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.segmentation_head.parameters():
            p.requires_grad = True

        for batch_idx, (image, label, _) in enumerate(tqdm(loader_val, total=len(loader_val), file=sys.stdout)):
            image, label = image.to(self.configs.ExpConfig.device), label.to(self.configs.ExpConfig.device)
            optimizer.zero_grad()
            outputs = model(image)
            evidence = nfn.softplus(outputs, beta=20)
            alpha = evidence + 1
            edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)
            reward = torch.mean(self.criterion(edl_u, label, alpha, evidence))
            total_loss = reward
            total_loss = torch.mean(total_loss)
            losses.append(total_loss.item())
            total_loss.backward()

            for n, p in model.named_parameters():
                if p.grad is not None:
                    with torch.no_grad():
                        dims = len(self.precision_matrices[n].shape)
                        dims = tuple([i for i in range(dims)])
                        tmp_precision = torch.sqrt((1 / (self.precision_matrices[n].data + 1e-20)))
                        tmp_precision.data = nfn.normalize(tmp_precision.data, p=1, dim=dims)
                        p.grad.data = p.grad.data * tmp_precision.data

            train_loss += total_loss
            optimizer.step()

        all_train_losses = np.asarray(losses).mean(axis=0)
        for p in model.parameters():
            p.requires_grad = True
        return train_loss / len(loader_val), all_train_losses

    def validate_FGRM_epoch(self, model: nn.Module, loader_val: DataLoader) -> Tuple[float, float, float, float]:
        model.eval()
        val_loss = 0
        val_dice = []
        val_ece = []
        val_mi = []
        with torch.no_grad():
            for batch_idx, (image, label, _) in enumerate(tqdm(loader_val, total=len(loader_val), file=sys.stdout)):
                image, label = image.to(self.configs.ExpConfig.device), label.to(self.configs.ExpConfig.device)
                outputs = model(image)
                evidence = nfn.softplus(outputs, beta=20)
                alpha = evidence + 1
                edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)
                soft_output = nfn.normalize(alpha, p=1, dim=1)

                seg = torch.argmax(evidence.squeeze(), dim=1).detach().cpu().numpy()
                lbl = label.squeeze().detach().cpu().numpy()

                evals = self.metrics.get_evaluations(
                    seg,
                    lbl,
                    evidence.detach().cpu().numpy(),
                    soft_output.detach().cpu().numpy(),
                    label.detach().cpu().numpy(),
                    edl_u.detach().cpu().numpy()
                )
                val_dice.append(evals["dsc_seg"])
                val_ece.append(evals["ece"])
                val_mi.append(evals["mi"])

        assert all([dsc <= 1.0001 for dsc in val_dice])
        return val_loss / len(loader_val), sum(val_dice) / len(val_dice), sum(val_ece) / len(val_ece), sum(val_mi) / len(val_mi)
