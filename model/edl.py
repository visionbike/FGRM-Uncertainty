from typing import Any, Tuple
import sys
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

__all__ = ["EDLModel"]


class EDLModel:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.ExpConfig.log == "wandb":
            self.run = wandb.init(
                project="edl_uncertainty_estimation",
                name=self.configs.ExpConfig.exp_name,
                config={
                    "method": self.configs.LossConfig.name,
                    "lambda_epochs": self.configs.SchedulerConfig.num_epochs_lambda,
                    "total_epochs": self.configs.ExpConfig.num_epochs,
                    "AU_warmup": self.configs.SchedulerConfig.au_warmup,
                    "experiment_name": self.configs.ExpConfig.exp_name,
                    "lr": self.configs.OptimConfig.lr,
                }
            )

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = None
        self.early_stopping = None

    def prepare_model(self, model_path: None | str = None):
        model = get_network(
            self.configs.ExpConfig.task,
            self.configs.DataConfig.num_classes,
            encoder_weights=None
        )
        if model_path is not None:
            model.load_weights(torch.load(model_path))
        model.to(self.configs.ExpConfig.device)
        return model

    def prepare_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self.configs.OptimConfig.lr)

    def prepare_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            min_lr=self.configs.SchedulerConfig.lr_min,
            factor=self.configs.SchedulerConfig.lr_decay_factor,
            patience=self.configs.SchedulerConfig.patience,
            verbose=True
        )

    def prepare_criterion(self) -> nn.Module:
        return get_criterion(
            name=self.configs.LossConfig.name,
            num_classes=self.configs.DataConfig.num_classes,
            step_lambda=self.configs.SchedulerConfig.num_epochs_lambda,
            step_total=self.configs.SchedulerConfig.au_warmup,
            eps=1e-10,
            disentangle=False)

    def prepare_early_stopping(self) -> Any:
        return EarlyStopping(patience=self.configs.SchedulerConfig.patience, verbose=True)

    def prepare_metrics(self) -> Any:
        return Metrics(
            num_classes=self.configs.DataConfig.num_classes,
            edl_uncertainty=self.configs.ExpConfig.edl_uncertainty,
            device=self.configs.ExpConfig.device
        )

    def run_EDL(self, run_test=False):
        loader_train = get_dataloader(
            self.configs.DataConfig.path,
            self.configs.DataConfig.name,
            "train",
            self.configs.DataConfig.batch_size,
            True,
            self.configs.DataConfig.num_workers
        )
        loader_val = get_dataloader(
            self.configs.DataConfig.path,
            self.configs.DataConfig.name,
            "val",
            self.configs.DataConfig.batch_size,
            True,
            self.configs.DataConfig.num_workers
        )

        self.model = self.prepare_model()
        self.criterion = self.prepare_criterion()
        self.optimizer = self.prepare_optimizer(self.model)
        self.scheduler = self.prepare_scheduler(self.optimizer)
        self.early_stopping = self.prepare_early_stopping()
        self.metrics = self.prepare_metrics()

        self.train_EDL(loader_train, loader_val)

        # if run_test:
        #     # Run testing
        #     loader_test = get_dataloader(
        #         self.configs.DataConfig.path,
        #         self.configs.DataConfig.name,
        #         "train",
        #         self.configs.DataConfig.batch_size,
        #         True,
        #         self.configs.DataConfig.num_workers
        #     )
        #
        #     self.test(loader_test)

    def train_EDL(self, loader_train: DataLoader, loader_val: DataLoader) -> Tuple[float, float]:
        train_loss, val_loss = 0., 0.
        for alpha, epoch in zip(np.linspace(0.01, 0.99, num=self.configs.ExpConfig.num_epochs), range(self.configs.ExpConfig.num_epochs)):
            # Train
            train_loss, all_head_losses = self.train_EDL_epoch(self.model, loader_train, self.optimizer, epoch)

            print(f'EPOCH {epoch}/{self.configs.ExpConfig.num_epochs}: Loss:', all_head_losses)

            val_loss, val_dice, val_ece, val_mi = self.validate_EDL_epoch(self.model, loader_val, epoch)

            annealing_start = torch.tensor(0.01, dtype=torch.float32)
            annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / self.configs.SchedulerConfig.au_warmup * epoch)
            annealing_coef = min(1, epoch / self.configs.SchedulerConfig.num_epochs_lambda)
            annealing_AU = min(1, annealing_AU)

            # Update scheduler and early stopping
            if self.configs.log == "wandb":
                wandb.log({"training_loss": all_head_losses, "val dice": val_dice,
                           "val ece": val_ece, "val mi": val_mi,
                           "annealing_au": annealing_AU, "annealing_coef": annealing_coef})
            #
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)
            # Push to tensorboard if enabled

            if self.early_stopping.early_stop:
                print(f"EARLY STOPPING at EPOCH {epoch + 1}")
                break

        torch.save(self.model.state_dict(), f"{self.configs.ExpConfig.model_path}/final_model.pt")
        return train_loss, val_loss

    def train_EDL_epoch(self, model: nn.Module, loader_train: DataLoader, optimizer: optim.Optimizer, epoch: int) -> Tuple[float, np.ndarray]:
        model.train()
        train_loss = 0.
        losses = []
        for idx, (image, label, _) in enumerate(tqdm(loader_train, total=len(loader_train), file=sys.stdout)):
            image, label = image.to(self.configs.ExpConfig.device), label.to(self.configs.ExpConfig.device)
            optimizer.zero_grad()
            outputs = model(image)
            evidence = nfn.softplus(outputs, beta=20)
            alpha = evidence + 1
            total_loss = self.criterion(label, alpha, evidence, epoch)

            total_loss = torch.mean(total_loss)
            losses.append(total_loss.item())
            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        losses = np.asarray(losses).mean(axis=0)
        return train_loss / len(loader_train), losses

    def validate_EDL_epoch(self, model: nn.Module, loader_val: DataLoader, epoch: int) -> Tuple[float, float, float, float]:
        model.eval()
        val_loss = 0
        val_dice = []
        val_ece = []
        val_mi = []
        with torch.no_grad():
            for batch_idx, (image, label, _) in enumerate(tqdm(loader_val, total=len(loader_val), file=sys.stdout)):
                data, label = data.to(self.configs.ExpConfig.device), label.cuda(self.configs.ExpConfig.device)
                outputs = model(data)
                evidence = nfn.softplus(outputs, beta=20)
                alpha = evidence + 1
                soft_output = nfn.normalize(evidence, p=1, dim=1)
                edl_u = self.configs.NUM_CLASSES / torch.sum(alpha, dim=1, keepdim=False)

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

                total_loss = self.criterion(label, alpha, evidence, epoch)

                total_loss = torch.mean(total_loss)
                val_loss += total_loss.item()
        assert all([dsc <= 1.0001 for dsc in val_dice])
        return val_loss / len(loader_val), sum(val_dice) / len(val_dice), sum(val_ece) / len(val_ece), sum(val_mi) / len(val_mi)
