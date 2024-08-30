from typing import Callable
import numpy as np

__all__ = ["EarlyStopping"]


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0., trace_func: Callable = print) -> None:
        """

        :param patience: how long to wait after last time validation loss improved. Default: 7.
        :param verbose: if True, prints a message for each validation loss improvement. Default: False.
        :param delta: minimum change in the monitored quantity to qualify as an improvement. Default: 0.
        :param trace_func: trace printing function. Default: print.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss: float) -> None:
        """

        :param val_loss: validation loss value.
        :return:
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        if score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
