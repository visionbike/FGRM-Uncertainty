from typing import Callable, List
import enum
import numpy as np
from pathlib import Path

__all__ = [
    "Task",
    "Organ",
    "norm_minmax",
    "make_experimental_folders",
    "EarlyStopping",
    "AverageMeter",
    "ProgressMeter"
]


class Task(enum.Enum):
    SEGMENTATION = "segmentation"


class Organ(enum.Enum):
    ESD = "ESD"
    LC = "LC"


def norm_minmax(data: np.ndarray, nmax: float = 1., nmin: float = 0.) -> np.ndarray:
    """
    Min-max normalization.

    :param data: input data.
    :param nmax: new maximum value. Default: 1.
    :param nmin: new minimum value. Default: 0.
    :return:
    """
    return (data - data.min()) * ((nmax - nmin) / (data.max() - data.min() + 1e-8)) + nmin


def make_experimental_folders(path: str, name: str) -> [Path, Path, Path]:
    """
    Create experiment folder with subfolders figures, models, segmentations.

    :param path: where the experiment folder is created.
    :param name: all experiment related outputs will be here.
    :return: list of models_path, figures_path, seg_out_path
    """
    path_base = Path(path)
    path_result = path_base / name
    path_figure = path_result / "figures"
    path_model = path_result / "models"
    path_metric = path_result / "metrics"
    #
    if not path_result.exists():
        path_result.mkdir(parents=True, exist_ok=True)
    if not path_figure.exists():
        path_figure.mkdir(parents=True, exist_ok=True)
    if not path_model.exists():
        path_model.mkdir(parents=True, exist_ok=True)
    if not path_metric.exists():
        path_metric.mkdir()
    #
    return path_model, path_figure, path_metric


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
        if score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self, name: str, fmt: str = ":f") -> None:
        """

        :param name: metric name.
        :param fmt: format string. Default: ":f".
        """
        self.name = name
        self.fmt = fmt
        #
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        #
        self.reset()

    def reset(self) -> None:
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val: float, n: int = 1) -> None:
        """

        :param val: validation value.
        :param n: batch size. Default: 1.
        :return:
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    Progress meter for training and validation.
    """
    def __init__(self, num_batches: int, meters: List[Callable], prefix: str = "") -> None:
        """

        :param num_batches: number of batches.
        :param meters: meter functions.
        :param prefix: prefix to prepend to each metric name. Default: "".
        """
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        """

        :param num_batches: number of batches.
        :return:
        """
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
