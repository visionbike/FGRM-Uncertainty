from pathlib import Path

__all__ = ["make_experimental_folders"]


def make_experimental_folders(path: str, name: str) -> [str, str, str, str]:
    """
    Create experiment folder with subfolders figures, models, segmentations.

    :param path: where the experiment folder is created.
    :param name: all experiment related outputs will be here.
    :return: list of exp_path, models_path, figures_path, seg_out_path
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
    return str(path_result), str(path_model), str(path_figure), str(path_metric)
