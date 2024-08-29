import shutil
from abc import ABC
from pathlib import Path
import torch
from yacs.config import CfgNode as cn
from utils.config_parser.base import *
from utils import make_experimental_folders

__all__ = ["ConfigParserYaml"]

VALID_TYPES = {tuple, list, str, int, float, bool, None, torch.Tensor}


def convert_to_dict(cfg_node, keys) -> dict:
    """
    Convert a configure node to dictionary.

    :param cfg_node: the input configuration node.
    :param keys: the list of keys.
    :return: a dictionary of CfgNode.
    """
    if not isinstance(cfg_node, cn):
        if type(cfg_node) not in VALID_TYPES:
            print(f"Key '{keys}' with value '{cfg_node}' is invalid.")
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, keys + [k])
        return cfg_dict


class ConfigParserYaml(ConfigParserBase, ABC):
    """
    The custom configuration parser for YAML config files.
    """

    def __init__(self, description: str):
        """
        :param description: the description for the parser.
        """
        super().__init__()
        self.parser.description = description
        # initial parser
        self.init_args()

    def _add_arguments(self):
        self.parser.add_argument("--cfg", default="./cfgs/config.yaml", help="the path of YAML config file.")

    def _print_args(self):
        super()._print_args()

    def init_args(self):
        self._add_arguments()

    def parse(self, cmd_args: list = None):
        """
        :param cmd_args: command arguments.
        :return: the output ConfigNode.
        """
        self.args = self.parser.parse_args(cmd_args)

        print("### Load the YAML config file...")
        with open(self.args.cfg, "r") as f:
            cfgs = cn.load_cfg(f)
            print(f"Successfully loading the config YAML file!")
        # initiate experiment configs
        cfgs.ExpConfig.exp_name = f"{cfgs.ExpConfig.name}_{cfgs.DataConfig.name}_{cfgs.LossConfig.name}_{cfgs.ExpConfig.experiment}"
        save_path = Path("./results")
        save_path.mkdir(parents=True, exist_ok=True)
        cfgs.ExpConfig.exp_path, cfgs.ExpConfig.model_path, cfgs.ExpConfig.figure_path, cfgs.ExpConfig.metric_path = make_experimental_folders(str(save_path), cfgs.ExpConfig.exp_name)
        # get device
        cfgs.ExpConfig.device = "cpu" if cfgs.ExpConfig.device_id == -1 else f"cuda:{cfgs.ExpConfig.device_id}"
        if not (Path(cfgs.ExpConfig.exp_path) / f"config.yaml").exists():
            shutil.copyfile(self.args.cfg, f"{cfgs.ExpConfig.exp_path}/config.yaml")

        # print configurations
        print(f"### Configurations:\n{cfgs}")
        return cfgs
