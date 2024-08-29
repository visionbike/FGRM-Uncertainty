import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from model import FGRMModel
from utils.config_parser import ConfigParserYaml


if __name__ == "__main__":
    seed = 0
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = ConfigParserYaml(description='FGRM Training Configuration')
    args = parser.parse()
    fgrm = FGRMModel(args)
    fgrm.run_FGRM()
