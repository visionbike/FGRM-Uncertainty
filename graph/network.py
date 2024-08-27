import torch.nn as nn
from segmentation_models_pytorch import UnetPlusPlus

__all__ = ["get_network"]


def get_network(task: str, num_classes: int, encoder_weights: None | str = "imagenet") -> nn.Module:
    if task == "segmentation":
        network = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            decoder_channels=[1024, 512, 256, 128, 64],
            decoder_attention_type="scse",
            in_channels=3,
            classes=num_classes,
        )
    else:
        raise NotImplementedError
    return network
