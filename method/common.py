from segmentation_models_pytorch import UnetPlusPlus
from segmentation_models_pytorch.losses.dice import DiceLoss
from utils import Task


def get_model_for_task(task, num_classes, encoder_weights='imagenet'):
    # TODO check the number of classes for cardiac (segmentation: 4, classification: ?)
    num_classes = num_classes
    if task == Task.SEGMENTATION:
        architecture = UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            decoder_channels=(1024, 512, 256, 128, 64),
            decoder_attention_type='scse',
            in_channels=3,
            classes=num_classes,
        )

        return architecture, []


def get_criterion_for_task(task, classes, experiment_name):
    if task == Task.SEGMENTATION:
        return DiceLoss(
            mode='multiclass',
            classes=classes,
            log_loss=False,
            from_logits=True,
            smooth=0.0000001,
            ignore_index=None,
        )
