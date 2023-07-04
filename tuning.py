import re
from typing import Union

from torch import nn
from torchvision.models import get_model, ResNet, AlexNet, ConvNeXt, DenseNet, EfficientNet, GoogLeNet, Inception3, \
    MaxVit, MNASNet, MobileNetV2, MobileNetV3, RegNet, ShuffleNetV2, SqueezeNet, SwinTransformer, VGG, VisionTransformer

BUILTIN_TUNERS = {}


def register_tuner(fn):
    key = fn.__name__
    BUILTIN_TUNERS[key] = fn
    return fn


def get_model_tuner(name: str):
    name = re.sub(r"[\d_].*", "", name)
    try:
        fn = BUILTIN_TUNERS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name}")
    return fn


def get_tuned_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    model = get_model(name, weights="DEFAULT")
    fn = get_model_tuner(name)
    fn(model, num_classes)
    return model


def get_tuned_linear(m: nn.Linear, num_classes: int) -> nn.Linear:
    num_features = m.in_features
    return nn.Linear(num_features, num_classes, bias=m.bias)


def tune_linear_in_sequential(seq: nn.Sequential, num_classes: int):
    seq[-1] = get_tuned_linear(seq[-1], num_classes)


@register_tuner
def alexnet(model: AlexNet, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def convnext(model: ConvNeXt, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def densenet(model: DenseNet, num_classes: int):
    model.classifier = get_tuned_linear(model.classifier, num_classes)


@register_tuner
def efficientnet(model: EfficientNet, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def googlenet(model: GoogLeNet, num_classes: int):
    if model.aux_logits:
        m: nn.Dropout = model.aux1.dropout
        dropout: float = m.p
        cls = type(model.aux1)
        model.aux1 = cls(512, num_classes, dropout=dropout)
        model.aux2 = cls(528, num_classes, dropout=dropout)

    model.fc = get_tuned_linear(model.fc, num_classes)


@register_tuner
def inception(model: Inception3, num_classes: int):
    model.fc = get_tuned_linear(model.fc, num_classes)


@register_tuner
def maxvit(model: MaxVit, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def mnasnet(model: MNASNet, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def mobilenet(model: Union[MobileNetV2, MobileNetV3], num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def regnet(model: RegNet, num_classes: int):
    model.fc = get_tuned_linear(model.fc, num_classes)


@register_tuner
def resnet(model: ResNet, num_classes: int):
    model.fc = get_tuned_linear(model.fc, num_classes)


@register_tuner
def resnext(model: ResNet, num_classes: int):
    resnet(model, num_classes)


@register_tuner
def shufflnet(model: ShuffleNetV2, num_classes: int):
    model.fc = get_tuned_linear(model.fc, num_classes)


@register_tuner
def squeezenet(model: SqueezeNet, num_classes: int):
    m: nn.Conv2d = model.classifier[1]
    model.classifier[1] = nn.Conv2d(m.in_channels, num_classes, kernel_size=m.kernel_size[0])


@register_tuner
def swin(model: SwinTransformer, num_classes: int):
    model.head = get_tuned_linear(model.head, num_classes)


@register_tuner
def vgg(model: VGG, num_classes: int):
    tune_linear_in_sequential(model.classifier, num_classes)


@register_tuner
def vit(model: VisionTransformer, num_classes: int):
    tune_linear_in_sequential(model.heads, num_classes)


@register_tuner
def wide(model: ResNet, num_classes: int):
    resnet(model, num_classes)
