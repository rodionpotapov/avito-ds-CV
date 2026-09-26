"""Модель: MobileNetV3-Large (torchvision) с одним выходом — логитом поворота на 180°."""
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_model(pretrained: bool = True) -> nn.Module:
    """MobileNetV3-Large. pretrained=True — веса ImageNet (для обучения);
    False — пустая архитектура (для инференса: веса потом грузим свои, ничего не скачиваем).
    Родную голову на 1000 классов ImageNet заменяем на Dropout + Linear(960 -> 1)."""
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[0].in_features  # 960 каналов после global average pooling
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 1))
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def count_flops(model: nn.Module, input_size=(1, 3, 48, 192)) -> int:
    """FLOPs одного прохода (умножение и сложение считаются отдельно, MACs = FLOPs / 2).
    Считаются свёртки и линейные слои — это почти все вычисления сети."""
    model.eval()
    counter = FlopCounterMode(display=False)
    with counter:
        model(torch.zeros(input_size))
    return counter.get_total_flops()
