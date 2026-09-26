"""Общие утилиты: фиксация seed, выбор устройства, приведение картинки к RGB."""
import os
import random

import numpy as np
import torch
from PIL import Image


def seed_everything(seed: int = 42) -> None:
    """Фиксируем все генераторы случайных чисел, чтобы запуски воспроизводились."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """cuda (RTX / Colab) -> mps (Mac на Apple Silicon) -> cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_rgb(img: Image.Image) -> Image.Image:
    """Прозрачность (RGBA/LA/P) кладём на белый фон, а не на чёрный; остальное — просто в RGB."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")
