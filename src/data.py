"""Препроцессинг, аугментации и датасеты для классификации поворота кропа (0° / 180°).

Главные правила (от них зависит, выучит ли модель ориентацию, а не артефакт):
1. Поворот на 180° делается ДО resize_pad. Тогда паддинг одинаковый для обоих классов
   и не подсказывает метку.
2. Никаких отражений (flip): зеркального текста в тесте нет.
3. Всё, что зависит от ориентации (сдвиги кропа, наклон), делаем до поворота;
   деградации (низкое разрешение, blur, JPEG, шум) — после поворота, чтобы их
   артефакты не были «привязаны» к одной ориентации.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

from src.utils import to_rgb

IMG_H, IMG_W = 48, 192                   # вход модели: как у классификатора ориентации PaddleOCR
MEAN = (0.485, 0.456, 0.406)             # нормализация ImageNet (backbone предобучен на ImageNet)
STD = (0.229, 0.224, 0.225)
PAD_COLOR = tuple(round(m * 255) for m in MEAN)  # паддинг цветом среднего -> после нормализации ≈ 0
BILINEAR = Image.Resampling.BILINEAR


# ---------------------------------------------------------------- препроцессинг (train = val = test)

def resize_pad(img: Image.Image, h: int = IMG_H, w: int = IMG_W) -> Image.Image:
    """Высота -> h с сохранением пропорций. Если кроп шире w — сжимаем по ширине до w
    (вертикальная форма букв, по которой видно верх/низ, сохраняется). Если уже — паддинг
    по центру цветом среднего."""
    new_w = max(1, min(w, round(img.width * h / img.height)))
    img = img.resize((new_w, h), BILINEAR)
    if new_w == w:
        return img
    canvas = Image.new("RGB", (w, h), PAD_COLOR)
    canvas.paste(img, ((w - new_w) // 2, 0))
    return canvas


_MEAN = np.array(MEAN, dtype=np.float32)
_STD = np.array(STD, dtype=np.float32)


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB (H, W, 3) uint8 -> нормализованный тензор (3, H, W) float32."""
    x = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Обратно в картинку (H, W, 3) в [0, 1] — только для визуализации."""
    return np.clip(t.numpy().transpose(1, 2, 0) * _STD + _MEAN, 0, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    """Единый путь для валидации и теста: RGB -> resize_pad -> тензор."""
    return to_tensor(resize_pad(to_rgb(img)))


# ---------------------------------------------------------------- аугментации

def _border_color(img: Image.Image) -> tuple:
    """Медианный цвет рамки кропа — им заливаем углы после наклона, чтобы не было чёрных треугольников."""
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def jitter_crop(img: Image.Image, box, rng: np.random.Generator) -> Image.Image:
    """Случайный кроп вокруг бокса: имитирует неточность детектора.
    Каждая граница — случайно между «впритык к боксу» и «всё сохранённое поле» (10% высоты).
    Иногда подрезаем текст сбоку/сверху: в тесте тоже бывают обрезанные буквы."""
    W, H = img.size
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    x0, x1 = rng.uniform(0, bx0), rng.uniform(bx1, W)
    y0, y1 = rng.uniform(0, by0), rng.uniform(by1, H)
    if rng.random() < 0.3:
        cut = rng.uniform(0, 0.1) * bw
        if rng.random() < 0.5:
            x0 += cut
        else:
            x1 -= cut
    if rng.random() < 0.2:
        cut = rng.uniform(0, 0.1) * bh
        if rng.random() < 0.5:
            y0 += cut
        else:
            y1 -= cut
    x0, y0 = int(x0), int(y0)
    x1 = min(W, max(x0 + 4, int(np.ceil(x1))))
    y1 = min(H, max(y0 + 4, int(np.ceil(y1))))
    return img.crop((x0, y0, x1, y1))


def augment_upright(img: Image.Image, box, rng: np.random.Generator) -> Image.Image:
    """Аугментации, которые делаем ДО поворота на 180° (текст ещё стоит правильно)."""
    img = jitter_crop(img, box, rng)
    # горизонтальное растяжение/сжатие: у наших кропов w/h ~2.5, у теста ~5 и длинные строки
    # сжимаются в resize_pad -> учим модель узким буквам
    if rng.random() < 0.5:
        f = float(np.exp(rng.uniform(np.log(0.5), np.log(1.5))))
        img = img.resize((max(4, round(img.width * f)), img.height), BILINEAR)
    # небольшой наклон: текст в тестовых боксах бывает под углом, модель не должна путать
    # «наклонён» и «перевёрнут»
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-7, 7), resample=BILINEAR, expand=True,
                         fillcolor=_border_color(img))
    # цвет: яркость, контраст, насыщенность; иногда ч/б и инверсия (светлый текст на тёмном)
    if rng.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 1.5))
    if rng.random() < 0.1:
        img = ImageOps.grayscale(img).convert("RGB")
    if rng.random() < 0.1:
        img = ImageOps.invert(img)
    return img


def degrade(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Деградации качества ПОСЛЕ поворота: пиксельные, размытые и пережатые кропы из EDA."""
    if rng.random() < 0.3:  # низкое разрешение: сжать до высоты 12–32 px и растянуть обратно
        # (в тесте 5% кропов ниже 16 px и четверть ниже 27 px)
        th = int(rng.integers(12, 33))
        if th < img.height:
            small = img.resize((max(2, round(img.width * th / img.height)), th), BILINEAR)
            img = small.resize(img.size, BILINEAR)
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.3:  # JPEG-артефакты
        buf = BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(20, 81)))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    if rng.random() < 0.1:  # гауссов шум
        a = np.asarray(img, dtype=np.float32)
        a += rng.normal(0, rng.uniform(3, 12), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


# ---------------------------------------------------------------- датасеты

class CropDataset(Dataset):
    """Кропы RusTitW. Метку y создаём сами: y=1 — повернули на 180°, y=0 — оставили.

    train=True:  случайный поворот (монетка заново на каждой эпохе) + аугментации.
    train=True, paired=True: пара из одного и того же кропа в двух ориентациях (y=0 и y=1)
                 с одинаковыми аугментациями — отличается только поворот (как в RotNet).
    train=False: валидация. Кроп строго по боксу (без полей, как у детектора),
                 без аугментаций, поворот фиксирован seed'ом -> набор одинаковый при каждом запуске.
    """

    def __init__(self, df, root, train: bool, seed: int = 42, paired: bool = False):
        self.root = Path(root)
        self.paths = df["path"].tolist()
        self.boxes = df[["bx0", "by0", "bx1", "by1"]].to_numpy(np.float32)
        self.train = train
        self.paired = train and paired
        self.labels = None if train else np.random.default_rng(seed).integers(0, 2, len(df))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = to_rgb(Image.open(self.root / self.paths[i]))
        if self.train:
            # генератор на каждый вызов из torch RNG: в воркерах DataLoader torch сам даёт
            # каждому воркеру свой seed от seed'а загрузчика -> аугментации воспроизводимы
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            img = augment_upright(img, self.boxes[i], rng)
            if self.paired:
                # один seed деградаций для обеих копий: у пары отличается только ориентация
                deg_seed = int(rng.integers(0, 2**31 - 1))
                up = degrade(img, np.random.default_rng(deg_seed))
                rot = degrade(img.transpose(Image.Transpose.ROTATE_180), np.random.default_rng(deg_seed))
                x = torch.stack([to_tensor(resize_pad(up)), to_tensor(resize_pad(rot))])
                return x, torch.tensor([0.0, 1.0])
            y = int(rng.random() < 0.5)
            if y:
                img = img.transpose(Image.Transpose.ROTATE_180)
            img = degrade(img, rng)
        else:
            x0, y0, x1, y1 = self.boxes[i]
            img = img.crop((int(x0), int(y0), max(int(x0) + 1, int(np.ceil(x1))),
                            max(int(y0) + 1, int(np.ceil(y1)))))
            y = int(self.labels[i])
            if y:
                img = img.transpose(Image.Transpose.ROTATE_180)
        return to_tensor(resize_pad(img)), torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    """Тестовые кропы Авито: только препроцессинг, без поворотов и аугментаций."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess(Image.open(self.paths[i])), i


def collate_pairs(batch):
    """Пары (2, 3, H, W) склеиваем в обычный батч: B пар -> 2B картинок, метки [0, 1, 0, 1, ...]."""
    xs, ys = zip(*batch)
    return torch.cat(xs), torch.cat(ys)


def make_loader(ds, shuffle: bool, batch_size: int = 256, num_workers: int = 4, seed: int = 42):
    """DataLoader с фиксированным генератором: он задаёт порядок батчей и seed'ы воркеров,
    поэтому аугментации воспроизводятся от запуска к запуску.
    Для парного датасета batch_size — число пар, в батче будет вдвое больше картинок."""
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed),
                      collate_fn=collate_pairs if getattr(ds, "paired", False) else None,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
