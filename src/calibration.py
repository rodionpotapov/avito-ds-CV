"""Калибровка и анализ валидации: temperature scaling, TTA с поворотом, диаграмма надёжности,
разбор ошибок по группам и проверка на утечку через артефакты JPEG."""
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data import CropDataset, resize_pad, to_tensor
from src.train import evaluate
from src.utils import to_rgb


# ---------------------------------------------------------------- temperature scaling

def _nll(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_temperature(logits: np.ndarray, y: np.ndarray, t_min: float = 0.05, t_max: float = 20.0,
                    iters: int = 100) -> float:
    """T = argmin NLL(σ(z / T), y). NLL выпукла по β = 1 / T, поэтому хватает поиска
    золотым сечением по β на отрезке [1 / t_max, 1 / t_min]."""
    a, b = 1 / t_max, 1 / t_min
    g = (np.sqrt(5) - 1) / 2
    for _ in range(iters):
        c, d = b - g * (b - a), a + g * (b - a)
        if _nll(logits * c, y) < _nll(logits * d, y):
            b = d
        else:
            a = c
    return float(2 / (a + b))


def crossfit_temperature(logits: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 42):
    """Честная оценка калибровки: исходные фото делим на две половины, T подбираем на одной
    и применяем к другой (и наоборот). Возвращает откалиброванные логиты и две температуры."""
    uniq = np.unique(groups)
    half = np.random.default_rng(seed).permutation(uniq)[: len(uniq) // 2]
    mask = np.isin(groups, half)
    z_cal, temps = np.empty(len(logits), dtype=np.float64), []
    for m in (mask, ~mask):
        t = fit_temperature(logits[~m], y[~m])
        z_cal[m] = logits[m] / t
        temps.append(t)
    return z_cal, temps


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Корзины по предсказанной вероятности: средняя уверенность против реальной доли y=1.
    ECE — взвешенное среднее расхождение между ними (0 — идеальная калибровка)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    table = (pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin")
             .agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index())
    ece = float((table["n"] * (table["mean_p"] - table["frac_pos"]).abs()).sum() / len(p))
    return table, ece


# ---------------------------------------------------------------- TTA

@torch.no_grad()
def predict_logits_pair(model, loader, device):
    """Логиты для кропа и для него же, повёрнутого на 180°. Поворачиваем уже готовый тензор:
    при паддинге по центру это то же самое, что повернуть кроп до препроцессинга
    (с точностью до сдвига на 1 px при нечётном паддинге)."""
    model.eval()
    z, z_rot = [], []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z.append(model(x).float().squeeze(1).cpu())
        z_rot.append(model(torch.rot90(x, 2, dims=(2, 3))).float().squeeze(1).cpu())
    return torch.cat(z).numpy(), torch.cat(z_rot).numpy()


def tta_logits(z: np.ndarray, z_rot: np.ndarray) -> np.ndarray:
    """Если кроп перевёрнут, его повёрнутая копия стоит правильно — у неё логит «с обратным знаком».
    Усредняем оба свидетельства."""
    return (z - z_rot) / 2


# ---------------------------------------------------------------- разбор ошибок

def text_type(t) -> str:
    t = t if isinstance(t, str) else ""
    cyr, lat = bool(re.search("[А-Яа-яЁё]", t)), bool(re.search("[A-Za-z]", t))
    if cyr and lat:
        return "кириллица + латиница"
    if cyr:
        return "кириллица"
    if lat:
        return "латиница"
    if re.search("[0-9]", t):
        return "только цифры"
    return "без букв и цифр"


def group_report(df: pd.DataFrame, logits: np.ndarray, y: np.ndarray, by: str) -> pd.DataFrame:
    """Метрики по группам: сколько кропов, accuracy, 1 − Brier."""
    rows = []
    for g, idx in df.groupby(by, observed=True).indices.items():
        m = evaluate(logits[idx], y[idx])
        rows.append({by: g, "кропов": len(idx), "доля": round(len(idx) / len(df), 3),
                     "acc": round(m["acc"], 4), "1 - Brier": round(m["score"], 4)})
    return pd.DataFrame(rows).sort_values("кропов", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- проверка на утечку

class ShiftedValDataset(CropDataset):
    """Та же валидация, но кроп расширен на 1–7 px симметрично (слева и справа на одно и то же
    число пикселей, сверху и снизу — тоже), берём реальные пиксели из сохранённых полей.
    Сдвигается «фаза» сетки JPEG-блоков 8×8 относительно краёв кропа, а текст не меняется,
    и расширение симметрично, поэтому поворот его не выдаёт. Если модель смотрит на буквы,
    а не на сетку сжатия, метрики почти не изменятся."""

    def __getitem__(self, i):
        img = to_rgb(Image.open(self.root / self.paths[i]))
        W, H = img.size
        x0, y0, x1, y1 = self.boxes[i]
        x0, y0 = int(x0), int(y0)
        x1, y1 = max(x0 + 1, int(np.ceil(x1))), max(y0 + 1, int(np.ceil(y1)))
        rng = np.random.default_rng(1000 + i)
        dx = min(x0, W - x1, int(rng.integers(1, 8)))
        dy = min(y0, H - y1, int(rng.integers(1, 8)))
        img = img.crop((x0 - dx, y0 - dy, x1 + dx, y1 + dy))
        y = int(self.labels[i])
        if y:
            img = img.transpose(Image.Transpose.ROTATE_180)
        return to_tensor(resize_pad(img)), torch.tensor(y, dtype=torch.float32)


def update_checkpoint(path, **fields) -> None:
    """Дописываем в чекпоинт параметры инференса (температура, TTA), не трогая веса."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt.update(fields)
    torch.save(ckpt, path)
