"""Нарезка RusTitW на строки-кропы.

Запускается на Kaggle: исходный датасет весит ~175 ГБ и лежит там, а скачиваем мы только
результат — папку кропов и crops.csv (~180 тыс. маленьких JPEG).
"""
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from src.utils import to_rgb

MAX_H = 64          # кропы храним высотой <= 64 px: вход модели 48, остаётся запас под аугментации
MIN_SIDE = 8        # боксы меньше 8 px отбрасываем (в тесте минимум 10 px)
MIN_ASPECT = 0.5    # вертикальный текст (w/h < 0.5) отбрасываем: в тесте его почти нет
MARGIN = 0.1        # поля вокруг бокса, доля высоты строки: из них при обучении делаем случайный сдвиг кропа
JPEG_Q = 95


def parse_boxes(s: str) -> list:
    """box_and_label — JSON-строка вида [[{left, top, width, height, label, shape}, ...]].
    Координаты — доли от размера картинки (0..1)."""
    boxes = []
    for item in json.loads(s):
        boxes.extend([item] if isinstance(item, dict) else item)
    return boxes


def split_lines(box: dict, W: int, H: int) -> list:
    """Переводим бокс в пиксели. Боксы в RusTitW бывают целыми абзацами (label с \\n),
    а тестовые кропы — отдельные строки. Поэтому многострочный бокс режем на n равных
    горизонтальных полос, n = число строк в label. Разрез приблизительный, но метка
    ориентации от этого не портится: её задаём мы сами поворотом."""
    x0, y0 = max(0.0, box["left"] * W), max(0.0, box["top"] * H)
    x1 = min(float(W), (box["left"] + box["width"]) * W)
    y1 = min(float(H), (box["top"] + box["height"]) * H)
    lines = [l.strip() for l in str(box.get("label", "")).split("\n") if l.strip()]
    n = max(1, len(lines))
    step = (y1 - y0) / n
    return [(x0, y0 + i * step, x1, y0 + (i + 1) * step, lines[i] if lines else "") for i in range(n)]


def process_image(task: tuple):
    """Одна исходная картинка -> список кропов-строк. Возвращает (записи, прочиталась_ли_картинка)."""
    split, source, img_path, boxes_json, out_dir = task
    if not isinstance(boxes_json, str):
        return [], True
    try:
        im = to_rgb(Image.open(img_path))
    except Exception:
        return [], False  # не прочиталась: считаем отдельно, чтобы не потерять данные молча
    W, H = im.size
    stem, recs = Path(img_path).stem, []
    for bi, box in enumerate(parse_boxes(boxes_json)):
        if box.get("shape", "rectangle") != "rectangle":
            continue
        for li, (x0, y0, x1, y1, text) in enumerate(split_lines(box, W, H)):
            w, h = x1 - x0, y1 - y0
            if w < MIN_SIDE or h < MIN_SIDE or w / h < MIN_ASPECT:
                continue
            m = MARGIN * h
            cx0, cy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
            cx1, cy1 = min(W, int(np.ceil(x1 + m))), min(H, int(np.ceil(y1 + m)))
            crop = im.crop((cx0, cy0, cx1, cy1))
            s = min(1.0, MAX_H / crop.height)
            if s < 1:
                crop = crop.resize((max(1, round(crop.width * s)), MAX_H), Image.Resampling.LANCZOS)
            rel = Path(split) / source / f"{stem}_{bi}_{li}.jpg"
            crop.save(Path(out_dir) / rel, quality=JPEG_Q)
            recs.append(dict(path=str(rel), split=split, source=source, image=stem, text=text,
                             w=crop.width, h=crop.height,
                             # где внутри кропа лежит сам бокс (без полей) — для сдвигов при обучении
                             bx0=(x0 - cx0) * s, by0=(y0 - cy0) * s,
                             bx1=min(crop.width, (x1 - cx0) * s), by1=min(crop.height, (y1 - cy0) * s)))
    return recs, True


def prepare_crops(base: Path, out: Path, synth_frac: float = 0.025, seed: int = 42) -> pd.DataFrame:
    """Режем весь real (train и test) и долю synth_frac картинок из train/synth.
    synth из test не берём: валидация — только реальные фото из test/real."""
    base, out = Path(base), Path(out)
    sources = [("train", "real", 1.0), ("test", "real", 1.0), ("train", "synth", synth_frac)]
    tasks = []
    for split, source, frac in sources:
        root = base / split / source
        cols = pd.read_csv(root / "info.csv", nrows=0).columns
        use = ["image_name", "box_and_label"] + (["image_path"] if "image_path" in cols else [])
        df = pd.read_csv(root / "info.csv", usecols=use)
        if frac < 1:
            df = df.sample(frac=frac, random_state=seed)  # фиксированный seed -> та же подвыборка
        paths = df["image_path"] if "image_path" in df else "images/" + df["image_name"]
        (out / split / source).mkdir(parents=True, exist_ok=True)
        tasks += [(split, source, str(root / p), b, str(out)) for p, b in zip(paths, df["box_and_label"])]
        print(f"{split}/{source}: картинок {len(df)}")

    recs, n_fail = [], 0
    with Pool(os.cpu_count()) as pool:
        for r, ok in tqdm(pool.imap_unordered(process_image, tasks, chunksize=16), total=len(tasks)):
            recs += r
            n_fail += not ok
    # imap_unordered отдаёт результаты в случайном порядке — сортируем, чтобы csv был детерминированным
    crops = pd.DataFrame(recs).sort_values("path").reset_index(drop=True)
    crops.to_csv(out / "crops.csv", index=False)
    print("не прочитались картинки:", n_fail)
    return crops


def check_exif(base: Path) -> pd.Series:
    """Считаем EXIF-ориентации real-фото. Если бы пиксели хранились «лёжа» или вверх ногами
    (тег != 1), боксы разметки не совпали бы с сырыми пикселями, а при теге 3 перевёрнутый
    текст получил бы метку «стоит правильно». Читается только заголовок файла."""
    orients = []
    for split in ["train", "test"]:
        root = Path(base) / split / "real"
        df = pd.read_csv(root / "info.csv", usecols=lambda c: c in {"image_name", "image_path"})
        paths = df["image_path"] if "image_path" in df else "images/" + df["image_name"]
        for p in tqdm(paths, desc=split):
            with Image.open(root / p) as im:
                orients.append(im.getexif().get(0x0112, 1))
    return pd.Series(orients, name="exif_orientation").value_counts()
