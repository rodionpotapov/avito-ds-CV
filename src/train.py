"""Обучение и оценка: метрики, цикл по эпохам, предсказания, сохранение и загрузка чекпоинта."""
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.model import build_model


# ---------------------------------------------------------------- метрики

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def evaluate(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> dict:
    """1 − Brier (главная метрика), Brier, logloss и accuracy по логитам.
    temperature делит логит перед сигмоидой (калибровка, раздел 6)."""
    z = logits / temperature
    p = sigmoid(z)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(np.mean(np.logaddexp(0, z) - y * z))  # устойчивая форма BCE
    acc = float(np.mean((p > 0.5) == (y == 1)))
    return {"score": 1 - brier, "brier": brier, "logloss": logloss, "acc": acc}


# ---------------------------------------------------------------- предсказания

@torch.no_grad()
def predict_logits(model: nn.Module, loader, device, progress: bool = False) -> np.ndarray:
    """Логиты для всего loader'а в исходном порядке (loader без shuffle)."""
    model.eval()
    batches = tqdm(loader, desc="валидация", leave=False) if progress else loader
    out = [model(x.to(device, non_blocking=True)).float().squeeze(1).cpu() for x, _ in batches]
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- обучение

def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Линейный разогрев lr за warmup_steps шагов, затем косинусное затухание до нуля."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _amp(device):
    """Смешанная точность (fp16) только на CUDA: там она ускоряет обучение; на MPS/CPU — fp32."""
    use_amp = device.type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
    return ctx, torch.amp.GradScaler("cuda", enabled=use_amp)


def save_checkpoint(model: nn.Module, path, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, **meta}, path)


def load_checkpoint(path, device):
    """Пустая архитектура + наши веса (ничего не скачивается). Возвращает (модель, метаданные)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt.pop("state_dict"))
    return model.to(device).eval(), ckpt


def time_steps(model: nn.Module, loader, device, n_steps: int = 30, warmup: int = 5) -> dict:
    """Короткий замер: сколько картинок в секунду отдаёт DataLoader и сколько проходит обучение
    (загрузка + forward + backward). Первые warmup батчей не считаем: запуск воркеров и прогрев."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    amp_ctx, scaler = _amp(device)

    assert len(loader) > warmup + 1, "в loader слишком мало батчей для замера"

    def run(train_step: bool) -> float:
        n_img, t0 = 0, None
        for step, (x, y) in enumerate(loader):
            if step == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0, n_img = time.perf_counter(), 0
            if step >= warmup + n_steps:
                break
            if train_step:
                x, y = x.to(device), y.to(device)
                with amp_ctx:
                    loss = loss_fn(model(x).squeeze(1), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss.item()  # синхронизация с устройством, чтобы замер был честным
            n_img += len(x)
        return n_img / (time.perf_counter() - t0)

    return {"loader_img_s": run(False), "train_img_s": run(True)}


def fit(model: nn.Module, train_loader, val_loader, val_labels: np.ndarray, device, *,
        epochs: int = 12, lr: float = 1e-3, weight_decay: float = 1e-4, warmup_epochs: int = 1,
        patience: int = 3, ckpt_path="weights/model.pt", meta: dict | None = None) -> pd.DataFrame:
    """Обучение с AdamW + разогрев + cosine. После каждой эпохи — метрики на валидации;
    лучший по val Brier чекпоинт сохраняется в ckpt_path. Ранняя остановка, если Brier
    не улучшался patience эпох подряд. Возвращает историю обучения по эпохам."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    sched = make_scheduler(opt, epochs * steps_per_epoch, warmup_epochs * steps_per_epoch)
    loss_fn = nn.BCEWithLogitsLoss()
    amp_ctx, scaler = _amp(device)

    history, best, bad_epochs = [], float("inf"), 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum, n = 0.0, 0
        bar = tqdm(train_loader, desc=f"эпоха {epoch}/{epochs}", leave=False, mininterval=1)
        for x, y in bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx:
                loss = loss_fn(model(x).squeeze(1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(x)
            n += len(x)
            bar.set_postfix(loss=f"{loss_sum / n:.4f}", img_s=f"{n / (time.perf_counter() - t0):.0f}", refresh=False)

        val = evaluate(predict_logits(model, val_loader, device, progress=True), val_labels)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / n,
               "val_loss": val["logloss"], "val_brier": val["brier"], "val_score": val["score"],
               "val_acc": val["acc"], "minutes": (time.perf_counter() - t0) / 60}
        history.append(row)

        improved = val["brier"] < best
        if improved:
            best, bad_epochs = val["brier"], 0
            save_checkpoint(model, ckpt_path, epoch=epoch, val_brier=best, **(meta or {}))
        else:
            bad_epochs += 1
        print(f"эпоха {epoch:2d} | train loss {row['train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
              f"val 1-Brier {row['val_score']:.4f} | val acc {row['val_acc']:.4f} | "
              f"{row['minutes']:.1f} мин{' | сохранён' if improved else ''}")
        if bad_epochs >= patience:
            print(f"ранняя остановка: {patience} эпохи без улучшения val Brier")
            break

    return pd.DataFrame(history)
