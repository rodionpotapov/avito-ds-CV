# Авито: поворот текстового кропа на 180°

Для каждого из 20 000 тестовых кропов OCR-детектора предсказываем `p_180` — вероятность, что текст повёрнут на 180°. Метрика — 1 − Brier.

## Решение

- MobileNetV3-Large (torchvision, веса ImageNet), вход 3 × 48 × 192, один выход — логит.
- Обучение на строках-кропах открытого датасета RusTitW. Метки создаём сами: кроп как есть — 0, повёрнутый на 180° — 1 (пары, как в RotNet).
- TTA с поворотом и temperature scaling.

| | |
|---|---|
| валидация (RusTitW test/real), 1 − Brier | 0.9586 |
| лидерборд, 1 − Brier | 0.9675 |
| размер | 2.97 млн параметров, 11.5 МБ |
| скорость на CPU (Ryzen 5 5600, батч 64) | ~1 мс на кроп |

Подробности, замеры и выводы — в `avito.ipynb`.

## Запуск

Python 3.13, torch 2.14.

```
pip install -r requirements.txt
```

1. Положить данные Авито в `test/`: `test/sample_submission.csv` и `test/test/images/*.png`.
2. Запустить `avito.ipynb` целиком из корня репозитория.

По умолчанию ноутбук загружает веса из `weights/model.pt` и пишет `submission.csv`. Видеокарта не нужна.

## Флаги

| где | флаг | по умолчанию | что менять |
|---|---|---|---|
| ячейка 0.3 | `TRAIN` | `False` — загрузить веса | `True` — обучить с нуля; нужны кропы RusTitW в `data/rustitw_crops` |
| ячейка 0.3 | `PREPARE_DATA` | `False` | `True` — нарезать RusTitW на кропы; только на Kaggle с подключённым датасетом (раздел 2) |
| ячейка 0.3 | `NUM_WORKERS` | `10` | для инференса можно уменьшить, результат не изменится; для обучения не менять, иначе изменятся аугментации |
| ячейка 8.1 | `INFER_DEVICE` | `torch.device("cpu")` | `DEVICE`, чтобы считать тест на GPU: быстрее, но `p_180` отличается от эталонного `submission.csv` до ~0.006 (TF32 на RTX 30xx/40xx), на лидерборде разница ~1e-6 |

Обучение и валидация сами выбирают устройство (`DEVICE`: cuda → mps → cpu), на CUDA включается fp16. Чтобы обучать на GPU под Windows, torch ставится с CUDA:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Структура

```
avito.ipynb        # всё решение: EDA, данные, обучение, калибровка, скорость, инференс
submission.csv     # итоговые предсказания
weights/           # model.pt (веса, T, флаг TTA), history.csv
src/               # модули, их создаёт ноутбук
```

## Open-source

- RusTitW — [arXiv:2303.16531](https://arxiv.org/abs/2303.16531), [Kaggle](https://www.kaggle.com/datasets/hardtype/rustitw-russian-language-visual-text-recognition)
- MobileNetV3-Large из torchvision
