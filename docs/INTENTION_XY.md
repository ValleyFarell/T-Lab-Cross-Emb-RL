# Декодер намерения в координаты `(x, y)`

## Интуиция

Намерение агента состоит из 128 чисел и не объясняет человеку, в какую часть лабиринта хочет двигаться робот. Вспомогательная модель учится переводить такое намерение в понятную точку на карте. Это позволяет визуализировать поведение и привязывать предложенную подцель к реальным офлайн-состояниям.

## Мотивация

Даже полезное намерение сложно анализировать, если нельзя сопоставить его с картой. Отдельный декодер не изменяет исходный агент, но делает возможными карты промежуточных целей, проверку ошибок и метод проекции подцели.

## Математика в трёх предложениях

Для каждого офлайн-состояния строится пара `z=normalize(B(s))` и `q=s[:2]`. Декодер `D_xy:ℝ¹²⁸→ℝ²` обучается предсказывать `q` по `z`, минимизируя ошибку восстановления координат. Его качество проверяется на целых отложенных траекториях, которые не участвовали в обучении.

## Архитектура и ошибка

$$
D_{xy}(z)\approx(x,y),
\qquad
\mathcal L=\frac1N\sum_i\|D_{xy}(z_i)-q_i\|_2^2.
$$

Стандартная архитектура: `128 → 512 → 512 → 512 → 2`. Первый скрытый блок использует нормализацию слоя и `tanh`, следующие — `GELU`; выходной слой линейный.

Покомпонентная ошибка вычисляется как

$$
\operatorname{RMSE}_{xy}=\sqrt{\operatorname{mean}(\Delta x^2,\Delta y^2)},
$$

а пространственная — как

$$
\operatorname{RMSE}_{\mathrm{space}}
=\sqrt{\operatorname{mean}(\Delta x^2+\Delta y^2)}.
$$

Это разные величины. Порог `--target-rmse` относится к покомпонентной ошибке на проверочных траекториях.

## Обучение

```powershell
python -m scripts.train_intention_xy_decoder `
  --checkpoint checkpoints/antmaze-medium-navigate-v0 `
  --output-dir artifacts/intention_xy_decoder_deep `
  --seed 0 `
  --max-samples 300000 `
  --encoding-batch-size 4096 `
  --hidden-dims 512 512 512 `
  --batch-size 1024 `
  --learning-rate 0.0003 `
  --max-epochs 500 `
  --patience 50 `
  --warmup-epochs 5 `
  --gradient-clip-norm 1.0 `
  --weight-decay 0.00001 `
  --target-rmse 0.3
```

`--max-samples` ограничивает число состояний, `--encoding-batch-size` — блок вычисления `B(s)`, `--hidden-dims` задаёт размеры слоёв, `--batch-size` — размер обучающего блока. `--learning-rate` определяет шаг обновления, `--max-epochs` — предел числа проходов, `--patience` — раннюю остановку, `--warmup-epochs` — плавный разгон обучения. `--gradient-clip-norm` ограничивает градиент, `--weight-decay` регулирует веса, `--target-rmse` задаёт желаемую ошибку координаты.

Сохраняются `decoder.npz`, `decoder_config.json`, `metrics.json` и `training_history.json`. Полные траектории заранее распределяются между обучением, проверкой и окончательным тестом, чтобы соседние состояния одной траектории не попали одновременно в разные выборки.

## Применение в Python

```python
from probes.intention_xy import IntentionXYDecoder

decoder = IntentionXYDecoder.load("artifacts/intention_xy_decoder_deep")
predicted_xy = decoder.predict(z)
```

Декодер обучен на `normalize(B(s))`. Представление произвольной награды или необработанный выход высокоуровневой политики могут лежать вне этого распределения; их интерпретацию нужно проверять отдельно.

## Визуализация

Одно состояние из офлайн-набора:

```powershell
python -m scripts.plot_intention_xy_prediction `
  --decoder artifacts/intention_xy_decoder_deep `
  --dataset train `
  --dataset-index 12345 `
  --output-dir artifacts/intention_xy_decoder_deep/diagnostics `
  --show
```

`--dataset` выбирает часть набора, `--dataset-index` — точное состояние, `--output-dir` — каталог изображений, `--show` открывает окно графика. Без `--dataset-index` состояние выбирается по `--seed`.

Намерения вдоль сохранённого маршрута:

```powershell
python -m scripts.plot_rollout_intentions `
  --trajectory ПУТЬ_К_ЭПИЗОДУ/trajectory.npz `
  --decoder artifacts/intention_xy_decoder_deep `
  --step-size 10 `
  --show
```

`--step-size 10` рисует каждое десятое намерение. Скрипт не меняет сохранённую траекторию и не нормализует намерения повторно.

Проверка: `python -m pytest -q tests/test_intention_xy_decoder.py`.
