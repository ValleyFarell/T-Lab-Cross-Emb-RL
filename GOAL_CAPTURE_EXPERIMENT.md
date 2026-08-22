# Диагностика точного захвата цели

Скрипт `scripts/diagnose_goal_capture.py` локализует причину петель около цели.
Он находит в сохранённых траекториях первое полное наблюдение Ant в кольце
`0.5 < distance < 1.0`, восстанавливает из него `qpos` и `qvel`, после чего
запускает несколько продолжений из идентичного физического состояния.

Сравниваются три типа управляющего латента:

- `task_raw` — исходный reward/task latent;
- `task_normalized` — тот же latent после официального `normalize_z`;
- `goal_state_XX` — `normalize(B(s_goal))` для реальных состояний offline-датасета
  внутри радиуса цели `0.5`.

## Запуск в PowerShell

В качестве `--source-results` укажите корень эксперимента, в котором лежат
папки запусков с `trajectory.npz` и `scenario.json`:

```powershell
python -m scripts.diagnose_goal_capture `
  --source-results results_raw `
  --goal-xy 4 4 `
  --goal-state-count 8 `
  --horizon 100 `
  --temperature 0 `
  --output-dir results_goal_capture
```

Для быстрой проверки на двух исходных состояниях:

```powershell
python -m scripts.diagnose_goal_capture `
  --source-results results_raw `
  --goal-xy 4 4 `
  --goal-state-count 3 `
  --horizon 50 `
  --max-source-runs 2 `
  --output-dir results_goal_capture_smoke
```

Важно: координаты `--goal-xy` должны совпадать с целью исходного эксперимента.
Скрипт намеренно не доверяет нарисованной звезде: он заново размечает
zero-shot dataset относительно переданной цели и записывает её в среду перед
каждой веткой.

## Результаты

- `config.json` — параметры, нормы латентов и происхождение goal-state latent;
- `branches.csv` — одна строка на исходное состояние и управляющий латент;
- `aggregate.csv` / `aggregate.json` — сводка по каждому варианту и всей семье
  `goal_state`;
- `branches/<run>/<condition>/summary.json` — метрики одной ветки;
- `branches/<run>/<condition>/trajectory.npz` — полная траектория, включая
  конечный `next_observation`.

Основные поля:

| Поле | Смысл |
| --- | --- |
| `hit_rate` | Доля веток, попавших в радиус `0.5` |
| `minimum_distance` | Минимальная дистанция до центра цели |
| `mean_radial_velocity` | Скорость к центру; положительная направлена внутрь |
| `mean_tangential_speed` | Модуль касательной скорости |
| `radius_exits` | Число выходов обратно за радиус `1.0` |
| `fell_below_0_3` | Диагностика физического падения Ant |

## Интерпретация

- Если большинство `goal_state_XX` надёжно попадает в `0.5`, а оба task latent
  нет, проблема в усреднённом reward-latent как интерфейсе low-level policy.
- Если task latent петляет, но отдельные `goal_state_XX` заметно различаются,
  важна ориентация, скорость или фаза шага целевого состояния.
- Если все варианты имеют низкую радиальную и высокую касательную скорость и
  часто выходят за `1.0`, проблема находится в самой low-level policy и её
  обучающей цели.
- Строка `family,goal_state` объединяет несколько интервенций на тех же
  исходных состояниях. Их нельзя считать независимыми environment seeds или
  использовать как искусственное увеличение размера выборки для p-value.
