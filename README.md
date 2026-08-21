# FB π-Switch experimental stand

Прозрачный экспериментальный стенд для исследования high-level планирования поверх
замороженных Forward-Backward representations. Основная среда —
`antmaze-medium-navigate-v0` из OGBench.

В текущем состоянии репозиторий:

- восстанавливает `F`, `B`, low-level policy и single-intention high actor из checkpoint;
- воспроизводит официальный FB π-Switch baseline без переобучения весов;
- запускает официальные и пользовательские start/goal-сценарии;
- раздельно контролирует случайность среды и контроллера;
- сохраняет полную траекторию и метаданные каждого запуска;
- агрегирует результаты и выполняет paired-сравнение методов на одинаковых seeds.

Новые high-level методы должны реализовывать интерфейс `HighLevelController`. Они не
должны менять checkpoint, low-level policy или набор сценариев baseline.

## Структура

```text
agents/                 исходные реализации агентов
baseline/               загрузка checkpoint и кодирование downstream task
checkpoints/             flags.json и params.pkl
controllers/            единый интерфейс high-level контроллеров
evaluation/             runner, логирование, агрегация, paired-анализ
scripts/                команды запуска и анализа
tests/                  проверки checkpoint, baseline и служебного кода
utils/                  среды, датасеты и сети исходного проекта
```

## Требования

- Windows 10/11 или Linux;
- Python 3.11;
- 64-bit Python;
- достаточно места для OGBench dataset;
- checkpoint в `checkpoints/antmaze-medium-navigate-v0/`.

## Установка на Windows PowerShell

Из корня проекта:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если для рабочего окружения создан `requirements-lock.txt`, для воспроизводимых
экспериментов используйте его вместо свободных диапазонов:

```powershell
python -m pip install -r requirements-lock.txt
```

Проверьте импорт основных компонентов:

```powershell
python -c "import jax, mujoco, ogbench; print('JAX', jax.__version__); print('MuJoCo', mujoco.__version__)"
```

### Windows Application Control и MuJoCo

Ошибка вида

```text
OSError: [WinError 4551] Application Control policy has blocked this file
```

возникает до выполнения кода стенда: Windows блокирует нативную DLL из пакета
MuJoCo. Сначала используйте версию MuJoCo из зафиксированного рабочего окружения.

Для `antmaze-medium-navigate-v0` дополнительные MuJoCo SDF-плагины не используются.
Если политика блокирует только `mujoco\plugin\sdf_plugin.dll`, её можно обратимо
исключить из автозагрузки:

```powershell
Rename-Item `
  .\.venv\Lib\site-packages\mujoco\plugin\sdf_plugin.dll `
  sdf_plugin.dll.disabled
```

Если блокируется основной `mujoco.dll` или Python extension, не отключайте защиту
Windows целиком. Нужен allow rule от администратора либо другое разрешённое
окружение.

## Тесты

Полная проверка:

```powershell
python -m pytest -q
```

Ключевые проверки baseline:

```powershell
python -m pytest -q `
  tests/test_checkpoint.py `
  tests/test_baseline_equivalence.py
```

Тест `test_temperature_zero_matches_official_sample_actions` подтверждает, что при
`temperature=0` обёртка baseline возвращает те же действия, что исходный агент.

## Один официальный baseline-эпизод

```powershell
python -m scripts.run_baseline `
  --task-id 1 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_baseline
```

`task_id` задаёт официальную OGBench пару:

| Task | Номинальный start `(x, y)` | Goal `(x, y)` |
|---:|---:|---:|
| 1 | `(0, 0)` | `(20, 20)` |
| 2 | `(0, 20)` | `(20, 0)` |
| 3 | `(8, 16)` | `(4, 12)` |
| 4 | `(16, 20)` | `(0, 20)` |
| 5 | `(20, 4)` | `(0, 0)` |

Цель фиксирована точно. К стартовому центру OGBench добавляет seed-зависимый шум до
`±1` по каждой координате. Это часть benchmark: одинаковый `environment_seed`
создаёт одинаковое фактическое начальное состояние для всех методов.

## Запуск официальной сетки задач и seeds

Через готовый скрипт:

```powershell
python -m scripts.evaluate_baseline `
  --tasks 1 2 3 4 5 `
  --seeds 0 1 2 3 4 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_baseline `
  --log-file results_baseline/evaluation_commands.json
```

Эквивалентный PowerShell-цикл:

```powershell
foreach ($task in 1..5) {
  foreach ($seed in 0..4) {
    python -m scripts.run_baseline `
      --task-id $task `
      --environment-seed $seed `
      --controller-seed 0 `
      --temperature 0 `
      --results-dir results_baseline
  }
}
```

Для итогового вывода используйте больше seeds, чем для smoke test. Набор seeds
фиксируется до просмотра результатов нового метода и не меняется между методами.

## Пользовательские start/goal-сценарии

Пример:

```powershell
python -m scripts.run_baseline `
  --start-xy 0 0 `
  --goal-xy 20 20 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_custom_baseline
```

`--task-id` нельзя совмещать с `--start-xy/--goal-xy`. Обе пользовательские
координаты обязательны.

Координаты должны быть центрами свободных ячеек. Для medium maze:

```text
x = 4 * j - 4
y = 4 * i - 4
```

Значения координат центров принадлежат `{0, 4, 8, 12, 16, 20}`, но некоторые
комбинации являются стенами. Скрипт проверяет границы, центр ячейки и стену до
запуска дорогого эпизода.

Свободные `x` для каждой строки:

| `y` | Допустимые `x` |
|---:|---|
| 0 | 0, 4, 16, 20 |
| 4 | 0, 4, 12, 16, 20 |
| 8 | 4, 8, 12 |
| 12 | 0, 4, 12, 16, 20 |
| 16 | 0, 8, 12, 20 |
| 20 | 0, 4, 8, 16, 20 |

Для пользовательской цели стенд не переиспользует latent от ближайшего
официального task. Он заново:

1. устанавливает цель в отдельной latent-среде;
2. размечает фиксированные первые `N=100000` offline-состояний относительно цели;
3. вызывает исходный `infer_latent`;
4. прекращает запуск, если `N_g=0`.

Это сохраняет zero-shot постановку: веса не обучаются и дополнительное
взаимодействие со средой для обучения не используется.

Несколько seeds для пользовательской пары:

```powershell
foreach ($seed in 0..9) {
  python -m scripts.run_baseline `
    --start-xy 0 0 `
    --goal-xy 20 20 `
    --environment-seed $seed `
    --controller-seed 0 `
    --temperature 0 `
    --results-dir results_custom_baseline
}
```

Пользовательские сценарии являются дополнительной диагностикой. Основной результат
следует сообщать на официальных OGBench tasks. Список дополнительных сценариев
фиксируется до сравнения методов, иначе возникает риск cherry-picking.

## Случайность и воспроизводимость

Используются два независимых seed:

- `environment_seed` управляет reset среды, стартовым шумом и NumPy/Python RNG;
- `controller_seed` создаёт отдельный JAX PRNG key для high- и low-level policy.

При `temperature=0` baseline policy детерминирована, но `controller_seed` всё равно
сохраняется для единого протокола с будущими стохастическими контроллерами.

Корректная проверка воспроизводимости:

1. запустить один и тот же сценарий дважды в разные каталоги;
2. сравнить `success`, `steps`, `path_length`, `final_distance`;
3. при необходимости сравнить массивы в `trajectory.npz`;
4. не сравнивать `duration_s`, поскольку время зависит от JAX-компиляции и нагрузки.

Для объективного сравнения candidate обязан получить те же:

```text
scenario_id
task_id или start_ij/goal_ij
environment_seed
controller_seed
temperature
```

## Сохранённые результаты

Каждый запуск создаёт:

```text
results_dir/
  baseline_task_1/
    config.json
    runs/
      000001/
        scenario.json
        summary.json
        trajectory.npz
        path.png
```

Для custom-сценария имя содержит start и goal. Например:

```text
baseline_custom-0_0-to-20_20/
```

Файлы:

- `scenario.json` — задача, клетки, seeds и temperature;
- `summary.json` — метод, фактические start/goal и итоговые метрики;
- `trajectory.npz` — observations, positions, actions, intentions и diagnostics;
- `path.png` — путь агента на карте;
- `config.json` — общая конфигурация серии запусков.

## Обычная агрегация

```powershell
python -m scripts.evaluate_results --results-dir results_baseline
```

Создаются:

```text
results_baseline/summary.json
results_baseline/summary.csv
```

Агрегация сообщает overall и per-task показатели. Она удобна для описания одного
метода, но сама по себе не учитывает связь результатов по одинаковым seeds.

## Paired-анализ baseline против candidate

Candidate должен сохранять результаты через тот же `EpisodeRunner` и
`save_episode_result`. После запуска обоих методов:

```powershell
python -m scripts.paired_analysis `
  --baseline-dir results_baseline `
  --candidate-dir results_candidate `
  --output comparisons/candidate_vs_baseline.json `
  --bootstrap-seed 0 `
  --bootstrap-samples 10000
```

Пара образуется только при полном совпадении:

```text
(scenario_id, task_id, start_ij, goal_ij,
 environment_seed, controller_seed, temperature)
```

Повторный запуск одного ключа внутри каталога одного метода считается дубликатом и
вызывает ошибку. Повторы необходимо хранить в отдельном каталоге, а не считать
независимыми seeds.

Отчёт содержит:

- success rate каждого метода и paired-разность;
- bootstrap 95% CI для paired-разности;
- число `baseline-only` и `candidate-only` успехов;
- exact McNemar p-value по discordant-парам;
- разности steps и path length только для пар, где успешны оба метода;
- разность final distance по всем парам;
- overall и per-scenario результаты;
- список unmatched запусков и сырые пары для аудита.

Все разности определены как:

```text
candidate - baseline
```

Поэтому:

- положительная `success_rate_delta` лучше для candidate;
- отрицательные `steps_delta_both_success` и
  `path_length_delta_both_success` лучше для candidate;
- отрицательная `final_distance_delta_all` лучше для candidate.

McNemar p-value и bootstrap CI не заменяют анализ поведения. При малом числе seeds
они имеют низкую статистическую мощность; это нужно явно указать в отчёте.

## Рекомендуемый экспериментальный протокол

1. Зафиксировать checkpoint, dependency lock и git commit.
2. Заранее определить официальные tasks, custom-сценарии и seeds.
3. Запустить baseline на полном наборе сценариев.
4. Не менять baseline-результаты после просмотра candidate.
5. Запустить candidate на тех же ключах сценариев.
6. Проверить отсутствие unmatched и duplicate runs.
7. Выполнить обычную агрегацию каждого метода.
8. Выполнить paired-анализ.
9. Исследовать траектории успехов, провалов и discordant-пар.
10. В отчёте отделить заранее заявленные метрики от exploratory-наблюдений.

Primary metric рекомендуется зафиксировать как success rate. Steps и path length
следует считать secondary metrics среди совместных успехов: сравнение только
успешных эпизодов имеет selection bias, поэтому оно не должно заменять primary
метрику.

## Добавление нового high-level контроллера

Контроллер наследуется от `controllers.base.HighLevelController` и реализует:

```python
def select_intention(
    self,
    observation,
    task_latent,
    *,
    rng,
    temperature,
):
    ...
```

Метод возвращает `IntentionSelection`. `EpisodeRunner` отвечает за seeds, вызов
low-level policy, завершение эпизода и логирование. Это не позволяет каждому
методу незаметно менять экспериментальный протокол.

## Ограничения

- Веса FB и политик считаются замороженными.
- Custom goal должен иметь `N_g > 0` в фиксированном zero-shot batch.
- Custom XY ограничены центрами свободных ячеек medium maze.
- Старт сохраняет официальный seed-зависимый шум OGBench.
- Пять seeds достаточны для проверки кода, но обычно недостаточны для сильного
  научного вывода.
- `duration_s` включает системный шум и JAX-компиляцию и не является основной
  метрикой эффективности метода.
- Paired-анализ показывает ассоциацию на выбранном наборе задач; он не доказывает
  перенос на другие лабиринты.

## Источники

- Stojanovic & Proutiere, *Switching Successor Measures for Hierarchical Zero-Shot Reinforcement Learning*.
- Touati & Ollivier, *Learning One Representation to Optimize All Rewards*.
- Park et al., *OGBench: Benchmarking Offline Goal-Conditioned RL*.
- [OGBench repository](https://github.com/seohongpark/ogbench)
- [Switching Successor Measures repository](https://github.com/stestoKTH/switching-successor-measures)

