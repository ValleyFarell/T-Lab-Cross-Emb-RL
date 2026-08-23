# Проверка подключения H0

H0 уже встроен в проект. Его планировщик отделён от исходного агента и подключается через:

```python
from hypotheses.h0 import TwoSwitchPlanner
from controllers import TwoSwitchController
```

Старый импорт `baseline.two_switch_planner` оставлен ради совместимости. Кандидаты могут поступать только из `train_dataset["observations"]`; использовать проверочные состояния или уже полученные оценочные траектории нельзя.

## Проверка

```powershell
python -m pytest -q
python -m scripts.run_h0 --help
```

Сравнение одного одинакового сценария:

```powershell
python -m scripts.run_baseline `
  --controller baseline `
  --task-id 1 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_baseline

python -m scripts.run_h0 `
  --task-id 1 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_h0
```

`--task-id` задаёт сценарий, оба `seed` фиксируют случайность, `--temperature 0` убирает дополнительную случайность действий, а `--results-dir` разводит серии по отдельным каталогам.

В `config.json` запуска H0 должен появиться раздел `method_config`, а в `trajectory.npz` — поля `diagnostic_h0_score`, `diagnostic_w1_index`, `diagnostic_w2_index`.
