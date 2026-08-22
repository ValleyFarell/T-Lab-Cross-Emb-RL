# H0 integration checklist

Copy the supplied files into the repository while preserving their relative
paths. Existing imports from `baseline.two_switch_planner` remain valid through
the compatibility shim, but new code should use:

```python
from hypotheses.h0 import TwoSwitchPlanner
from controllers import TwoSwitchController
```

The candidate source must remain training data only:

```python
train_dataset["observations"]
```

Do not build candidate sets from validation states or evaluation trajectories.

After copying, run:

```powershell
python -m pytest -q
python -m scripts.run_h0 --help
```

Then use the same scenario and seeds for baseline and H0:

```powershell
python -m scripts.run_baseline --controller baseline --task-id 1 --environment-seed 0 --controller-seed 0 --temperature 0
python -m scripts.run_h0 --task-id 1 --environment-seed 0 --controller-seed 0 --temperature 0
```

Verify that the H0 run's `config.json` contains `method_config` and that its
`trajectory.npz` contains `diagnostic_h0_score`, `diagnostic_w1_index`, and
`diagnostic_w2_index`.

