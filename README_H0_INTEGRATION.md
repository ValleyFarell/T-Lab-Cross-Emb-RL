# H0 integration

## Added files

Copy:

```
baseline/two_switch_planner.py
controllers/two_switch.py
scripts/run_h0.py
```

## Change required

Where the current evaluation creates:

```
BaselineController(frozen_fb)
```

replace it with:

```python
from scripts.run_h0 import make_h0_controller

controller = make_h0_controller(
    frozen_fb,
    train_dataset,
    max_candidates=512,
)
```

No training changes.

No checkpoint changes.

## Candidate source

Candidates must come from offline dataset:

```
train_dataset["observations"]
```

Do not use evaluation trajectories.

## Current execution mode

The planner computes:

```
(s, g) -> (w1, w2)
```

but returns the second-stage intention as the executable intention.

This matches the existing stateless controller interface.

A true stateful execution:

```
s -> w1 -> w2 -> g
```

requires extending EpisodeRunner with switch memory and hit detection.
