# H0 Two-Switch addon

Add these files:

- `baseline/two_switch_planner.py`
- `controllers/two_switch.py`

No training changes are required.

## Integration

Where the current code creates:

```python
BaselineController(frozen_fb)
```

replace it with:

```python
from baseline.two_switch_planner import TwoSwitchPlanner
from controllers.two_switch import TwoSwitchController

planner = TwoSwitchPlanner(
    frozen_fb,
    candidate_observations,
)

controller = TwoSwitchController(planner)
```

`candidate_observations` must be real offline dataset states.

The runner remains unchanged because it already consumes the generic
`HighLevelController.select_intention()` interface.

## Important

This is execute-first-and-replan behavior:
- the planner selects `(w1,w2)`;
- only `w2` is converted into the executed low-level intention.

For a literal `w1 -> w2 -> g` execution mode a stateful controller is required.
