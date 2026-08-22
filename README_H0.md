# H0: receding-horizon two-switch planning

H0 is an inference-only addon. It does not change training, checkpoints,
task-latent construction, environment reset, or low-level action sampling.

## Semantics

For the current state `s` and final reward latent `z_g`, the planner scores
candidate pairs `(w1, w2)` from the offline training dataset. It executes the
policy intention of `w1`, then replans:

```text
score (w1, w2) -> execute w1 -> observe next state -> replan
```

`w2` is a look-ahead state used in the two-switch score. It is **not** sent to
the low-level actor in the current step. This is receding-horizon planning, not
a state machine that blindly executes `w1`, then `w2`, then `g`.

The default `--h0-replan-interval 1` replans every environment step. Increasing
it reuses the selected `w1` intention for the requested number of steps.

## Files

- `hypotheses/h0/planner.py` — H0-specific planner;
- `controllers/two_switch.py` — adapter to the common controller interface;
- `scripts/run_h0.py` — factory and runnable launcher;
- `baseline/two_switch_planner.py` — compatibility import only.

This layout keeps H0 outside the baseline implementation. Future hypotheses
should receive their own `hypotheses/hN/` package and controller adapter.

## Run

Official OGBench task:

```powershell
python -m scripts.run_h0 `
  --task-id 1 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_h0
```

Custom start and goal:

```powershell
python -m scripts.run_h0 `
  --start-xy 0 0 `
  --goal-xy 4 4 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --results-dir results_h0
```

The equivalent generic command is:

```powershell
python -m scripts.run_baseline --controller h0 --task-id 1
```

## Performance parameters

Defaults are deliberately modest and transparent:

- `--max-candidates 64`: 4,096 scored pairs instead of 262,144 for 512
  candidates;
- `--pair-batch-size 4096`: caps the pair-dependent forward batch;
- `--h0-replan-interval 1`: exact receding-horizon behavior;
- `--eta-epsilon 1e-6`: rejects unstable denominators.

Candidate latents, `F(w_i, z_i)`, and their self-measures are computed once.
At each replanning step, only the genuinely pair-dependent `F(w1, z2)` values
are evaluated in chunks. The candidate subset is a deterministic linspace
selection, so the same dataset and parameters yield the same candidates.

If 64 candidates are still too slow, first try:

```powershell
python -m scripts.run_h0 --task-id 1 --max-candidates 32
```

Use the same H0 parameters for every paired method comparison.

## Numerical safety

For each `eta = numerator / denominator`:

- non-finite values and `abs(denominator) < eta_epsilon` invalidate the
  candidate or pair;
- valid ratios are clipped to the theoretical range `[0, 1]`;
- invalid pairs receive score `-inf`;
- a rollout stops with a clear error if no finite pair remains.

The trajectory diagnostics record invalid and clipped counts, selected eta
values, H0 score, selected candidate indices, and whether replanning occurred.

## Reproducibility metadata

`config.json` contains `method_config` with:

- source and selected candidate counts;
- maximum candidate count and selection method;
- exact source indices and SHA-256 candidate checksum;
- pair count and pair batch size;
- eta threshold and range;
- replanning interval and execution semantics.

A copy is stored both at experiment level and inside each run directory.

`trajectory.npz` contains every per-step controller diagnostic with the
`diagnostic_` prefix, including:

```text
diagnostic_h0_score
diagnostic_w1_index
diagnostic_w2_index
diagnostic_w1_source_index
diagnostic_w2_source_index
diagnostic_selected_eta1
diagnostic_selected_eta2
diagnostic_eta1_invalid_count
diagnostic_eta2_invalid_count
diagnostic_eta1_clipped_count
diagnostic_eta2_clipped_count
diagnostic_candidate_count
diagnostic_selected_subgoals_equal
diagnostic_h0_replanned
```

## Adding the next hypothesis

Keep the extension boundary small:

1. add `hypotheses/hN/` for hypothesis-specific planning;
2. implement a `HighLevelController` adapter with a unique `method_name`;
3. expose static parameters through `experiment_config()`;
4. return stable per-step numeric fields through `IntentionSelection.diagnostics`;
5. add one CLI choice and one branch in `build_controller()`;
6. add contract tests for score semantics, dispatch, logging, and invalid input.

No changes to `EpisodeRunner`, `save_episode_result`, or `EpisodeLogger` should
be needed.

