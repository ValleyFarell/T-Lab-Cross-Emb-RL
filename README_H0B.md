# H0-B: adaptive switching depth

H0-B extends H0 without replacing it. At each replanning step it evaluates:

```text
depth 1: s -> w -> g
depth 2: s -> w1 -> w2 -> g
```

and executes the first subgoal of the larger estimated value.

## Objective

The implemented decision is:

```text
best_V1 = max_w       V1(s, w, g)
best_V2 = max_w1,w2   V2(s, w1, w2, g)
depth   = 1 if best_V1 >= best_V2 else 2
```

For depth 1 the executed intention is `normalize(B(w))`. For depth 2 it is
`normalize(B(w1))`. The controller then replans according to
`--h0-replan-interval`.

Both estimates use the same raw reward latent for dot products and the same
normalized goal intention only as an input to `F`. They are compared on the
same absolute-value scale. Subtracting the direct value `V_pi_g(s)` would not
change the result because it is the same constant for both depths.

Exact ties prefer depth 1. No complexity penalty is otherwise used.

## What is and is not guaranteed

Every depth-1 candidate is explicitly embedded on the diagonal of the depth-2
matrix: `V2(w, w) := V1(w)`. Therefore the estimated plan classes are nested
and the following inequality is guaranteed inside the planner:

```text
best_V2 >= best_V1
selected_estimate = max(best_V1, best_V2) = best_V2
```

The separate depth decision is still meaningful: an exact tie selects depth 1;
depth 2 is selected only if a non-fallback pair gives a strict estimated
improvement. The inequality concerns maxima over the candidate classes, not an
arbitrary pointwise comparison between one candidate and one pair.

This does not prove that the actual return or success probability is larger.
The estimates still use learned `F/B`, clipped `eta`, a finite candidate subset,
and receding-horizon execution.

H0-B chooses only between depths 1 and 2. It does not choose direct depth 0.
That must be added explicitly if the scientific question is whether switching
is needed at all.

## Run

```powershell
python -m scripts.run_h0b `
  --start-xy 0 0 `
  --goal-xy 4 4 `
  --environment-seed 0 `
  --controller-seed 0 `
  --temperature 0 `
  --max-candidates 64 `
  --pair-batch-size 4096 `
  --h0-replan-interval 1 `
  --results-dir results_h0b
```

Equivalent generic launch:

```powershell
python -m scripts.run_baseline --controller h0b --task-id 1
```

## Paired comparison

Compare `baseline`, `h0`, and `h0b` on exactly the same scenario and
environment seeds. Keep the candidate source, `max_candidates`, eta threshold,
temperature, and checkpoint fixed.

The primary H0-B-specific diagnostics in `trajectory.npz` are:

```text
diagnostic_h0b_selected_depth
diagnostic_h0b_selected_value
diagnostic_h0b_selected_advantage_over_direct
diagnostic_h0b_best_v1
diagnostic_h0b_best_v2
diagnostic_h0b_v2_minus_v1
diagnostic_h0b_replanned
```

Useful additional results are the frequency of depth 2, depth-conditioned
success, and whether the selected depth changes near the goal. These are
descriptive diagnostics; they do not replace paired episode-level metrics.
