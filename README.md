# AdaptLearn-v1

A custom Gymnasium reinforcement learning environment simulating an AI tutoring
agent that must learn pedagogical actions to keep a simulated learner in a
state of productive engagement ("flow"). The agent chooses one of 8
pedagogical actions per step (raise/lower difficulty, offer a hint, give
encouragement, change task type, ...); a simulated learner reacts and
transitions between four engagement states (FRUSTRATED, BORED, CONFUSED,
ENGAGED). Nothing about the tutoring policy is hand-coded — it's learned
purely from reward feedback.

Four algorithms are trained and compared on it: **DQN**, **PPO**, **A2C**
(via Stable-Baselines3), and a from-scratch **REINFORCE** implementation.
A PyOpenGL renderer visualizes a trained agent's episodes as a small
classroom scene (seated learner, blackboard, desk, difficulty gauge, HUD).

See `environment/custom_env.py`'s module docstring for the full design
rationale (the theory behind the transition model, and every judgment call
made where the assignment brief left something open).

## Quickstart

```bash
uv sync                          # install dependencies
uv run pytest                    # run the test suite (headless, no rendering)
uv run python main.py demo       # watch the best trained agent in a live window
```

## `main.py` commands

`main.py` is the single entry point tying the project together — every
subcommand is a thin wrapper over an existing module (`training/dqn_training.py`,
`training/pg_training.py`, `training/compare_algorithms.py`,
`environment/rendering.py`); it contains no training/rendering logic itself.

```bash
uv run python main.py train --algo all           # run all four 12-run hyperparameter sweeps
uv run python main.py train --algo dqn --quick   # fast smoke test (a few configs, 2000 timesteps)
uv run python main.py compare                    # regenerate cross-algorithm comparison plots (logs/comparison/)
uv run python main.py demo                       # render the best model overall, live, real-time-paced
uv run python main.py demo --algo ppo            # render PPO's best run specifically
uv run python main.py demo --manual              # spacebar-advance pacing instead of fixed timing, for recording
uv run python main.py demo --seed 3
```

Rendering requires a display (GLFW window) and is skipped entirely during
training/tests — `AdaptLearnEnv(render_mode=None)`, the default, never touches
PyOpenGL/GLFW.

## Project layout

```
environment/
  custom_env.py     Gymnasium env: transition model, reward, observation/action spaces
  rendering.py       PyOpenGL classroom-scene renderer (human / human_manual / rgb_array)
training/
  dqn_training.py    DQN hyperparameter sweep (12 configs)
  pg_training.py     PPO / A2C / REINFORCE hyperparameter sweeps (12 configs each)
  compare_algorithms.py   Cross-algorithm comparison plots + summary table
  __init__.py        Shared training/plotting utilities used by all of the above
tests/
  test_*.py          Automated tests (reward values, terminal conditions, spaces, transitions, reproducibility)
  manual_*.py        Manual/visual diagnostic scripts (not part of the pytest suite)
  manual_policy_inspection.py   Action-frequency / state-conditioned cross-tab for a trained model
models/, logs/       Per-run trained models, monitor logs, summary CSVs, best_run.json, plots
```

## Current best results (20,000 timesteps/run, 12-run sweep each)

| Algorithm | Best run | Best final_mean_reward | Sweep mean ± std |
|---|---|---|---|
| DQN | `dqn_02` | 9.66 | 6.45 ± 2.14 |
| PPO | `ppo_11` | 7.67 | 4.13 ± 2.65 |
| A2C | `a2c_07` | 5.33 | 2.84 ± 1.77 |
| REINFORCE | `reinforce_00` | 3.08 | 0.60 ± 1.54 |

Regenerate this table and its plots with `uv run python main.py compare`
(writes to `logs/comparison/`). Full per-algorithm policy-inspection
findings (which actions each trained agent actually uses per engagement
state, and where each falls short of "sensible tutor" behavior) are not
duplicated here — see the report.

## Notes to fold into the report (not yet written up)

- **DQN CONFUSED-response variance.** After reverting the targeted CONFUSED
  reward bonus (see `RewardConfig`'s docstring in `environment/custom_env.py`
  for why it was tried and reverted), the new best DQN run (`dqn_02`) was
  re-inspected and its CONFUSED handling did not reproduce cleanly: it now
  resolves CONFUSED mostly via `advance_topic` (8/10 occurrences across the
  10-episode inspection) rather than `decrease_difficulty` (1/10), which is
  the opposite of what was previously documented for this exact model/state.
  Aggregate reward matched the historical value almost exactly (9.6585 vs.
  the previously reported ~9.66), so this looks less like a bad revert and
  more like CONFUSED being a low-frequency, high-variance state for this
  policy (~10 occurrences total in the eval batch) rather than a robustly
  "camped" behavior in either direction. Worth stating as an explicit
  limitation on the CONFUSED-handling comparison in the report, rather than
  treating either reading as ground truth.
