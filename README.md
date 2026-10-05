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

![AdaptLearn-v1 classroom renderer](assets/classroom_renderer.png)

**Demo video:** https://youtu.be/ECvT6zNF19Y

See `environment/custom_env.py`'s module docstring for the full design
rationale (the theory behind the transition model, and every judgment call
made where the assignment brief left something open).

## Quickstart

```bash
uv sync --extra dev              # install dependencies, including pytest (plain `uv sync` skips it)
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
  compare_algorithms.py         Cross-algorithm comparison plots + summary table
  convergence_analysis.py       Episodes-to-converge plot (peak / 90%-threshold marker), all four algorithms
  cumulative_reward_subplots.py Reward-over-training subplots, all four best runs
  generalization_plot.py        Bar-chart visualization of the generalization test results
  pg_entropy_combined.py        Combined PPO / A2C / REINFORCE entropy plot
  __init__.py        Shared training/plotting utilities used by all of the above
tests/
  test_*.py          Automated tests (reward values, terminal conditions, spaces, transitions, reproducibility)
  manual_*.py        Manual/visual diagnostic scripts (not part of the pytest suite)
  manual_policy_inspection.py   Action-frequency / state-conditioned cross-tab for a trained model
  generalization_test.py        Held-out-seed and edge-case stress test, all four algorithms
  stability_analysis.py         Reruns the four best runs with objective-loss / entropy logging enabled
models/, logs/       Per-run trained models, monitor logs, summary CSVs, best_run.json, plots
  logs/comparison/   Cross-algorithm comparison, convergence, and cumulative-reward plots
  logs/generalization/   Held-out-seed and edge-case results, plus the comparison plot
  logs/stability/    DQN loss / PG entropy curves (CSVs and plots) for the four best runs
  models/stability/  Models retrained with stability logging enabled, kept separate from the main 48-run sweep
api/                 Bonus: FastAPI wrapper exposing the environment as a JSON API (see below)
```

## Bonus: JSON API for frontend integration

`api/` is a minimal FastAPI wrapper demonstrating that AdaptLearn-v1 can be
driven entirely over JSON -- `POST /reset`, `POST /step` (the trained model
picks its own action if none is supplied), and `GET /episode-summary` -- so
a web or mobile frontend could integrate with it without ever running
Python, Gymnasium, or PyTorch itself.

```bash
uv run uvicorn api.app:app --reload   # then open http://127.0.0.1:8000/docs
```

`POST /step`, called with an empty body, showing the trained model choosing its own action live through Swagger UI:

![Swagger UI: POST /step](api/screenshots/swagger_step.png)

Full endpoint docs, more request/response examples, and the `/reset`
and `/episode-summary` screenshots: see [`api/README.md`](api/README.md).

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

## Known limitations

- **CONFUSED handling is a shared blind spot across algorithms.** DQN, PPO,
  and A2C all converge on responses to the learner's CONFUSED state that
  don't mechanically resolve it in the environment's own transition model
  (only `offer_hint`, `repeat_simplified`, or `change_task_type` do). This
  independent convergence across three structurally different algorithms
  points to CONFUSED's low occurrence frequency and comparatively weak
  reward signal, rather than a weakness in any one algorithm — see the
  report's Behavioural Policy Analysis and Reward Structure Iteration
  sections for the full investigation, including a targeted reward fix
  that was tried and reverted after it destabilized three of the four
  algorithms' broader policies.
- **REINFORCE's `seed` argument does not fully control reproducibility.**
  `training/pg_training.py`'s training loop calls `env.reset()` without a
  seed, so the environment's own stochasticity is not fixed across runs
  even when REINFORCE's `seed` (which only seeds network initialization)
  is held constant. Documented in the report as a scope limitation for
  REINFORCE's results specifically; DQN/PPO/A2C are unaffected since
  Stable-Baselines3 seeds the environment internally.
