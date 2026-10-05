"""Stage 3 review helper: manually triggers all 8 action cues in sequence
(without running a real episode), so each one can be watched appear, hold,
and fade on its own.

Mutates env.steps/env.last_action/env.difficulty directly (mirroring how
tests/manual_cycle_states.py mutates env.engagement_state for Stage 2) --
the renderer detects these exactly as it would a real env.step(), via
AdaptLearnRenderer._register_step.

Run with:
    uv run python tests/manual_trigger_actions.py
"""

from __future__ import annotations

import time

import glfw

from environment.custom_env import ACTION_NAMES, AdaptLearnEnv, EngagementState

HOLD_SECONDS = 2.5  # time between triggers, long enough to watch appear -> hold -> fade

# (action_index, difficulty_after) -- difficulty is driven explicitly here
# rather than left to increase/decrease_difficulty's real effect, so the
# gauge's movement is visible regardless of trigger order.
SEQUENCE = [
    (0, 2),  # increase_difficulty
    (0, 3),  # increase_difficulty again -- gauge should keep climbing
    (1, 2),  # decrease_difficulty
    (2, 2),  # change_task_type
    (3, 2),  # offer_hint
    (4, 2),  # add_encouragement
    (5, 2),  # introduce_challenge
    (6, 2),  # repeat_simplified
    (7, 3),  # advance_topic -- also raises difficulty, gauge should move too
]


def main() -> None:
    env = AdaptLearnEnv(render_mode="human")
    env.reset(seed=0)
    env.engagement_state = EngagementState.ENGAGED
    env.difficulty = 1
    renderer = env._renderer
    print("Window open. Triggering each action in turn:")

    for action, difficulty in SEQUENCE:
        print(f"  -> {ACTION_NAMES[action]} (difficulty -> {difficulty})")
        env.steps += 1
        env.last_action = action
        env.difficulty = difficulty
        start = time.time()
        while time.time() - start < HOLD_SECONDS:
            if glfw.window_should_close(renderer.window):
                env.close()
                return
            env.render()

    print("Sequence complete -- window stays open, close it to exit.")
    while not glfw.window_should_close(renderer.window):
        env.render()

    env.close()


if __name__ == "__main__":
    main()
