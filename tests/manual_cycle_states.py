"""Stage 2 review helper: manually cycles the avatar through all 4
engagement states (without running a real episode -- real transitions are
stochastic and might not hit every state), so the indicator-orb color and
posture tween can be watched directly for each one.

Sets `env.engagement_state` directly each phase; the renderer detects the
change and tweens exactly as it would from a real env.step() (see
AdaptLearnRenderer._current_mood).

Run with:
    uv run python tests/manual_cycle_states.py
"""

from __future__ import annotations

import time

import glfw

from environment.custom_env import AdaptLearnEnv, EngagementState

HOLD_SECONDS = 2.5  # time to hold each state so the tween + settled pose are both watchable

SEQUENCE = [
    EngagementState.ENGAGED,
    EngagementState.FRUSTRATED,
    EngagementState.BORED,
    EngagementState.CONFUSED,
    EngagementState.ENGAGED,
]


def main() -> None:
    env = AdaptLearnEnv(render_mode="human")
    env.reset(seed=0)
    renderer = env._renderer
    print("Window open. Cycling: " + " -> ".join(s.name for s in SEQUENCE))

    for state in SEQUENCE:
        print(f"  -> {state.name}")
        env.engagement_state = state
        start = time.time()
        while time.time() - start < HOLD_SECONDS:
            if glfw.window_should_close(renderer.window):
                env.close()
                return
            env.render()

    print("Cycle complete -- window stays open, close it to exit.")
    while not glfw.window_should_close(renderer.window):
        env.render()

    env.close()


if __name__ == "__main__":
    main()
