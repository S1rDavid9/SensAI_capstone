"""Stage 1 review helper: opens the static classroom scene and holds the
window open (no episode stepping -- nothing animates yet at this stage) so
it can be inspected at leisure. Close the window to exit.

Run with:
    uv run python tests/manual_view_scene.py
"""

from __future__ import annotations

import glfw

from environment.custom_env import AdaptLearnEnv


def main() -> None:
    env = AdaptLearnEnv(render_mode="human")
    env.reset(seed=0)
    print("Window open -- close it to exit.")

    renderer = env._renderer
    while not glfw.window_should_close(renderer.window):
        env.render()

    env.close()


if __name__ == "__main__":
    main()
